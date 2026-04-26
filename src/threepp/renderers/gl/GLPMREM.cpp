
#include "GLPMREM.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::gl;

namespace {

    constexpr int LOD_MIN = 4;
    constexpr int LOD_MAX = 8;
    constexpr int PMREM_SIZE_MAX = 1 << LOD_MAX;                     // 256
    constexpr int TOTAL_LODS = (LOD_MAX - LOD_MIN + 1) + 6;    // 11
    constexpr int PMREM_WIDTH = 3 * PMREM_SIZE_MAX;                  // 768
    constexpr int PMREM_HEIGHT = 3 * PMREM_SIZE_MAX;                 // 768

    // Per-LOD roughness, chosen to match three.js roughnessToMip() inverse:
    // LOD 0 is smoothest (mip=LOD_MAX=8, roughness~0), LOD 10 is roughest
    // (mip=m0=-2, roughness=1.0). `cube_uv_reflection_fragment.glsl` picks
    // the mip for a given material roughness and reads the corresponding
    // tile region via bilinearCubeUV().
    constexpr float LOD_ROUGHNESS[TOTAL_LODS] = {
            0.00f,  // LOD 0  (mip=8)
            0.08f,  // LOD 1  (mip=7)
            0.11f,  // LOD 2  (mip=6)
            0.15f,  // LOD 3  (mip=5)
            0.22f,  // LOD 4  (mip=4)
            0.31f,  // LOD 5  (mip=3, filterInt=1)
            0.40f,  // LOD 6  (mip=2, filterInt=2)
            0.53f,  // LOD 7  (mip=1, filterInt=3)
            0.67f,  // LOD 8  (mip=0, filterInt=4)
            0.80f,  // LOD 9  (mip=-1, filterInt=5)
            1.00f,  // LOD 10 (mip=-2, filterInt=6)
    };

    // Pure port of three.js PMREMGenerator._getCommonVertexShader(). Each vertex
    // carries a face index and a [0,1]² face-UV; getDirection() maps (face, uv) to
    // a world-space cube direction that the fragment shader samples with.
    const char* const VERTEX_SRC = R"(#version 330 core
in vec3 position;
in vec2 uv;
in float faceIndex;
out vec3 vOutputDirection;

vec3 getDirection(vec2 uv, float face) {
    uv = 2.0 * uv - 1.0;
    vec3 direction = vec3(uv, 1.0);
    if (face == 0.0) {
        direction = direction.zyx;
    } else if (face == 1.0) {
        direction = direction.xzy;
        direction.xz *= -1.0;
    } else if (face == 2.0) {
        direction.x *= -1.0;
    } else if (face == 3.0) {
        direction = direction.zyx;
        direction.xz *= -1.0;
    } else if (face == 4.0) {
        direction = direction.xzy;
        direction.xy *= -1.0;
    } else if (face == 5.0) {
        direction.z *= -1.0;
    }
    return direction;
}

void main() {
    vOutputDirection = getDirection(uv, faceIndex);
    gl_Position = vec4(position, 1.0);
}
)";

    // Pure port of three.js PMREMGenerator's equirect→cube fragment shader
    // (_getEquirectShader). Direct, non-convolved sample — produces a sharp
    // reflection at every LOD for now. GGX blur lives in three.js's _halfBlur
    // pass, which can be added later without changing the atlas layout or
    // consumer shader (cube_uv_reflection_fragment.glsl).
    const char* const FRAGMENT_SRC = R"(#version 330 core
precision highp float;
precision highp int;

in vec3 vOutputDirection;
out vec4 fragColor;

uniform sampler2D envMap;

#define RECIPROCAL_PI 0.31830988618
#define RECIPROCAL_PI2 0.15915494

vec2 equirectUv(in vec3 dir) {
    float u = atan(dir.z, dir.x) * RECIPROCAL_PI2 + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * RECIPROCAL_PI + 0.5;
    return vec2(u, v);
}

void main() {
    vec3 outputDirection = normalize(vOutputDirection);
    vec2 uv = equirectUv(outputDirection);
    fragColor = vec4(texture(envMap, uv).rgb, 1.0);
}
)";

    // Build geometry for one LOD. UVs are extended by half-texel on each side so
    // that pixel centres in the rendered tile align exactly with the face-local
    // pixel grid that `bilinearCubeUV()` reads (which treats the tile as a
    // [0..faceSize-1] integer grid, not as a [0,1] texture). See three.js
    // PMREMGenerator._createPlanes — sizeLod-1 is critical.
    std::shared_ptr<BufferGeometry> createLodPlane(int sizeLod) {
        const float texelSize = 1.0f / static_cast<float>(sizeLod - 1);
        const float uvMin = -texelSize * 0.5f;
        const float uvMax = 1.0f + texelSize * 0.5f;

        std::vector<float> positions;
        std::vector<float> uvs;
        std::vector<float> faceIndices;
        positions.reserve(6 * 6 * 3);
        uvs.reserve(6 * 6 * 2);
        faceIndices.reserve(6 * 6);

        for (int face = 0; face < 6; ++face) {
            const float x0 = static_cast<float>(face % 3) * (2.0f / 3.0f) - 1.0f;
            const float x1 = x0 + (2.0f / 3.0f);
            const float y0 = face > 2 ? 0.0f : -1.0f;
            const float y1 = y0 + 1.0f;

            const float pos[18] = {
                    x0, y0, 0, x1, y0, 0, x1, y1, 0,
                    x0, y0, 0, x1, y1, 0, x0, y1, 0};
            const float uv[12] = {
                    uvMin, uvMin, uvMax, uvMin, uvMax, uvMax,
                    uvMin, uvMin, uvMax, uvMax, uvMin, uvMax};

            for (float p : pos) positions.push_back(p);
            for (float u : uv) uvs.push_back(u);
            for (int i = 0; i < 6; ++i) faceIndices.push_back(static_cast<float>(face));
        }

        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        geom->setAttribute("faceIndex", FloatBufferAttribute::create(faceIndices, 1));
        return geom;
    }

}// namespace

struct GLPMREM::Impl {
    std::shared_ptr<RawShaderMaterial> material;
    std::shared_ptr<OrthographicCamera> camera;
};

GLPMREM::GLPMREM(GLRenderer& r)
    : renderer(r), impl(std::make_unique<Impl>()) {

    impl->material = RawShaderMaterial::create();
    impl->material->name = "PMREM.equirectGGX";
    impl->material->vertexShader = VERTEX_SRC;
    impl->material->fragmentShader = FRAGMENT_SRC;
    impl->material->uniforms["envMap"] = Uniform();
    impl->material->uniforms["roughness"] = Uniform();
    impl->material->uniforms["roughness"].setValue(0.0f);
    impl->material->uniforms["numSamples"] = Uniform();
    impl->material->uniforms["numSamples"].setValue(128);
    impl->material->depthTest = false;
    impl->material->depthWrite = false;
    impl->material->blending = Blending::None;
    impl->material->side = Side::Double;

    impl->camera = OrthographicCamera::create();
}

GLPMREM::~GLPMREM() = default;

std::unique_ptr<RenderTarget> GLPMREM::fromEquirectangular(Texture& equirect) {

    RenderTarget::Options options;
    options.type = Type::HalfFloat;
    options.format = Format::RGBA;
    options.encoding = ColorSpace::Linear;
    // NearestFilter — cube_uv_reflection_fragment.glsl does its own manual
    // 4-tap bilinear via 4 textureLod reads with +texelSize offsets, so GL
    // must not filter or we'd double-filter across face seams.
    options.magFilter = Filter::Nearest;
    options.minFilter = Filter::Nearest;
    options.wrapS = TextureWrapping::ClampToEdge;
    options.wrapT = TextureWrapping::ClampToEdge;
    options.generateMipmaps = false;
    options.depthBuffer = false;

    auto target = RenderTarget::create(PMREM_WIDTH, PMREM_HEIGHT, options);
    // cube_uv_reflection_fragment.glsl activates via Mapping::CubeUVReflection
    // — this is the signal to the material pipeline to compile ENVMAP_TYPE_CUBE_UV.
    target->texture->mapping = Mapping::CubeUVReflection;
    target->scissorTest = true;

    impl->material->uniforms["envMap"].setValue(&equirect);

    auto* oldTarget = renderer.getRenderTarget();
    const bool oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;

    // Clear the full atlas once before rendering tiles, so any regions not
    // covered by a tile start in a defined state (not GPU garbage).
    target->viewport.set(0, 0, static_cast<float>(PMREM_WIDTH), static_cast<float>(PMREM_HEIGHT));
    target->scissor.set(0, 0, static_cast<float>(PMREM_WIDTH), static_cast<float>(PMREM_HEIGHT));
    renderer.setRenderTarget(target.get(), 0, 0);
    renderer.clear(true, true, false);

    for (int lod = 0; lod < TOTAL_LODS; ++lod) {
        // Per-LOD tile geometry, mirroring three.js PMREMGenerator._halfBlur layout.
        const int sizeLod = (lod <= LOD_MAX - LOD_MIN)
                ? (PMREM_SIZE_MAX >> lod)
                : (PMREM_SIZE_MAX >> (LOD_MAX - LOD_MIN));  // 16 for extra LODs

        const int extraOffset = (lod > LOD_MAX - LOD_MIN) ? (lod - (LOD_MAX - LOD_MIN)) : 0;
        const int x = 3 * std::max(0, PMREM_SIZE_MAX - 2 * sizeLod);
        const int y = (lod == 0 ? 0 : 2 * PMREM_SIZE_MAX) + 2 * sizeLod * extraOffset;
        const int w = 3 * sizeLod;
        const int h = 2 * sizeLod;

        target->viewport.set(static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(w), static_cast<float>(h));
        target->scissor.set(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(w), static_cast<float>(h));

        impl->material->uniforms["roughness"].setValue(LOD_ROUGHNESS[lod]);
        // Fewer samples for smoothest / roughest extremes: LOD 0 is direct
        // fetch (1 sample); high roughness lobes are wide so low sample counts
        // still converge.
        const int samples = (lod == 0) ? 1 : (lod <= 2 ? 256 : 128);
        impl->material->uniforms["numSamples"].setValue(samples);

        // Fresh geometry per LOD: the half-texel UV extension depends on sizeLod,
        // so cube_uv_reflection_fragment.glsl's pixel-centre sampling aligns.
        auto geometry = createLodPlane(sizeLod);
        Mesh mesh(geometry, impl->material);
        mesh.frustumCulled = false;

        renderer.setRenderTarget(target.get(), 0, 0);
        renderer.render(mesh, *impl->camera);

        geometry->dispose();
    }

    renderer.autoClear = oldAutoClear;
    renderer.setRenderTarget(oldTarget, 0, 0);

    target->scissorTest = false;
    target->viewport.set(0, 0, static_cast<float>(PMREM_WIDTH), static_cast<float>(PMREM_HEIGHT));
    target->scissor.set(0, 0, static_cast<float>(PMREM_WIDTH), static_cast<float>(PMREM_HEIGHT));

    return target;
}
