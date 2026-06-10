
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
#include <vector>

using namespace threepp;
using namespace threepp::gl;

namespace {

    // ---------------------------------------------------------------------
    // Equirect LOD-strip atlas (replaces the old cube-face "cubeUV" atlas).
    //
    // The old GL IBL path packed 6 cube faces per roughness LOD into a 768x768
    // atlas and read it with a hand-rolled bilinear that clamped at every
    // cube-face boundary. On smoothly-curved surfaces (cylinders, capsules,
    // spheres) whose normal sweeps the full horizon, those face boundaries plus
    // the very low per-face resolution (16x16 for the roughest/diffuse tile)
    // showed up as hard vertical streaks under a high-contrast HDR — a defect
    // unique to GL (WGPU/Vulkan sample a prefiltered equirect mip chain).
    //
    // We now store N_LODS *full equirect* strips stacked vertically, each
    // GGX-prefiltered at an increasing roughness, and read them with hardware
    // bilinear + Repeat-U. Azimuth is seamless (Repeat wrap, no cube faces) and
    // the per-strip resolution is high enough that the banding is gone. The
    // material-facing entry point in cube_uv_reflection_fragment.glsl keeps the
    // name `textureCubeUV` for drop-in compatibility (same signature, same
    // Mapping::CubeUVReflection trigger).
    //
    // Strip L occupies atlas rows [L*STRIP_H, (L+1)*STRIP_H); roughness is
    // linear in L (roughness = L/(N_LODS-1)), matching the WGPU PMREM convention
    // and the shader's `roughness * (N_LODS-1)` LOD lookup.
    // ---------------------------------------------------------------------

    constexpr int STRIP_W = 512;
    constexpr int STRIP_H = 256;
    constexpr int N_LODS = 7;
    constexpr int ATLAS_W = STRIP_W;            // 512
    constexpr int ATLAS_H = STRIP_H * N_LODS;   // 1792

    // Must match EQ_STRIP_W / EQ_STRIP_H / EQ_N_LODS in
    // cube_uv_reflection_fragment.glsl.
    constexpr float LOD_ROUGHNESS[N_LODS] = {
            0.0f / 6.0f,  // 0.000  (sharpest — direct equirect copy)
            1.0f / 6.0f,  // 0.167
            2.0f / 6.0f,  // 0.333
            3.0f / 6.0f,  // 0.500
            4.0f / 6.0f,  // 0.667
            5.0f / 6.0f,  // 0.833
            6.0f / 6.0f,  // 1.000  (roughest — diffuse irradiance)
    };

    // Fullscreen quad. The render viewport restricts it to one strip; uv spans
    // [0,1]^2 over the strip, uv.y increasing upward (GL bottom-up) so uv.y=0 is
    // the south pole. getDirection() in the fragment shader maps (uv) -> world
    // direction; the material's eqUvFromDir() is its exact inverse.
    const char* const VERTEX_SRC = R"(#version 330 core
in vec3 position;
in vec2 uv;
out vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
)";

    // GGX importance-sampled equirect prefilter. Identical math to the WGPU
    // (WgpuPMREM.cpp) and Vulkan (prefilter_env.comp) prefilters: integrate the
    // source equirect over a GGX lobe around the output direction (= N for the
    // prefilter), weighted by NdotL. roughness==0 collapses to a direct fetch.
    const char* const FRAGMENT_SRC = R"(#version 330 core
precision highp float;
precision highp int;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D envMap;
uniform float roughness;
uniform int numSamples;

#define PI        3.14159265359
#define PI2       6.28318530718
#define RECIP_PI  0.31830988618
#define RECIP_2PI 0.15915494309

// Strip uv -> world direction (inverse of equirectUv below).
vec3 dirFromUv(vec2 uv) {
    float phi   = (uv.x - 0.5) * PI2;   // longitude about +Y, [-PI, PI]
    float theta = (uv.y - 0.5) * PI;    // latitude, [-PI/2, PI/2]
    float cosT = cos(theta);
    return vec3(cosT * cos(phi), sin(theta), cosT * sin(phi));
}

vec2 equirectUv(vec3 dir) {
    float u = atan(dir.z, dir.x) * RECIP_2PI + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * RECIP_PI + 0.5;
    return vec2(u, v);
}

float vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), vdc(i));
}

vec3 importanceSampleGGX(vec2 Xi, vec3 N, float a) {
    float phi = PI2 * Xi.x;
    float cosTheta = sqrt(max((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y), 0.0));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    vec3 H_t = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 upN = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(upN, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H_t.x + bitangent * H_t.y + N * H_t.z);
}

float ggxD(float NdotH, float a) {
    float a2 = a * a;
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

void main() {
    vec3 N = normalize(dirFromUv(vUv));

    if (roughness <= 0.0) {
        fragColor = vec4(texture(envMap, equirectUv(N)).rgb, 1.0);
        return;
    }

    // Colbert/Krivanek mip-biased importance sampling: each GGX sample reads the
    // source at a mip whose texels cover ~the sample's solid angle, so a small,
    // ultra-bright sun is pre-averaged into its neighbourhood instead of being
    // hit by a handful of samples (which produced a scattered-dots starburst on
    // the wide roughness lobes). The source equirect carries a full mip chain.
    float a = roughness * roughness;
    vec2 srcSize = vec2(textureSize(envMap, 0));
    float saTexel = 4.0 * PI / (srcSize.x * srcSize.y);

    vec3 accumColor = vec3(0.0);
    float accumWeight = 0.0;
    uint NS = uint(numSamples);
    for (uint i = 0u; i < NS; i++) {
        vec2 Xi = hammersley(i, NS);
        vec3 H = importanceSampleGGX(Xi, N, a);
        vec3 L = normalize(2.0 * dot(N, H) * H - N);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float NdotH = max(dot(N, H), 0.0);
            float pdf = ggxD(NdotH, a) * 0.25 + 1e-4;        // V=N -> VdotH=NdotH
            float saSample = 1.0 / (float(NS) * pdf);
            float mip = 0.5 * log2(saSample / saTexel);
            // Floor the source mip by roughness. The Hammersley GGX directions
            // are discrete, so a tiny ultra-bright sun gets hit at a few phi
            // spokes that survive as radial streaks on the wide rough lobes
            // (→ vertical streaks on curved surfaces). Reading a sufficiently
            // blurred source mip spreads each sample's footprint enough to
            // overlap its neighbours and erase the spokes. (WGPU avoids this via
            // low-res rough mips; our strips are full-res so we blur at source.)
            // The floor is RELATIVE to the source size: roughness=1 reads the
            // ~16x8 mip (log2(W)-4), which keeps the hemisphere-scale variation
            // a directional diffuse ambient needs. An absolute floor (was
            // `roughness * 8.0`) overshot small sources past their whole mip
            // pyramid into the 1x1 global average — diffuse IBL lost all
            // directionality (caught by the top-lit furnace test).
            mip = max(mip, roughness * max(log2(srcSize.x) - 4.0, 0.0));
            mip = max(mip, 0.0);
            // Clamp per-channel to bound residual fireflies / RGBE Inf.
            vec3 s = min(textureLod(envMap, equirectUv(L), mip).rgb, vec3(50.0));
            accumColor += s * NdotL;
            accumWeight += NdotL;
        }
    }
    fragColor = vec4(accumColor / max(accumWeight, 0.001), 1.0);
}
)";

    // One fullscreen quad ([-1,1]^2, uv [0,1]^2). The viewport restricts it to
    // the active strip, so the same geometry is reused for every LOD.
    std::shared_ptr<BufferGeometry> createFullscreenQuad() {
        const std::vector<float> positions = {
                -1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, 1.f, 0.f,
                -1.f, -1.f, 0.f, 1.f, 1.f, 0.f, -1.f, 1.f, 0.f};
        const std::vector<float> uvs = {
                0.f, 0.f, 1.f, 0.f, 1.f, 1.f,
                0.f, 0.f, 1.f, 1.f, 0.f, 1.f};

        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
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
    // Hardware bilinear — the shader samples the strips directly (no manual
    // 4-tap). Repeat on U makes the equirect azimuth seam (atan2 at -X) wrap
    // seamlessly; V is clamped per-strip in the shader to avoid bleeding into
    // the neighbouring roughness strip.
    options.magFilter = Filter::Linear;
    options.minFilter = Filter::Linear;
    options.wrapS = TextureWrapping::Repeat;
    options.wrapT = TextureWrapping::ClampToEdge;
    options.generateMipmaps = false;
    options.depthBuffer = false;

    auto target = RenderTarget::create(ATLAS_W, ATLAS_H, options);
    // cube_uv_reflection_fragment.glsl activates via Mapping::CubeUVReflection
    // — the signal to compile ENVMAP_TYPE_CUBE_UV (now the equirect-strip path).
    target->texture->mapping = Mapping::CubeUVReflection;
    target->scissorTest = true;

    impl->material->uniforms["envMap"].setValue(&equirect);

    auto* oldTarget = renderer.getRenderTarget();
    const bool oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;

    // Clear the full atlas once so any uncovered region is defined.
    target->viewport.set(0, 0, static_cast<float>(ATLAS_W), static_cast<float>(ATLAS_H));
    target->scissor.set(0, 0, static_cast<float>(ATLAS_W), static_cast<float>(ATLAS_H));
    renderer.setRenderTarget(target.get(), 0, 0);
    renderer.clear(true, true, false);

    auto geometry = createFullscreenQuad();
    Mesh mesh(geometry, impl->material);
    mesh.frustumCulled = false;

    for (int lod = 0; lod < N_LODS; ++lod) {
        const int y = lod * STRIP_H;
        target->viewport.set(0.f, static_cast<float>(y),
                             static_cast<float>(STRIP_W), static_cast<float>(STRIP_H));
        target->scissor.set(0.f, static_cast<float>(y),
                            static_cast<float>(STRIP_W), static_cast<float>(STRIP_H));

        impl->material->uniforms["roughness"].setValue(LOD_ROUGHNESS[lod]);
        // LOD 0 is a direct fetch (1 sample); narrow low-roughness lobes need
        // more samples to converge; wide rough lobes need fewer.
        const int samples = (lod == 0) ? 1 : (LOD_ROUGHNESS[lod] < 0.3f ? 256 : 128);
        impl->material->uniforms["numSamples"].setValue(samples);

        renderer.setRenderTarget(target.get(), 0, 0);
        renderer.render(mesh, *impl->camera);
    }

    geometry->dispose();

    renderer.autoClear = oldAutoClear;
    renderer.setRenderTarget(oldTarget, 0, 0);

    target->scissorTest = false;
    target->viewport.set(0, 0, static_cast<float>(ATLAS_W), static_cast<float>(ATLAS_H));
    target->scissor.set(0, 0, static_cast<float>(ATLAS_W), static_cast<float>(ATLAS_H));

    return target;
}
