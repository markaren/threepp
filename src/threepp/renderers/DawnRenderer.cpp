
#include "threepp/renderers/DawnRenderer.hpp"

#include "dawn/DawnGeometries.hpp"
#include "dawn/DawnTextures.hpp"
#include "dawn/DawnState.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/ShadowMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/dawn/GPUTexture.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/math/Frustum.hpp"

#include "threepp/renderers/common/Lights.hpp"
#include "threepp/renderers/common/RenderLists.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#define GLFW_INCLUDE_NONE
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
// Match the linkage used by the implementation (extern "C" in C++).
#define STBIWDEF extern "C"
#include "stb_image_write.h"
#undef STBIWDEF

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;

#ifdef __APPLE__
extern "C" void* dawn_create_metal_layer(void* nsWindow);
#endif

namespace {

    // Maximum time to wait for async WebGPU operations before aborting.
    constexpr auto WGPU_ASYNC_TIMEOUT = std::chrono::seconds(10);

    // Feature bitmask for pipeline caching
    // Bits 0-3: material features
    // Bits 4-5: cull mode (00=None, 01=Front, 10=Back)
    // Bit 6: wireframe (LineList vs TriangleList)
    // Bits 7-9: blend mode (000=Normal, 001=None, 010=Additive, 011=Subtractive, 100=Multiply)
    // Feature flags use uint64_t to avoid bit collisions (>32 features)
    constexpr uint64_t FEAT_NONE       = 0;
    constexpr uint64_t FEAT_TEXTURE    = 1ULL << 0;
    constexpr uint64_t FEAT_LIGHTING   = 1ULL << 1;
    constexpr uint64_t FEAT_SPECULAR   = 1ULL << 2;
    constexpr uint64_t FEAT_PBR        = 1ULL << 3;
    constexpr uint64_t FEAT_NORMAL_MAP = 1ULL << 10;

    constexpr uint64_t CULL_SHIFT = 4;
    constexpr uint64_t CULL_MASK  = 0x3ULL << CULL_SHIFT;
    constexpr uint64_t CULL_NONE  = 0ULL << CULL_SHIFT;
    constexpr uint64_t CULL_FRONT = 1ULL << CULL_SHIFT;
    constexpr uint64_t CULL_BACK  = 2ULL << CULL_SHIFT;

    constexpr uint64_t WIREFRAME_BIT = 1ULL << 6;

    constexpr uint64_t BLEND_SHIFT      = 7;
    constexpr uint64_t BLEND_MASK       = 0x7ULL << BLEND_SHIFT;
    constexpr uint64_t BLEND_NORMAL     = 0ULL << BLEND_SHIFT;
    constexpr uint64_t BLEND_DISABLED   = 1ULL << BLEND_SHIFT;
    constexpr uint64_t BLEND_ADDITIVE   = 2ULL << BLEND_SHIFT;
    constexpr uint64_t BLEND_SUBTRACTIVE= 3ULL << BLEND_SHIFT;
    constexpr uint64_t BLEND_MULTIPLY   = 4ULL << BLEND_SHIFT;

    constexpr uint64_t DEPTH_WRITE_OFF  = 1ULL << 11;
    constexpr uint64_t FEAT_SHADOW      = 1ULL << 12;
    constexpr uint64_t FEAT_FOG_LINEAR  = 1ULL << 13;
    constexpr uint64_t FEAT_FOG_EXP2    = 1ULL << 14;
    constexpr uint64_t FEAT_INSTANCE_COLOR    = 1ULL << 15;
    constexpr uint64_t FEAT_DISPLACEMENT_MAP  = 1ULL << 16;
    constexpr uint64_t FEAT_MORPH_TARGETS     = 1ULL << 17;

    // Topology mode (bits 18-19)
    constexpr uint64_t TOPO_SHIFT       = 18;
    constexpr uint64_t TOPO_MASK        = 0x3ULL << TOPO_SHIFT;
    constexpr uint64_t TOPO_TRIANGLE    = 0ULL << TOPO_SHIFT;
    constexpr uint64_t TOPO_LINE_LIST   = 1ULL << TOPO_SHIFT;
    constexpr uint64_t TOPO_LINE_STRIP  = 2ULL << TOPO_SHIFT;
    constexpr uint64_t TOPO_POINT_LIST  = 3ULL << TOPO_SHIFT;

    // Instancing bit
    constexpr uint64_t FEAT_INSTANCED   = 1ULL << 20;

    // Vertex colors bit
    constexpr uint64_t FEAT_VERTEX_COLORS = 1ULL << 21;

    // Additional texture map features
    constexpr uint64_t FEAT_EMISSIVE_MAP  = 1ULL << 22;
    constexpr uint64_t FEAT_ROUGHNESS_MAP = 1ULL << 23;
    constexpr uint64_t FEAT_METALNESS_MAP = 1ULL << 24;
    constexpr uint64_t FEAT_AO_MAP        = 1ULL << 25;
    constexpr uint64_t FEAT_ALPHA_MAP     = 1ULL << 26;
    constexpr uint64_t FEAT_SPECULAR_MAP  = 1ULL << 27;
    constexpr uint64_t FEAT_LIGHT_MAP     = 1ULL << 28;
    constexpr uint64_t FEAT_SRGB_OUTPUT   = 1ULL << 29;
    constexpr uint64_t FEAT_BUMP_MAP      = 1ULL << 30;
    constexpr uint64_t FEAT_GRADIENT_MAP  = 1ULL << 31;

    // Tone mapping mode (bits 32-34) — no longer collides with bits 15-17
    constexpr uint64_t TONEMAP_SHIFT    = 32;
    constexpr uint64_t TONEMAP_MASK     = 0x7ULL << TONEMAP_SHIFT;
    constexpr uint64_t TONEMAP_NONE     = 0ULL << TONEMAP_SHIFT;
    constexpr uint64_t TONEMAP_LINEAR   = 1ULL << TONEMAP_SHIFT;
    constexpr uint64_t TONEMAP_REINHARD = 2ULL << TONEMAP_SHIFT;
    constexpr uint64_t TONEMAP_CINEON   = 3ULL << TONEMAP_SHIFT;
    constexpr uint64_t TONEMAP_ACES     = 4ULL << TONEMAP_SHIFT;

    // Environment map
    constexpr uint64_t FEAT_ENV_MAP    = 1ULL << 35;

    // SkinnedMesh
    constexpr uint64_t FEAT_SKINNING   = 1ULL << 36;

    constexpr uint32_t SHADOW_MAP_SIZE = 1024;
    constexpr int MAX_SHADOW_LIGHTS = 4;
    // Per-light: lightVP(64) + bias(4) + normalBias(4) + padding(8) = 80 bytes
    constexpr size_t SHADOW_UNIFORM_PER_LIGHT = 80;
    // Total: count(4) + pad(12) + N * per_light
    constexpr size_t SHADOW_UNIFORM_SIZE = 16 + MAX_SHADOW_LIGHTS * SHADOW_UNIFORM_PER_LIGHT; // 336 bytes

    constexpr int MAX_DIR_LIGHTS   = 4;
    constexpr int MAX_POINT_LIGHTS = 4;
    constexpr int MAX_SPOT_LIGHTS  = 4;
    constexpr int MAX_HEMI_LIGHTS  = 2;

    // Transform: model(64) + view(64) + proj(64) + normalMatrix(48 = 3*vec4 padded) + cameraPos(12) + pad(4) = 256
    constexpr size_t TRANSFORM_UNIFORM_SIZE = 256;

    // Material: diffuse(16) + specularAndShininess(16) + roughnessMetalnessOpacity(16) + emissive(16) + flags(16) + fog(16) + toneMapping(16) = 112, pad to 128
    constexpr size_t MATERIAL_UNIFORM_SIZE = 128;

    // Light: header(32) + dir(4*32=128) + point(4*48=192) + spot(4*64=256) + hemi(2*48=96) = 704
    constexpr size_t LIGHT_UNIFORM_SIZE = 704;


    std::string buildWGSL(uint64_t features) {
        std::ostringstream s;

        s << R"(
struct TransformUniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct MaterialUniforms {
    diffuse: vec4<f32>,
    specularAndShininess: vec4<f32>,
    roughnessMetalnessOpacity: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
    clipPlane: vec4<f32>,
};
@group(0) @binding(1) var<uniform> material: MaterialUniforms;
)";

        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);
        if (lit) {
            s << "struct DirectionalLightGPU { direction: vec3<f32>, _p0: f32, color: vec3<f32>, _p1: f32, };\n";
            s << "struct PointLightGPU { position: vec3<f32>, _p0: f32, color: vec3<f32>, distance: f32, decay: f32, _p1: f32, _p2: f32, _p3: f32, };\n";
            s << "struct SpotLightGPU { position: vec3<f32>, _p0: f32, direction: vec3<f32>, _p1: f32, color: vec3<f32>, distance: f32, decay: f32, coneCos: f32, penumbraCos: f32, _p2: f32, };\n";
            s << "struct HemisphereLightGPU { direction: vec3<f32>, _p0: f32, skyColor: vec3<f32>, _p1: f32, groundColor: vec3<f32>, _p2: f32, };\n";
            s << "struct LightData {\n";
            s << "  numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,\n";
            s << "  ambient: vec3<f32>, _pad: f32,\n";
            s << "  directional: array<DirectionalLightGPU, " << MAX_DIR_LIGHTS << ">,\n";
            s << "  point: array<PointLightGPU, " << MAX_POINT_LIGHTS << ">,\n";
            s << "  spot: array<SpotLightGPU, " << MAX_SPOT_LIGHTS << ">,\n";
            s << "  hemi: array<HemisphereLightGPU, " << MAX_HEMI_LIGHTS << ">,\n";
            s << "};\n";
            s << "@group(0) @binding(2) var<uniform> lights: LightData;\n";
        }

        if (features & FEAT_TEXTURE) {
            s << "@group(0) @binding(3) var t_diffuse: texture_2d<f32>;\n";
            s << "@group(0) @binding(4) var s_diffuse: sampler;\n";
        }
        if (features & FEAT_NORMAL_MAP) {
            s << "@group(0) @binding(5) var t_normalMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(6) var s_normalMap: sampler;\n";
        }

        if (features & FEAT_EMISSIVE_MAP) {
            s << "@group(0) @binding(10) var t_emissiveMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(11) var s_emissiveMap: sampler;\n";
        }
        if (features & FEAT_ROUGHNESS_MAP) {
            s << "@group(0) @binding(12) var t_roughnessMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(13) var s_roughnessMap: sampler;\n";
        }
        if (features & FEAT_METALNESS_MAP) {
            s << "@group(0) @binding(14) var t_metalnessMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(15) var s_metalnessMap: sampler;\n";
        }
        if (features & FEAT_AO_MAP) {
            s << "@group(0) @binding(16) var t_aoMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(17) var s_aoMap: sampler;\n";
        }
        if (features & FEAT_ALPHA_MAP) {
            s << "@group(0) @binding(18) var t_alphaMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(19) var s_alphaMap: sampler;\n";
        }
        if (features & FEAT_SPECULAR_MAP) {
            s << "@group(0) @binding(20) var t_specularMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(21) var s_specularMap: sampler;\n";
        }
        if (features & FEAT_LIGHT_MAP) {
            s << "@group(0) @binding(22) var t_lightMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(23) var s_lightMap: sampler;\n";
        }
        if (features & FEAT_BUMP_MAP) {
            s << "@group(0) @binding(24) var t_bumpMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(25) var s_bumpMap: sampler;\n";
        }
        if (features & FEAT_GRADIENT_MAP) {
            s << "@group(0) @binding(26) var t_gradientMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(27) var s_gradientMap: sampler;\n";
        }
        if (features & FEAT_DISPLACEMENT_MAP) {
            s << "@group(0) @binding(30) var t_displacementMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(31) var s_displacementMap: sampler;\n";
        }
        if (features & FEAT_ENV_MAP) {
            s << "@group(0) @binding(32) var t_envMap: texture_cube<f32>;\n";
            s << "@group(0) @binding(33) var s_envMap: sampler;\n";
        }

        if (features & FEAT_INSTANCED) {
            s << "struct InstanceData {\n";
            s << "    model: mat4x4<f32>,\n";
            if (features & FEAT_INSTANCE_COLOR) {
                s << "    color: vec4<f32>,\n";
            }
            s << "};\n";
            s << "@group(0) @binding(28) var<storage, read> instances: array<InstanceData>;\n";
        }

        if (features & FEAT_MORPH_TARGETS) {
            // Storage buffer: [numTargets, influence0, influence1, ..., pad..., morphPositions...]
            // morphPositions: for each target, vertexCount vec3s stored as vec4s
            s << "struct MorphData {\n";
            s << "    numTargets: u32,\n";
            s << "    _pad0: u32, _pad1: u32, _pad2: u32,\n";
            s << "    influences: array<vec4<f32>, 2>,\n"; // up to 8 influences (2 vec4s)
            s << "    positions: array<vec4<f32>>,\n";     // packed morph positions
            s << "};\n";
            s << "@group(0) @binding(29) var<storage, read> morph: MorphData;\n";
        }

        if (features & FEAT_SKINNING) {
            s << "struct SkinData {\n";
            s << "    bindMatrix: mat4x4<f32>,\n";
            s << "    bindMatrixInverse: mat4x4<f32>,\n";
            s << "    boneCount: u32,\n";
            s << "    _pad0: u32, _pad1: u32, _pad2: u32,\n";
            s << "    bones: array<mat4x4<f32>>,\n";
            s << "};\n";
            s << "@group(0) @binding(34) var<storage, read> skin: SkinData;\n";
            s << "struct SkinVertex {\n";
            s << "    index: vec4<f32>,\n";
            s << "    weight: vec4<f32>,\n";
            s << "};\n";
            s << "@group(0) @binding(35) var<storage, read> skinVertices: array<SkinVertex>;\n";
        }

        if (features & FEAT_SHADOW) {
            s << "const MAX_SHADOW_LIGHTS: u32 = " << MAX_SHADOW_LIGHTS << "u;\n";
            s << R"(
struct ShadowLightData {
    lightVP: mat4x4<f32>,
    bias: f32,
    normalBias: f32,
    _pad0: f32,
    _pad1: f32,
};
struct ShadowUniforms {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lights: array<ShadowLightData, MAX_SHADOW_LIGHTS>,
};
@group(0) @binding(7) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(8) var t_shadowMaps: texture_depth_2d_array;
@group(0) @binding(9) var s_shadowMap: sampler_comparison;
)";
        }

        s << "\nstruct VertexInput {\n";
        s << "    @location(0) position: vec3<f32>,\n";
        s << "    @location(1) normal: vec3<f32>,\n";
        s << "    @location(2) uv: vec2<f32>,\n";
        s << "    @location(3) color: vec3<f32>,\n";
        s << "};\n";

        s << "struct VertexOutput {\n";
        s << "    @builtin(position) clipPos: vec4<f32>,\n";
        s << "    @location(0) worldPos: vec3<f32>,\n";
        s << "    @location(1) worldNormal: vec3<f32>,\n";
        s << "    @location(2) uv: vec2<f32>,\n";
        // Shadow: lightSpacePos removed — computed in fragment shader from worldPos
        if (features & FEAT_VERTEX_COLORS) {
            s << "    @location(4) vertexColor: vec3<f32>,\n";
        }
        if (features & FEAT_INSTANCE_COLOR) {
            s << "    @location(5) instanceColor: vec3<f32>,\n";
        }
        s << "};\n";

        // Build vertex function signature with optional builtins
        s << "\n@vertex\n";
        {
            bool needInstanceIdx = features & FEAT_INSTANCED;
            bool needVertexIdx = (features & FEAT_MORPH_TARGETS) || (features & FEAT_SKINNING);
            s << "fn vs_main(in: VertexInput";
            if (needInstanceIdx) s << ", @builtin(instance_index) iid: u32";
            if (needVertexIdx) s << ", @builtin(vertex_index) vid: u32";
            s << ") -> VertexOutput {\n";
        }
        s << "    var out: VertexOutput;\n";

        // Morph target blending: blend base position with morph targets
        if (features & FEAT_MORPH_TARGETS) {
            s << "    var morphedPos = in.position;\n";
            s << "    let numT = morph.numTargets;\n";
            s << "    for (var t = 0u; t < numT; t++) {\n";
            s << "        let inf = morph.influences[t / 4u][t % 4u];\n";
            s << "        if (inf > 0.0) {\n";
            s << "            let mp = morph.positions[t * " << "arrayLength(&morph.positions) / max(numT, 1u)" << " + vid];\n";
            s << "            morphedPos = morphedPos + (mp.xyz - in.position) * inf;\n";
            s << "        }\n";
            s << "    }\n";
        }

        // Optionally displace position along normal before transform
        if (features & FEAT_DISPLACEMENT_MAP) {
            std::string posVar = (features & FEAT_MORPH_TARGETS) ? "morphedPos" : "in.position";
            s << "    let dispAmount = textureSampleLevel(t_displacementMap, s_displacementMap, in.uv, 0.0).r;\n";
            s << "    let displacedPos = " << posVar << " + in.normal * dispAmount * material.roughnessMetalnessOpacity.w;\n";
        }
        // Determine which position variable to use
        std::string posExpr = "in.position";
        if (features & FEAT_MORPH_TARGETS) posExpr = "morphedPos";
        if (features & FEAT_DISPLACEMENT_MAP) posExpr = "displacedPos";

        // Skinning: blend bone transforms per-vertex
        if (features & FEAT_SKINNING) {
            s << "    let sv = skinVertices[vid];\n";
            s << "    var skinnedPos = vec4<f32>(0.0);\n";
            s << "    var skinnedNormal = vec3<f32>(0.0);\n";
            s << "    let bindPos = skin.bindMatrix * vec4<f32>(" << posExpr << ", 1.0);\n";
            s << "    let bindNrm = (skin.bindMatrix * vec4<f32>(in.normal, 0.0)).xyz;\n";
            s << "    for (var bi = 0u; bi < 4u; bi++) {\n";
            s << "        let w = sv.weight[bi];\n";
            s << "        if (w > 0.0) {\n";
            s << "            let bIdx = u32(sv.index[bi]);\n";
            s << "            let bm = skin.bones[bIdx];\n";
            s << "            skinnedPos += (bm * bindPos) * w;\n";
            s << "            skinnedNormal += (mat3x3<f32>(bm[0].xyz, bm[1].xyz, bm[2].xyz) * bindNrm) * w;\n";
            s << "        }\n";
            s << "    }\n";
            s << "    let finalSkinPos = (skin.bindMatrixInverse * skinnedPos).xyz;\n";
            s << "    let finalSkinNormal = normalize((skin.bindMatrixInverse * vec4<f32>(skinnedNormal, 0.0)).xyz);\n";
            posExpr = "finalSkinPos";
        }

        std::string normalExpr = "in.normal";
        if (features & FEAT_SKINNING) normalExpr = "finalSkinNormal";

        if (features & FEAT_INSTANCED) {
            s << "    let instanceModel = transform.model * instances[iid].model;\n";
            s << "    let worldPos4 = instanceModel * vec4<f32>(" << posExpr << ", 1.0);\n";
            s << "    let im3 = mat3x3<f32>(instanceModel[0].xyz, instanceModel[1].xyz, instanceModel[2].xyz);\n";
            s << "    out.worldNormal = normalize(im3 * " << normalExpr << ");\n";
        } else {
            s << "    let worldPos4 = transform.model * vec4<f32>(" << posExpr << ", 1.0);\n";
            s << "    let nm = mat3x3<f32>(transform.normalCol0.xyz, transform.normalCol1.xyz, transform.normalCol2.xyz);\n";
            s << "    out.worldNormal = normalize(nm * " << normalExpr << ");\n";
        }
        s << "    out.worldPos = worldPos4.xyz;\n";
        s << "    out.uv = in.uv;\n";
        s << "    out.clipPos = transform.proj * transform.view * worldPos4;\n";

        // Shadow: lightSpacePos now computed in fragment shader from worldPos
        if (features & FEAT_VERTEX_COLORS) {
            s << "\n    out.vertexColor = in.color;\n";
        }
        if (features & FEAT_INSTANCE_COLOR) {
            s << "    out.instanceColor = instances[iid].color.rgb;\n";
        }

        s << R"(
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clipping plane test (active when clipPlane normal is non-zero)
    let cpn = material.clipPlane.xyz;
    if (dot(cpn, cpn) > 0.0) {
        if (dot(in.worldPos, cpn) + material.clipPlane.w < 0.0) {
            discard;
        }
    }
    var baseColor = material.diffuse.rgb;
    var opacity = material.roughnessMetalnessOpacity.z;
    var roughness = material.roughnessMetalnessOpacity.x;
    var metalness = material.roughnessMetalnessOpacity.y;
)";
        if (features & FEAT_VERTEX_COLORS) {
            s << "    baseColor = baseColor * in.vertexColor;\n";
        }
        if (features & FEAT_INSTANCE_COLOR) {
            s << "    baseColor = baseColor * in.instanceColor;\n";
        }

        if (features & FEAT_TEXTURE) {
            s << "    let texColor = textureSample(t_diffuse, s_diffuse, in.uv);\n";
            s << "    baseColor = baseColor * texColor.rgb;\n";
        }
        if (features & FEAT_ROUGHNESS_MAP) {
            s << "    roughness = roughness * textureSample(t_roughnessMap, s_roughnessMap, in.uv).g;\n";
        }
        if (features & FEAT_METALNESS_MAP) {
            s << "    metalness = metalness * textureSample(t_metalnessMap, s_metalnessMap, in.uv).b;\n";
        }
        if (features & FEAT_ALPHA_MAP) {
            s << "    opacity = opacity * textureSample(t_alphaMap, s_alphaMap, in.uv).r;\n";
        }

        if (lit) {
            if (features & FEAT_NORMAL_MAP) {
                s << R"(
    // Screen-space normal map perturbation (no tangent attributes needed)
    let dPdx = dpdx(in.worldPos);
    let dPdy = dpdy(in.worldPos);
    let dUVdx = dpdx(in.uv);
    let dUVdy = dpdy(in.uv);
    let T = normalize(dPdx * dUVdy.y - dPdy * dUVdx.y);
    let B = normalize(dPdy * dUVdx.x - dPdx * dUVdy.x);
    let geomN = normalize(in.worldNormal);
    let TBN = mat3x3<f32>(T, B, geomN);
    let nmSample = textureSample(t_normalMap, s_normalMap, in.uv).rgb * 2.0 - vec3<f32>(1.0);
    let normalScale = material.flags.zw;
    let scaledNm = vec3<f32>(nmSample.xy * normalScale, nmSample.z);
    let N = normalize(TBN * scaledNm);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
            } else if (features & FEAT_BUMP_MAP) {
                // Bump map: perturb normal using height gradient via screen-space derivatives
                s << R"(
    let bmp_dPdx = dpdx(in.worldPos);
    let bmp_dPdy = dpdy(in.worldPos);
    let bmp_dUVdx = dpdx(in.uv);
    let bmp_dUVdy = dpdy(in.uv);
    let Hll = textureSample(t_bumpMap, s_bumpMap, in.uv).r;
    let dBx = textureSample(t_bumpMap, s_bumpMap, in.uv + bmp_dUVdx).r - Hll;
    let dBy = textureSample(t_bumpMap, s_bumpMap, in.uv + bmp_dUVdy).r - Hll;
    let bumpScale = material.flags.y;
    let geomN = normalize(in.worldNormal);
    let crossX = cross(bmp_dPdy, geomN);
    let crossY = cross(geomN, bmp_dPdx);
    let surfGrad = (crossX * dBx + crossY * dBy) * bumpScale;
    let N = normalize(in.worldNormal - surfGrad);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
            } else {
                s << R"(
    let N = normalize(in.worldNormal);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
            }
            s << R"(
    var diffuseLight = lights.ambient;
    var specularLight = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < lights.numDir; i++) {
        let L = normalize(lights.directional[i].direction);
        var NdotL = max(dot(N, L), 0.0);
)";
            if (features & FEAT_GRADIENT_MAP) {
                s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
            }
            s << R"(
        diffuseLight += lights.directional[i].color * NdotL;
)";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.directional[i].color * material.specularAndShininess.rgb * s; }\n";
            }
            if (features & FEAT_PBR) {
                // GGX/Trowbridge-Reitz NDF with Schlick Fresnel
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let NdotV = max(dot(N, V), 0.001);\n";
                s << "          let r = roughness; let m = metalness;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.directional[i].color * F * D * NdotL; }\n";
            }
            s << "    }\n";

            s << R"(
    for (var i = 0u; i < lights.numPoint; i++) {
        let lv = lights.point[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        var NdotL = max(dot(N, L), 0.0);
        var att = 1.0;
        if (lights.point[i].distance > 0.0) {
            let r2 = clamp(1.0 - pow(d / lights.point[i].distance, 4.0), 0.0, 1.0);
            att = r2 * r2 / (d * d + 0.0001);
        }
)";
            if (features & FEAT_GRADIENT_MAP) {
                s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
            }
            s << "        diffuseLight += lights.point[i].color * NdotL * att;\n";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.point[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
            if (features & FEAT_PBR) {
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let r = roughness; let m = metalness;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.point[i].color * F * D * NdotL * att; }\n";
            }
            s << "    }\n";

            s << R"(
    for (var i = 0u; i < lights.numSpot; i++) {
        let lv = lights.spot[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        var NdotL = max(dot(N, L), 0.0);
        let ac = dot(L, normalize(lights.spot[i].direction));
        let se = smoothstep(lights.spot[i].coneCos, lights.spot[i].penumbraCos, ac);
        var att = se;
        if (lights.spot[i].distance > 0.0) {
            let r2 = clamp(1.0 - pow(d / lights.spot[i].distance, 4.0), 0.0, 1.0);
            att = att * r2 * r2 / (d * d + 0.0001);
        }
)";
            if (features & FEAT_GRADIENT_MAP) {
                s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
            }
            s << "        diffuseLight += lights.spot[i].color * NdotL * att;\n";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.spot[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
            if (features & FEAT_PBR) {
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let r = roughness; let m = metalness;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.spot[i].color * F * D * NdotL * att; }\n";
            }
            s << R"(
    }
    for (var i = 0u; i < lights.numHemi; i++) {
        let w = 0.5 * dot(N, lights.hemi[i].direction) + 0.5;
        diffuseLight += mix(lights.hemi[i].groundColor, lights.hemi[i].skyColor, w);
    }
)";
            // Apply shadow factor from all active shadow lights
            if (features & FEAT_SHADOW) {
                s << R"(
    {
        var combinedShadow = 1.0;
        let worldPos4 = vec4<f32>(in.worldPos, 1.0);
        for (var si = 0u; si < shadow.count; si++) {
            let lightSpacePos = shadow.lights[si].lightVP * worldPos4;
            let projCoords = lightSpacePos.xyz / lightSpacePos.w;
            let shadowUV = vec2<f32>(projCoords.x * 0.5 + 0.5, 1.0 - (projCoords.y * 0.5 + 0.5));
            let currentDepth = projCoords.z - shadow.lights[si].bias;
            if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0 && currentDepth >= 0.0 && currentDepth <= 1.0) {
                let texelSize = 1.0 / f32(textureDimensions(t_shadowMaps).x);
                var shadow_sum = 0.0;
                for (var sy = -1; sy <= 1; sy++) {
                    for (var sx = -1; sx <= 1; sx++) {
                        let offset = vec2<f32>(f32(sx), f32(sy)) * texelSize;
                        shadow_sum += textureSampleCompare(t_shadowMaps, s_shadowMap, shadowUV + offset, i32(si), currentDepth);
                    }
                }
                combinedShadow = min(combinedShadow, shadow_sum / 9.0);
            }
        }
        // Shadow only attenuates diffuse/specular, not ambient/emissive
        diffuseLight = lights.ambient + (diffuseLight - lights.ambient) * combinedShadow;
        specularLight = specularLight * combinedShadow;
    }
)";
            }

            // Compute emissive contribution
            if (features & FEAT_EMISSIVE_MAP) {
                s << "    var emissiveColor = material.emissive.rgb * textureSample(t_emissiveMap, s_emissiveMap, in.uv).rgb;\n";
            } else {
                s << "    var emissiveColor = material.emissive.rgb;\n";
            }

            // SpecularMap modulates specular light contribution
            if (features & FEAT_SPECULAR_MAP) {
                s << "    specularLight = specularLight * textureSample(t_specularMap, s_specularMap, in.uv).rgb;\n";
            }

            if (features & FEAT_PBR) {
                // Ambient specular: metallic surfaces reflect ambient light
                s << "    let F0 = mix(vec3<f32>(0.04), baseColor, metalness);\n";
                if (features & FEAT_ENV_MAP) {
                    // Env map reflection: sample cube map using reflection vector (V, N already defined)
                    s << "    let R = reflect(-V, N);\n";
                    s << "    let envColor = textureSample(t_envMap, s_envMap, R).rgb;\n";
                    s << "    let ambientSpecular = F0 * envColor * material.emissive.w;\n";
                } else {
                    s << "    let ambientSpecular = F0 * lights.ambient * (1.0 - roughness);\n";
                }
                s << "    baseColor = baseColor * (1.0 - metalness) * diffuseLight + specularLight + ambientSpecular + emissiveColor;\n";
            } else {
                s << "    baseColor = baseColor * diffuseLight + specularLight + emissiveColor;\n";
            }

            // LightMap adds baked illumination
            if (features & FEAT_LIGHT_MAP) {
                s << "    baseColor = baseColor + textureSample(t_lightMap, s_lightMap, in.uv).rgb;\n";
            }

            // AO map darkens ambient contribution
            if (features & FEAT_AO_MAP) {
                s << "    {\n";
                s << "        let ao = textureSample(t_aoMap, s_aoMap, in.uv).r;\n";
                s << "        let aoIntensity = material.flags.x;\n";
                s << "        baseColor = baseColor * mix(1.0, ao, aoIntensity);\n";
                s << "    }\n";
            }
        } else {
            // Unlit path: emissiveMap still applies
            if (features & FEAT_EMISSIVE_MAP) {
                s << "    baseColor = baseColor + material.emissive.rgb * textureSample(t_emissiveMap, s_emissiveMap, in.uv).rgb;\n";
            }
        }

        // Tone mapping (applied before fog)
        if ((features & TONEMAP_MASK) != TONEMAP_NONE) {
            s << "    let exposure = material.fogParams.w;\n";
            s << "    baseColor = baseColor * exposure;\n";
            if ((features & TONEMAP_MASK) == TONEMAP_REINHARD) {
                s << "    baseColor = baseColor / (vec3<f32>(1.0) + baseColor);\n";
            } else if ((features & TONEMAP_MASK) == TONEMAP_CINEON) {
                // Optimized filmic operator (Uncharted2-like)
                s << "    let x = max(vec3<f32>(0.0), baseColor - vec3<f32>(0.004));\n";
                s << "    baseColor = (x * (6.2 * x + vec3<f32>(0.5))) / (x * (6.2 * x + vec3<f32>(1.7)) + vec3<f32>(0.06));\n";
            } else if ((features & TONEMAP_MASK) == TONEMAP_ACES) {
                s << "    let a = baseColor * (baseColor * 2.51 + vec3<f32>(0.03));\n";
                s << "    let b = baseColor * (baseColor * 2.43 + vec3<f32>(0.59)) + vec3<f32>(0.14);\n";
                s << "    baseColor = clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));\n";
            }
            // TONEMAP_LINEAR: just exposure multiplication (done above)
        }

        // Fog (applied after tone mapping, mixes with fog color based on distance)
        if (features & FEAT_FOG_LINEAR) {
            s << "    {\n";
            s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
            s << "        let fogFactor = clamp((material.fogParams.y - fogDist) / (material.fogParams.y - material.fogParams.x), 0.0, 1.0);\n";
            s << "        baseColor = mix(material.fogColor.rgb, baseColor, fogFactor);\n";
            s << "    }\n";
        } else if (features & FEAT_FOG_EXP2) {
            s << "    {\n";
            s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
            s << "        let fogDensity = material.fogParams.z;\n";
            s << "        let fogFactor = exp(-fogDensity * fogDensity * fogDist * fogDist);\n";
            s << "        baseColor = mix(material.fogColor.rgb, baseColor, clamp(fogFactor, 0.0, 1.0));\n";
            s << "    }\n";
        }

        // sRGB output encoding (linear to sRGB gamma)
        if (features & FEAT_SRGB_OUTPUT) {
            s << "    baseColor = pow(baseColor, vec3<f32>(1.0 / 2.2));\n";
        }

        s << "    return vec4<f32>(baseColor, opacity);\n}\n";
        return s.str();
    }

}// namespace


struct DawnRenderer::Impl {

    DawnRenderer& scope;
    Canvas& canvas;

    // Core WebGPU objects
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;

    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;

    WindowSize size_;
    float pixelRatio_ = 1.0f;
    Color clearColor_{0x000000};
    float clearAlpha_ = 1.0f;

    // MSAA configuration
    uint32_t sampleCount_ = 1;

    // Viewport & scissor state (Feature 4)
    struct { float x=0, y=0, w=0, h=0; } viewport_;
    struct { uint32_t x=0, y=0, w=0, h=0; } scissor_;
    bool scissorTest_ = false;

    // Uniform buffers
    WGPUBuffer lightBuffer = nullptr;       // binding 2 (per-frame, shared)

    // Pipeline cache keyed by feature bitmask
    struct PipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
    };
    std::unordered_map<uint64_t, PipelineEntry> pipelineCache;

    // Subsystem: shared state
    dawn::DawnState dawnState;

    // Subsystem: geometry buffer management (with version-based updates)
    std::unique_ptr<dawn::DawnGeometries> geometries;

    // Subsystem: texture upload, caching, version tracking
    std::unique_ptr<dawn::DawnTextures> textures;

    // Render target cache (Feature 5)
    struct RTEntry {
        WGPUTexture colorTexture = nullptr;       // resolve target (1x) — readback source
        WGPUTextureView colorView = nullptr;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        // MSAA textures (only created when sampleCount > 1)
        WGPUTexture msaaColorTexture = nullptr;   // multi-sampled render target
        WGPUTextureView msaaColorView = nullptr;
        WGPUTexture msaaDepthTexture = nullptr;
        WGPUTextureView msaaDepthView = nullptr;
        unsigned int width = 0, height = 0;
        uint32_t sampleCount = 1;
    };
    std::unordered_map<std::string, RTEntry> rtCache;
    RenderTarget* currentRenderTarget_ = nullptr;

    // Light state

    // Render target state
    int activeCubeFace_ = 0;
    int activeMipmapLevel_ = 0;

    // Shadow mapping state (supports multiple shadow-casting lights)
    struct ShadowLightEntry {
        WGPUTextureView layerView = nullptr;  // per-layer view for rendering
        Matrix4 lightVP;
        float bias = 0.005f;
        float normalBias = 0.0f;
    };
    struct ShadowState {
        ShadowLightEntry lights[MAX_SHADOW_LIGHTS];
        int activeLightCount = 0;
        // Single 2D array texture with MAX_SHADOW_LIGHTS layers
        WGPUTexture depthArrayTexture = nullptr;
        WGPUTextureView depthArrayView = nullptr;  // full array view for sampling in fragment shader
        WGPUSampler comparisonSampler = nullptr;
        WGPUBuffer uniformBuffer = nullptr;
        WGPURenderPipeline depthPipeline = nullptr;
        WGPUPipelineLayout depthPipelineLayout = nullptr;
        WGPUBindGroupLayout depthBindGroupLayout = nullptr;
        WGPUShaderModule depthShader = nullptr;
        WGPUBuffer depthTransformBuffer = nullptr;
        bool active = false;
        bool initialized = false;
    } shadowState;

    // Frustum culling
    Frustum frustum_;
    Vector3 _vector3;

    // Render list for opaque/transparent sorting
    RenderList renderList_;

    // Custom shader pipeline cache (for ShaderMaterial)
    struct CustomPipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        size_t shaderHash = 0;
        // Track the bind group layout structure
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
    };
    std::unordered_map<Material*, CustomPipelineEntry> customPipelineCache;

    // Render info/statistics
    struct {
        size_t frame = 0;
        size_t calls = 0;
        size_t triangles = 0;
        size_t lines = 0;
        size_t points = 0;
        size_t geometries = 0;
        size_t textures = 0;
    } renderInfo;

    bool initialized = false;

    explicit Impl(DawnRenderer& scope, Canvas& canvas)
        : scope(scope), canvas(canvas), size_(canvas.size()) {

        viewport_.w = static_cast<float>(size_.width());
        viewport_.h = static_cast<float>(size_.height());
        scissor_.w = static_cast<uint32_t>(size_.width());
        scissor_.h = static_cast<uint32_t>(size_.height());

        initWebGPU();
    }

    void initWebGPU() {
        // Create instance with primary backends (Vulkan/Metal/DX12).
        // Avoid GL backend as it conflicts with GLFW's GL context.
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            std::cerr << "DawnRenderer: Failed to create WebGPU instance" << std::endl;
            return;
        }

        // Try to create surface from GLFW window.
        // Surface creation may fail in headless environments — that's OK,
        // the renderer can still operate with render targets only.
        createSurface();

        // Request adapter (surface is optional — nullptr works for offscreen)
        requestAdapter();
        if (!adapter) {
            std::cerr << "DawnRenderer: Failed to get adapter" << std::endl;
            return;
        }

        // Request device
        requestDevice();
        if (!device) {
            std::cerr << "DawnRenderer: Failed to get device" << std::endl;
            return;
        }

        queue = wgpuDeviceGetQueue(device);

        // Configure surface (only if we have one)
        if (surface) {
            configureSurface();
        }

        // Populate shared state for subsystems
        dawnState.device = device;
        dawnState.queue = queue;
        dawnState.surfaceFormat = surfaceFormat;

        // Create uniform buffers
        createUniformBuffers();

        // Initialize subsystems
        textures = std::make_unique<dawn::DawnTextures>(dawnState);
        textures->createDummyTexture();

        geometries = std::make_unique<dawn::DawnGeometries>(dawnState);

        initialized = true;
        std::cout << "DawnRenderer: WebGPU initialized successfully"
                  << (surface ? "" : " (headless, no surface)") << std::endl;
    }

    void createSurface() {
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        WGPUSurfaceDescriptor surfDesc{};
        WGPUStringView label = {.data = "threepp_surface", .length = 15};
        surfDesc.label = label;

#if defined(__linux__)
        Display* x11Display = glfwGetX11Display();
        ::Window x11Window = glfwGetX11Window(glfwWindow);

        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.chain.next = nullptr;
        xlibSource.display = x11Display;
        xlibSource.window = static_cast<uint64_t>(x11Window);
        surfDesc.nextInChain = &xlibSource.chain;

#elif defined(_WIN32)
        WGPUSurfaceSourceWindowsHWND hwndSource{};
        hwndSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        hwndSource.chain.next = nullptr;
        hwndSource.hinstance = GetModuleHandle(nullptr);
        hwndSource.hwnd = glfwGetWin32Window(glfwWindow);
        surfDesc.nextInChain = &hwndSource.chain;

#elif defined(__APPLE__)
        void* metalLayer = dawn_create_metal_layer(glfwGetCocoaWindow(glfwWindow));
        WGPUSurfaceSourceMetalLayer metalSource{};
        metalSource.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
        metalSource.chain.next = nullptr;
        metalSource.layer = metalLayer;
        surfDesc.nextInChain = &metalSource.chain;
#endif

        surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    void requestAdapter() {
        struct UserData {
            WGPUAdapter adapter = nullptr;
            bool done = false;
        } userData;

        WGPURequestAdapterOptions options{};
        options.compatibleSurface = surface; // nullptr in headless mode

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                ud->adapter = adapter;
            } else {
                std::cerr << "DawnRenderer: Adapter request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuInstanceRequestAdapter(instance, &options, callbackInfo);

        // Poll until callback fires, with timeout
        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestAdapter timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.adapter) {
            throw std::runtime_error("DawnRenderer: failed to obtain adapter");
        }
        adapter = userData.adapter;
    }

    void requestDevice() {
        struct UserData {
            WGPUDevice device = nullptr;
            bool done = false;
        } userData;

        WGPUDeviceDescriptor deviceDesc{};
        WGPUStringView label = {.data = "threepp_device", .length = 14};
        deviceDesc.label = label;

        WGPURequestDeviceCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success) {
                ud->device = device;
            } else {
                std::cerr << "DawnRenderer: Device request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuAdapterRequestDevice(adapter, &deviceDesc, callbackInfo);

        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestDevice timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.device) {
            throw std::runtime_error("DawnRenderer: failed to obtain device");
        }
        device = userData.device;
    }

    void configureSurface() {
        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width = static_cast<uint32_t>(std::floor(size_.width() * pixelRatio_));
        config.height = static_cast<uint32_t>(std::floor(size_.height() * pixelRatio_));
        config.presentMode = WGPUPresentMode_Fifo;
        config.alphaMode = WGPUCompositeAlphaMode_Auto;
        config.viewFormatCount = 0;
        config.viewFormats = nullptr;

        wgpuSurfaceConfigure(surface, &config);
    }

    PipelineEntry& getOrCreatePipeline(uint64_t features) {
        auto it = pipelineCache.find(features);
        if (it != pipelineCache.end()) return it->second;

        PipelineEntry entry{};

        // Shader module
        std::string wgsl = buildWGSL(features);
        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.chain.next = nullptr;
        wgslSource.code = {.data = wgsl.c_str(), .length = wgsl.size()};

        WGPUShaderModuleDescriptor shaderDesc{};
        shaderDesc.nextInChain = &wgslSource.chain;
        WGPUStringView shaderLabel = {.data = "dawn_shader", .length = 11};
        shaderDesc.label = shaderLabel;
        entry.shader = wgpuDeviceCreateShaderModule(device, &shaderDesc);
        if (!entry.shader) {
            std::cerr << "DawnRenderer: Failed to create shader module for features 0x"
                      << std::hex << features << std::dec << std::endl;
            return pipelineCache[features]; // return default-initialized entry
        }

        // Bind group layout entries
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);

        // Binding 0: transform uniforms
        { WGPUBindGroupLayoutEntry e{}; e.binding = 0;
          e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
          e.buffer.type = WGPUBufferBindingType_Uniform;
          e.buffer.minBindingSize = TRANSFORM_UNIFORM_SIZE;
          bglEntries.push_back(e); }

        // Binding 1: material uniforms (vertex stage needed for displacement map)
        { WGPUBindGroupLayoutEntry e{}; e.binding = 1;
          e.visibility = WGPUShaderStage_Fragment |
                         ((features & FEAT_DISPLACEMENT_MAP) ? WGPUShaderStage_Vertex : 0);
          e.buffer.type = WGPUBufferBindingType_Uniform;
          e.buffer.minBindingSize = MATERIAL_UNIFORM_SIZE;
          bglEntries.push_back(e); }

        // Binding 2: light uniforms (if lit)
        if (lit) {
            WGPUBindGroupLayoutEntry e{}; e.binding = 2;
            e.visibility = WGPUShaderStage_Fragment;
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.minBindingSize = LIGHT_UNIFORM_SIZE;
            bglEntries.push_back(e);
        }

        // Binding 3: texture, Binding 4: sampler (if textured)
        if (features & FEAT_TEXTURE) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 3;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 4;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        // Binding 7-9: shadow map (if shadows enabled)
        if (features & FEAT_SHADOW) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 7;
              e.visibility = WGPUShaderStage_Fragment;
              e.buffer.type = WGPUBufferBindingType_Uniform;
              e.buffer.minBindingSize = SHADOW_UNIFORM_SIZE;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 8;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Depth;
              e.texture.viewDimension = WGPUTextureViewDimension_2DArray;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 9;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Comparison;
              bglEntries.push_back(e); }
        }

        // Binding 5: normal map, Binding 6: normal map sampler
        if (features & FEAT_NORMAL_MAP) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 5;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 6;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        // Helper lambda to add texture+sampler binding pair
        auto addTexSamplerBindings = [&](uint32_t texBinding, uint32_t sampBinding) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = texBinding;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = sampBinding;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        };

        if (features & FEAT_EMISSIVE_MAP)  addTexSamplerBindings(10, 11);
        if (features & FEAT_ROUGHNESS_MAP) addTexSamplerBindings(12, 13);
        if (features & FEAT_METALNESS_MAP) addTexSamplerBindings(14, 15);
        if (features & FEAT_AO_MAP)        addTexSamplerBindings(16, 17);
        if (features & FEAT_ALPHA_MAP)     addTexSamplerBindings(18, 19);
        if (features & FEAT_SPECULAR_MAP)  addTexSamplerBindings(20, 21);
        if (features & FEAT_LIGHT_MAP)     addTexSamplerBindings(22, 23);
        if (features & FEAT_BUMP_MAP)      addTexSamplerBindings(24, 25);
        if (features & FEAT_GRADIENT_MAP)  addTexSamplerBindings(26, 27);

        if (features & FEAT_INSTANCED) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 28;
            e.visibility = WGPUShaderStage_Vertex;
            e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            bglEntries.push_back(e);
        }

        if (features & FEAT_MORPH_TARGETS) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 29;
            e.visibility = WGPUShaderStage_Vertex;
            e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            bglEntries.push_back(e);
        }

        if (features & FEAT_DISPLACEMENT_MAP) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 30;
              e.visibility = WGPUShaderStage_Vertex;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 31;
              e.visibility = WGPUShaderStage_Vertex;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        if (features & FEAT_ENV_MAP) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 32;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_Cube;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 33;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        if (features & FEAT_SKINNING) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 34;
              e.visibility = WGPUShaderStage_Vertex;
              e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 35;
              e.visibility = WGPUShaderStage_Vertex;
              e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
              bglEntries.push_back(e); }
        }

        WGPUBindGroupLayoutDescriptor bglDesc{};
        WGPUStringView bglLabel = {.data = "bind_group_layout", .length = 17};
        bglDesc.label = bglLabel;
        bglDesc.entryCount = bglEntries.size();
        bglDesc.entries = bglEntries.data();
        entry.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        // Pipeline layout
        WGPUPipelineLayoutDescriptor plDesc{};
        WGPUStringView plLabel = {.data = "pipeline_layout", .length = 15};
        plDesc.label = plLabel;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &entry.bindGroupLayout;
        entry.layout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex buffer layout: pos(vec3) + normal(vec3) + uv(vec2) + color(vec3) = 44 bytes
        WGPUVertexAttribute attrs[4]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0; attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
        attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = dawn::VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 4;
        vbLayout.attributes = attrs;

        // Blend state — driven by the blend bits in the pipeline key
        WGPUBlendState blendState{};
        uint32_t blendBits = features & BLEND_MASK;
        WGPUColorTargetState colorTarget{};
        colorTarget.format = surfaceFormat;
        colorTarget.writeMask = WGPUColorWriteMask_All;
        if (blendBits == BLEND_DISABLED) {
            // No blending
            colorTarget.blend = nullptr;
        } else {
            if (blendBits == BLEND_ADDITIVE) {
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_One;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == BLEND_SUBTRACTIVE) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrc;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == BLEND_MULTIPLY) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_Src;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else {
                // BLEND_NORMAL (default)
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            }
            colorTarget.blend = &blendState;
        }

        WGPUStringView fsEntry = {.data = "fs_main", .length = 7};
        WGPUFragmentState fragmentState{};
        fragmentState.module = entry.shader;
        fragmentState.entryPoint = fsEntry;
        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth24Plus;
        depthStencil.depthWriteEnabled = (features & DEPTH_WRITE_OFF)
                                          ? WGPUOptionalBool_False
                                          : WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        WGPURenderPipelineDescriptor pipelineDesc{};
        WGPUStringView pipeLabel = {.data = "dawn_pipeline", .length = 13};
        pipelineDesc.label = pipeLabel;
        pipelineDesc.layout = entry.layout;

        WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
        pipelineDesc.vertex.module = entry.shader;
        pipelineDesc.vertex.entryPoint = vsEntry;
        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &vbLayout;

        // Topology selection: wireframe, line, points, or triangles
        uint32_t topoBits = features & TOPO_MASK;
        if (features & WIREFRAME_BIT) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == TOPO_LINE_LIST) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == TOPO_LINE_STRIP) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineStrip;
            pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Uint32;
        } else if (topoBits == TOPO_POINT_LIST) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_PointList;
        } else {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        }
        pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;

        // Face culling from pipeline key
        uint32_t cullBits = features & CULL_MASK;
        if (cullBits == CULL_FRONT) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Front;
        } else if (cullBits == CULL_BACK) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Back;
        } else {
            pipelineDesc.primitive.cullMode = WGPUCullMode_None;
        }
        pipelineDesc.depthStencil = &depthStencil;
        pipelineDesc.multisample.count = sampleCount_;
        pipelineDesc.multisample.mask = 0xFFFFFFFF;
        pipelineDesc.fragment = &fragmentState;

        entry.pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
        if (!entry.pipeline) {
            std::cerr << "DawnRenderer: Failed to create render pipeline for features 0x"
                      << std::hex << features << std::dec << std::endl;
        }

        pipelineCache[features] = entry;
        return pipelineCache[features];
    }

    void createUniformBuffers() {
        auto makeBuffer = [&](const char* name, size_t nameLen, size_t sz) {
            WGPUBufferDescriptor d{};
            d.label = {.data = name, .length = nameLen};
            d.size = sz;
            d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            return wgpuDeviceCreateBuffer(device, &d);
        };
        lightBuffer     = makeBuffer("light_buf", 9, LIGHT_UNIFORM_SIZE);
    }

    // NOTE: Texture upload/cache and geometry buffers are in DawnTextures and DawnGeometries.
    // Shadow map helpers
    void initShadowMap() {
        if (shadowState.initialized) return; // already initialized
        shadowState.initialized = true;

        // Create a single 2D array depth texture with MAX_SHADOW_LIGHTS layers
        {
            WGPUTextureDescriptor td{};
            td.label = {.data = "shadow_depth_array", .length = 18};
            td.size = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, MAX_SHADOW_LIGHTS};
            td.mipLevelCount = 1;
            td.sampleCount = 1;
            td.dimension = WGPUTextureDimension_2D;
            td.format = WGPUTextureFormat_Depth32Float;
            td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
            shadowState.depthArrayTexture = wgpuDeviceCreateTexture(device, &td);

            // Create full array view for sampling in fragment shader
            WGPUTextureViewDescriptor avd{};
            avd.label = {.data = "shadow_array_view", .length = 17};
            avd.format = WGPUTextureFormat_Depth32Float;
            avd.dimension = WGPUTextureViewDimension_2DArray;
            avd.baseMipLevel = 0;
            avd.mipLevelCount = 1;
            avd.baseArrayLayer = 0;
            avd.arrayLayerCount = MAX_SHADOW_LIGHTS;
            avd.aspect = WGPUTextureAspect_DepthOnly;
            shadowState.depthArrayView = wgpuTextureCreateView(shadowState.depthArrayTexture, &avd);

            // Create per-layer views for rendering shadow passes
            for (int i = 0; i < MAX_SHADOW_LIGHTS; i++) {
                WGPUTextureViewDescriptor lvd{};
                lvd.label = {.data = "shadow_layer_view", .length = 17};
                lvd.format = WGPUTextureFormat_Depth32Float;
                lvd.dimension = WGPUTextureViewDimension_2D;
                lvd.baseMipLevel = 0;
                lvd.mipLevelCount = 1;
                lvd.baseArrayLayer = static_cast<uint32_t>(i);
                lvd.arrayLayerCount = 1;
                lvd.aspect = WGPUTextureAspect_DepthOnly;
                shadowState.lights[i].layerView = wgpuTextureCreateView(shadowState.depthArrayTexture, &lvd);
            }
        }

        // Create comparison sampler
        WGPUSamplerDescriptor sd{};
        sd.label = {.data = "shadow_samp", .length = 11};
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.magFilter = WGPUFilterMode_Linear;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.compare = WGPUCompareFunction_Less;
        sd.maxAnisotropy = 1; // WebGPU requires maxAnisotropy >= 1
        shadowState.comparisonSampler = wgpuDeviceCreateSampler(device, &sd);

        // Create shadow uniform buffer
        WGPUBufferDescriptor bd{};
        bd.label = {.data = "shadow_ub", .length = 9};
        bd.size = SHADOW_UNIFORM_SIZE;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        shadowState.uniformBuffer = wgpuDeviceCreateBuffer(device, &bd);

        // Create depth-only transform buffer
        bd.label = {.data = "shadow_xform", .length = 12};
        bd.size = 64; // just the lightVP matrix
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        shadowState.depthTransformBuffer = wgpuDeviceCreateBuffer(device, &bd);

        // Create depth-only render pipeline
        std::string depthWGSL = R"(
struct DepthUniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: DepthUniforms;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec3<f32> };
@vertex fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}
@fragment fn fs_main() {}
)";

        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.code = {.data = depthWGSL.c_str(), .length = depthWGSL.size()};

        WGPUShaderModuleDescriptor smd{};
        smd.nextInChain = &wgslSource.chain;
        smd.label = {.data = "shadow_shader", .length = 13};
        shadowState.depthShader = wgpuDeviceCreateShaderModule(device, &smd);

        // Bind group layout: one uniform buffer
        WGPUBindGroupLayoutEntry bglEntry{};
        bglEntry.binding = 0;
        bglEntry.visibility = WGPUShaderStage_Vertex;
        bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
        bglEntry.buffer.minBindingSize = 64;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = {.data = "shadow_bgl", .length = 10};
        bglDesc.entryCount = 1;
        bglDesc.entries = &bglEntry;
        shadowState.depthBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = {.data = "shadow_pl", .length = 9};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &shadowState.depthBindGroupLayout;
        shadowState.depthPipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex layout (same as main pipeline)
        WGPUVertexAttribute attrs[4]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0; attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
        attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = dawn::VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 4;
        vbLayout.attributes = attrs;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth32Float;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        WGPURenderPipelineDescriptor pipeDesc{};
        pipeDesc.label = {.data = "shadow_pipe", .length = 11};
        pipeDesc.layout = shadowState.depthPipelineLayout;

        WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
        pipeDesc.vertex.module = shadowState.depthShader;
        pipeDesc.vertex.entryPoint = vsEntry;
        pipeDesc.vertex.bufferCount = 1;
        pipeDesc.vertex.buffers = &vbLayout;

        pipeDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipeDesc.primitive.frontFace = WGPUFrontFace_CCW;
        pipeDesc.primitive.cullMode = WGPUCullMode_Front; // Render back faces to prevent shadow acne
        pipeDesc.depthStencil = &depthStencil;
        pipeDesc.multisample.count = 1;  // Shadow pass always 1x (renders to depth-only texture)
        pipeDesc.multisample.mask = 0xFFFFFFFF;
        // No fragment state needed for depth-only pass
        pipeDesc.fragment = nullptr;

        shadowState.depthPipeline = wgpuDeviceCreateRenderPipeline(device, &pipeDesc);
    }

    void renderShadowPass(WGPUCommandEncoder encoder, Object3D& scene, const Matrix4& lightVP, int lightIndex = 0) {
        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = shadowState.lights[lightIndex].layerView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = {.data = "shadow_pass", .length = 11};
        passDesc.colorAttachmentCount = 0;
        passDesc.colorAttachments = nullptr;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
        wgpuRenderPassEncoderSetViewport(pass, 0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0.0f, 1.0f);
        wgpuRenderPassEncoderSetPipeline(pass, shadowState.depthPipeline);

        renderShadowObject(pass, scene, lightVP);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

    void renderShadowObject(WGPURenderPassEncoder pass, Object3D& object, const Matrix4& lightVP) {
        if (auto mesh = object.as<Mesh>()) {
            auto geometry = mesh->geometry();
            if (mesh->castShadow && geometry && geometry->hasAttribute("position")) {
                // Compute MVP for this mesh from light's perspective
                Matrix4 mvp;
                mvp.multiplyMatrices(lightVP, *mesh->matrixWorld);

                // Upload MVP matrix
                wgpuQueueWriteBuffer(queue, shadowState.depthTransformBuffer, 0, mvp.elements.data(), 64);

                // Create bind group
                WGPUBindGroupEntry entry{};
                entry.binding = 0;
                entry.buffer = shadowState.depthTransformBuffer;
                entry.offset = 0;
                entry.size = 64;

                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.label = {.data = "shadow_bg", .length = 9};
                bgDesc.layout = shadowState.depthBindGroupLayout;
                bgDesc.entryCount = 1;
                bgDesc.entries = &entry;
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

                auto& gb = geometries->getOrCreateGeometryBuffers(geometry.get());
                if (gb.vertexBuffer) {
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                         gb.vertexCount * dawn::VERTEX_STRIDE);
                    if (gb.indexBuffer) {
                        wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                             WGPUIndexFormat_Uint32, 0,
                                                             gb.indexCount * sizeof(uint32_t));
                        wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                    } else {
                        wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                    }
                }
                wgpuBindGroupRelease(bg);
            }
        }

        for (auto& child : object.children) {
            renderShadowObject(pass, *child, lightVP);
        }
    }

    void disposeShadowMap() {
        if (shadowState.initialized) {
            // Release per-layer views
            for (int i = 0; i < MAX_SHADOW_LIGHTS; i++) {
                if (shadowState.lights[i].layerView) {
                    wgpuTextureViewRelease(shadowState.lights[i].layerView);
                }
            }
            // Release array texture and view
            if (shadowState.depthArrayView) wgpuTextureViewRelease(shadowState.depthArrayView);
            if (shadowState.depthArrayTexture) wgpuTextureRelease(shadowState.depthArrayTexture);
            if (shadowState.comparisonSampler) wgpuSamplerRelease(shadowState.comparisonSampler);
            if (shadowState.uniformBuffer) wgpuBufferRelease(shadowState.uniformBuffer);
            if (shadowState.depthTransformBuffer) wgpuBufferRelease(shadowState.depthTransformBuffer);
            if (shadowState.depthPipeline) wgpuRenderPipelineRelease(shadowState.depthPipeline);
            if (shadowState.depthPipelineLayout) wgpuPipelineLayoutRelease(shadowState.depthPipelineLayout);
            if (shadowState.depthBindGroupLayout) wgpuBindGroupLayoutRelease(shadowState.depthBindGroupLayout);
            if (shadowState.depthShader) wgpuShaderModuleRelease(shadowState.depthShader);
            shadowState = {};
        }
    }

    // Render target helpers (Feature 5)
    void releaseRTEntry(RTEntry& e) {
        if (e.msaaColorView) wgpuTextureViewRelease(e.msaaColorView);
        if (e.msaaColorTexture) wgpuTextureRelease(e.msaaColorTexture);
        if (e.msaaDepthView) wgpuTextureViewRelease(e.msaaDepthView);
        if (e.msaaDepthTexture) wgpuTextureRelease(e.msaaDepthTexture);
        if (e.colorView) wgpuTextureViewRelease(e.colorView);
        if (e.colorTexture) wgpuTextureRelease(e.colorTexture);
        if (e.depthView) wgpuTextureViewRelease(e.depthView);
        if (e.depthTexture) wgpuTextureRelease(e.depthTexture);
    }

    RTEntry& getOrCreateRT(RenderTarget* rt) {
        auto it = rtCache.find(rt->uuid);
        if (it != rtCache.end() && it->second.width == rt->width
            && it->second.height == rt->height && it->second.sampleCount == sampleCount_) {
            return it->second;
        }
        // Release old
        if (it != rtCache.end()) {
            releaseRTEntry(it->second);
        }

        RTEntry entry{};
        entry.width = rt->width;
        entry.height = rt->height;
        entry.sampleCount = sampleCount_;

        // Resolve target (always 1x) — used for readback and texture binding
        WGPUTextureDescriptor ctd{};
        ctd.label = {.data = "rt_color", .length = 8};
        ctd.size = {rt->width, rt->height, 1};
        ctd.mipLevelCount = 1;
        ctd.sampleCount = 1;
        ctd.dimension = WGPUTextureDimension_2D;
        ctd.format = surfaceFormat;
        ctd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
        entry.colorTexture = wgpuDeviceCreateTexture(device, &ctd);
        entry.colorView = wgpuTextureCreateView(entry.colorTexture, nullptr);

        WGPUTextureDescriptor dtd{};
        dtd.label = {.data = "rt_depth", .length = 8};
        dtd.size = {rt->width, rt->height, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = sampleCount_;
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        entry.depthTexture = wgpuDeviceCreateTexture(device, &dtd);
        entry.depthView = wgpuTextureCreateView(entry.depthTexture, nullptr);

        // MSAA color texture (multi-sampled render attachment)
        if (sampleCount_ > 1) {
            WGPUTextureDescriptor msaaCtd{};
            msaaCtd.label = {.data = "rt_msaa_color", .length = 13};
            msaaCtd.size = {rt->width, rt->height, 1};
            msaaCtd.mipLevelCount = 1;
            msaaCtd.sampleCount = sampleCount_;
            msaaCtd.dimension = WGPUTextureDimension_2D;
            msaaCtd.format = surfaceFormat;
            msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
            entry.msaaColorTexture = wgpuDeviceCreateTexture(device, &msaaCtd);
            entry.msaaColorView = wgpuTextureCreateView(entry.msaaColorTexture, nullptr);
        }

        rtCache[rt->uuid] = entry;
        return rtCache[rt->uuid];
    }

    // Pack light data into GPU buffer using world-space coordinates.
    // Unlike GLRenderer which uses view-space via Lights::setupView(), the Dawn
    // renderer computes lighting in world space, so we extract world-space
    // positions/directions directly from the light objects.
    void uploadLightDataWorldSpace(Object3D& scene) {
        std::vector<float> data(LIGHT_UNIFORM_SIZE / sizeof(float), 0.0f);
        auto* u32 = reinterpret_cast<uint32_t*>(data.data());

        uint32_t nDir = 0, nPt = 0, nSp = 0, nHm = 0;
        float ambR = 0, ambG = 0, ambB = 0;

        // Temporary storage
        struct DirEntry { Vector3 dir; Color col; };
        struct PtEntry  { Vector3 pos; Color col; float dist; float decay; };
        struct SpEntry  { Vector3 pos; Vector3 dir; Color col; float dist; float decay; float coneCos; float penumbraCos; };
        struct HmEntry  { Vector3 dir; Color sky; Color gnd; };
        std::vector<DirEntry> dirs;
        std::vector<PtEntry>  pts;
        std::vector<SpEntry>  sps;
        std::vector<HmEntry>  hms;

        std::function<void(Object3D&)> collect = [&](Object3D& obj) {
            if (auto al = obj.as<AmbientLight>()) {
                ambR += al->color.r * al->intensity;
                ambG += al->color.g * al->intensity;
                ambB += al->color.b * al->intensity;
            } else if (auto dl = obj.as<DirectionalLight>()) {
                if (dirs.size() < static_cast<size_t>(MAX_DIR_LIGHTS)) {
                    Vector3 lightPos, targetPos;
                    lightPos.setFromMatrixPosition(*dl->matrixWorld);
                    targetPos.setFromMatrixPosition(*dl->target().matrixWorld);
                    Vector3 direction = lightPos.clone().sub(targetPos).normalize();
                    dirs.push_back({direction, Color(dl->color).multiplyScalar(dl->intensity)});
                }
            } else if (auto pl = obj.as<PointLight>()) {
                if (pts.size() < static_cast<size_t>(MAX_POINT_LIGHTS)) {
                    Vector3 pos;
                    pos.setFromMatrixPosition(*pl->matrixWorld);
                    pts.push_back({pos, Color(pl->color).multiplyScalar(pl->intensity), pl->distance, pl->decay});
                }
            } else if (auto sl = obj.as<SpotLight>()) {
                if (sps.size() < static_cast<size_t>(MAX_SPOT_LIGHTS)) {
                    Vector3 pos, targetPos;
                    pos.setFromMatrixPosition(*sl->matrixWorld);
                    targetPos.setFromMatrixPosition(*sl->target().matrixWorld);
                    Vector3 direction = pos.clone().sub(targetPos).normalize();
                    sps.push_back({pos, direction, Color(sl->color).multiplyScalar(sl->intensity),
                                   sl->distance, sl->decay,
                                   std::cos(sl->angle), std::cos(sl->angle * (1.0f - sl->penumbra))});
                }
            } else if (auto hl = obj.as<HemisphereLight>()) {
                if (hms.size() < static_cast<size_t>(MAX_HEMI_LIGHTS)) {
                    Vector3 dir;
                    dir.setFromMatrixPosition(*hl->matrixWorld).normalize();
                    hms.push_back({dir, Color(hl->color).multiplyScalar(hl->intensity),
                                   Color(hl->groundColor).multiplyScalar(hl->intensity)});
                }
            }
            for (auto& child : obj.children) collect(*child);
        };
        collect(scene);

        nDir = dirs.size(); nPt = pts.size(); nSp = sps.size(); nHm = hms.size();
        u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
        data[4] = ambR; data[5] = ambG; data[6] = ambB; data[7] = 0;

        size_t off = 8;
        for (uint32_t i = 0; i < nDir; i++) {
            data[off+0] = dirs[i].dir.x; data[off+1] = dirs[i].dir.y; data[off+2] = dirs[i].dir.z; data[off+3] = 0;
            data[off+4] = dirs[i].col.r; data[off+5] = dirs[i].col.g; data[off+6] = dirs[i].col.b; data[off+7] = 0;
            off += 8;
        }

        off = 8 + MAX_DIR_LIGHTS * 8;
        for (uint32_t i = 0; i < nPt; i++) {
            data[off+0] = pts[i].pos.x; data[off+1] = pts[i].pos.y; data[off+2] = pts[i].pos.z; data[off+3] = 0;
            data[off+4] = pts[i].col.r; data[off+5] = pts[i].col.g; data[off+6] = pts[i].col.b; data[off+7] = pts[i].dist;
            data[off+8] = pts[i].decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
            off += 12;
        }

        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12;
        for (uint32_t i = 0; i < nSp; i++) {
            data[off+0] = sps[i].pos.x; data[off+1] = sps[i].pos.y; data[off+2] = sps[i].pos.z; data[off+3] = 0;
            data[off+4] = sps[i].dir.x; data[off+5] = sps[i].dir.y; data[off+6] = sps[i].dir.z; data[off+7] = 0;
            data[off+8] = sps[i].col.r; data[off+9] = sps[i].col.g; data[off+10] = sps[i].col.b; data[off+11] = sps[i].dist;
            data[off+12] = sps[i].decay; data[off+13] = sps[i].coneCos; data[off+14] = sps[i].penumbraCos; data[off+15] = 0;
            off += 16;
        }

        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12 + MAX_SPOT_LIGHTS * 16;
        for (uint32_t i = 0; i < nHm; i++) {
            data[off+0] = hms[i].dir.x; data[off+1] = hms[i].dir.y; data[off+2] = hms[i].dir.z; data[off+3] = 0;
            data[off+4] = hms[i].sky.r; data[off+5] = hms[i].sky.g; data[off+6] = hms[i].sky.b; data[off+7] = 0;
            data[off+8] = hms[i].gnd.r; data[off+9] = hms[i].gnd.g; data[off+10] = hms[i].gnd.b; data[off+11] = 0;
            off += 12;
        }

        wgpuQueueWriteBuffer(queue, lightBuffer, 0, data.data(), LIGHT_UNIFORM_SIZE);
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        // Reset per-frame statistics
        renderInfo.frame++;
        renderInfo.calls = 0;
        renderInfo.triangles = 0;
        renderInfo.lines = 0;
        renderInfo.points = 0;
        renderInfo.geometries = geometries->count();
        renderInfo.textures = textures->count();

        // Update window size if changed
        auto currentSize = canvas.size();
        if (currentSize.width() != size_.width() || currentSize.height() != size_.height()) {
            size_ = currentSize;
            if (surface) configureSurface();
            viewport_.w = static_cast<float>(size_.width());
            viewport_.h = static_cast<float>(size_.height());
            scissor_.w = static_cast<uint32_t>(size_.width());
            scissor_.h = static_cast<uint32_t>(size_.height());
        }

        // Update matrices
        scene.updateMatrixWorld();
        if (!camera.parent) {
            camera.updateMatrixWorld();
        }
        camera.updateWorldMatrix(false, false);

        Matrix4 projectionMatrix = camera.projectionMatrix;
        Matrix4 viewMatrix = camera.matrixWorldInverse;

        // Remap NDC z from [-1,1] (OpenGL convention used by three.js matrices)
        // to [0,1] (WebGPU/Vulkan convention). Apply: z' = 0.5*z + 0.5*w
        {
            auto& e = projectionMatrix.elements;
            e[2]  = 0.5f * e[2]  + 0.5f * e[3];
            e[6]  = 0.5f * e[6]  + 0.5f * e[7];
            e[10] = 0.5f * e[10] + 0.5f * e[11];
            e[14] = 0.5f * e[14] + 0.5f * e[15];
        }

        // Upload world-space light data directly from the scene
        uploadLightDataWorldSpace(scene);

        // Shadow pass: find all shadow-casting directional lights (up to MAX_SHADOW_LIGHTS)
        shadowState.active = false;
        shadowState.activeLightCount = 0;
        {
            // Collect all shadow-casting lights (DirectionalLight and SpotLight)
            struct ShadowCaster {
                Light* light;
                LightWithShadow* shadowInterface;
            };
            std::vector<ShadowCaster> shadowLights;
            std::function<void(Object3D&)> findShadowLights = [&](Object3D& obj) {
                if (obj.castShadow && static_cast<int>(shadowLights.size()) < MAX_SHADOW_LIGHTS) {
                    if (auto dl = obj.as<DirectionalLight>()) {
                        shadowLights.push_back({dl, dl});
                    } else if (auto sl = obj.as<SpotLight>()) {
                        shadowLights.push_back({sl, sl});
                    }
                }
                for (auto& child : obj.children) findShadowLights(*child);
            };
            findShadowLights(scene);

            if (!shadowLights.empty()) {
                initShadowMap();
                shadowState.active = true;
                shadowState.activeLightCount = static_cast<int>(shadowLights.size());

                // Build shadow uniform buffer: [count, pad, pad, pad, light0{VP,bias,normalBias,pad,pad}, light1{...}, ...]
                std::vector<float> shadowData(SHADOW_UNIFORM_SIZE / sizeof(float), 0.0f);
                // First 4 floats: count as int bits, then padding
                auto countBits = static_cast<uint32_t>(shadowLights.size());
                std::memcpy(&shadowData[0], &countBits, sizeof(uint32_t));

                for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
                    auto* light = shadowLights[i].light;
                    auto& shadow = shadowLights[i].shadowInterface->shadow;
                    shadow->updateMatrices(*light);

                    shadowState.lights[i].lightVP = shadow->matrix;
                    shadowState.lights[i].bias = shadow->bias;
                    shadowState.lights[i].normalBias = shadow->normalBias;

                    // Z-remap the light VP for the uniform buffer
                    Matrix4 shadowVP = shadow->matrix;
                    {
                        auto& e = shadowVP.elements;
                        e[2]  = 0.5f * e[2]  + 0.5f * e[3];
                        e[6]  = 0.5f * e[6]  + 0.5f * e[7];
                        e[10] = 0.5f * e[10] + 0.5f * e[11];
                        e[14] = 0.5f * e[14] + 0.5f * e[15];
                    }

                    // Offset into shadowData: 4 floats header + i * 20 floats per light
                    size_t offset = 4 + i * (SHADOW_UNIFORM_PER_LIGHT / sizeof(float));
                    std::memcpy(&shadowData[offset], shadowVP.elements.data(), 64);
                    shadowData[offset + 16] = shadow->bias;
                    shadowData[offset + 17] = shadow->normalBias;
                }
                wgpuQueueWriteBuffer(queue, shadowState.uniformBuffer, 0, shadowData.data(), SHADOW_UNIFORM_SIZE);

                // Render shadow depth pass for each light
                for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
                    auto& shadow = shadowLights[i].shadowInterface->shadow;

                    WGPUCommandEncoderDescriptor shadowEncDesc{};
                    shadowEncDesc.label = {.data = "shadow_enc", .length = 10};
                    WGPUCommandEncoder shadowEncoder = wgpuDeviceCreateCommandEncoder(device, &shadowEncDesc);

                    Matrix4 lightProj = shadow->camera->projectionMatrix;
                    {
                        auto& e = lightProj.elements;
                        e[2]  = 0.5f * e[2]  + 0.5f * e[3];
                        e[6]  = 0.5f * e[6]  + 0.5f * e[7];
                        e[10] = 0.5f * e[10] + 0.5f * e[11];
                        e[14] = 0.5f * e[14] + 0.5f * e[15];
                    }
                    Matrix4 lightVP;
                    lightVP.multiplyMatrices(lightProj, shadow->camera->matrixWorldInverse);

                    renderShadowPass(shadowEncoder, scene, lightVP, i);

                    WGPUCommandBufferDescriptor shadowCmdDesc{};
                    shadowCmdDesc.label = {.data = "shadow_cmd", .length = 10};
                    WGPUCommandBuffer shadowCmd = wgpuCommandEncoderFinish(shadowEncoder, &shadowCmdDesc);
                    wgpuQueueSubmit(queue, 1, &shadowCmd);
                    wgpuCommandBufferRelease(shadowCmd);
                    wgpuCommandEncoderRelease(shadowEncoder);
                }
            }
        }

        // Determine render target views
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTextureView resolveView = nullptr;  // MSAA resolve target (non-null when MSAA active)
        WGPUTexture frameDepthTexture = nullptr;
        WGPUTexture frameMsaaColorTexture = nullptr;
        WGPUTexture frameMsaaDepthTexture = nullptr;
        WGPUSurfaceTexture surfaceTexture{};
        bool useSurface = (currentRenderTarget_ == nullptr && surface != nullptr);

        if (currentRenderTarget_ == nullptr && surface == nullptr) {
            // Headless mode with no render target set — nothing to render to
            return;
        }

        if (useSurface) {
            wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
            if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
                std::cerr << "DawnRenderer: Failed to acquire surface texture (status "
                          << static_cast<int>(surfaceTexture.status) << ")" << std::endl;
                return;
            }
            WGPUTextureViewDescriptor vd{};
            vd.label = {.data = "surface_view", .length = 12};
            vd.format = surfaceFormat;
            vd.dimension = WGPUTextureViewDimension_2D;
            vd.baseMipLevel = 0; vd.mipLevelCount = 1;
            vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
            vd.aspect = WGPUTextureAspect_All;
            uint32_t w = static_cast<uint32_t>(size_.width());
            uint32_t h = static_cast<uint32_t>(size_.height());

            if (sampleCount_ > 1) {
                // MSAA: surface view becomes resolve target, create MSAA color texture
                resolveView = wgpuTextureCreateView(surfaceTexture.texture, &vd);

                WGPUTextureDescriptor msaaCtd{};
                msaaCtd.label = {.data = "frame_msaa_color", .length = 16};
                msaaCtd.size = {w, h, 1};
                msaaCtd.mipLevelCount = 1;
                msaaCtd.sampleCount = sampleCount_;
                msaaCtd.dimension = WGPUTextureDimension_2D;
                msaaCtd.format = surfaceFormat;
                msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
                frameMsaaColorTexture = wgpuDeviceCreateTexture(device, &msaaCtd);
                colorView = wgpuTextureCreateView(frameMsaaColorTexture, nullptr);
            } else {
                colorView = wgpuTextureCreateView(surfaceTexture.texture, &vd);
            }

            WGPUTextureDescriptor dtd{};
            dtd.label = {.data = "depth_tex", .length = 9};
            dtd.size = {w, h, 1};
            dtd.mipLevelCount = 1; dtd.sampleCount = sampleCount_;
            dtd.dimension = WGPUTextureDimension_2D;
            dtd.format = WGPUTextureFormat_Depth24Plus;
            dtd.usage = WGPUTextureUsage_RenderAttachment;
            frameDepthTexture = wgpuDeviceCreateTexture(device, &dtd);
            depthView = wgpuTextureCreateView(frameDepthTexture, nullptr);
        } else {
            auto& rt = getOrCreateRT(currentRenderTarget_);
            depthView = rt.depthView;
            if (sampleCount_ > 1 && rt.msaaColorView) {
                // MSAA: render into multi-sampled texture, resolve to 1x target
                colorView = rt.msaaColorView;
                resolveView = rt.colorView;
            } else {
                colorView = rt.colorView;
            }
        }

        // Command encoder
        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = {.data = "cmd_enc", .length = 7};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        // Determine clear color (scene background overrides if set)
        auto* sceneObj = scene.as<Scene>();
        Color effectiveClearColor = clearColor_;
        float effectiveClearAlpha = clearAlpha_;
        if (sceneObj && sceneObj->background.isColor()) {
            effectiveClearColor = sceneObj->background.color();
            effectiveClearAlpha = 1.0f;
        }

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = resolveView;  // non-null when MSAA is active
        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(effectiveClearColor.r),
                static_cast<double>(effectiveClearColor.g),
                static_cast<double>(effectiveClearColor.b),
                static_cast<double>(effectiveClearAlpha)};

        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = {.data = "render_pass", .length = 11};
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

        // Set viewport (Feature 4)
        wgpuRenderPassEncoderSetViewport(pass, viewport_.x, viewport_.y, viewport_.w, viewport_.h, 0.0f, 1.0f);
        if (scissorTest_) {
            wgpuRenderPassEncoderSetScissorRect(pass, scissor_.x, scissor_.y, scissor_.w, scissor_.h);
        }

        // Set up frustum for culling and collect renderables
        renderList_.init();
        Matrix4 projScreenMatrix;
        projScreenMatrix.multiplyMatrices(projectionMatrix, viewMatrix);
        frustum_.setFromProjectionMatrix(projScreenMatrix);
        collectRenderables(scene, projScreenMatrix, camera, 0);
        if (scope.sortObjects) {
            renderList_.sort();
        }
        renderList_.finish();

        // Extract fog and tone mapping state from scene
        Color fogColor;
        float fogNear = 0, fogFar = 0, fogDensity = 0;
        uint64_t fogBits = 0;
        if (sceneObj && sceneObj->fog) {
            if (auto* f = std::get_if<Fog>(&*sceneObj->fog)) {
                fogColor = f->color;
                fogNear = f->nearPlane;
                fogFar = f->farPlane;
                fogBits = FEAT_FOG_LINEAR;
            } else if (auto* f2 = std::get_if<FogExp2>(&*sceneObj->fog)) {
                fogColor = f2->color;
                fogDensity = f2->density;
                fogBits = FEAT_FOG_EXP2;
            }
        }

        uint64_t tonemapBits = TONEMAP_NONE;
        switch (scope.toneMapping) {
            case ToneMapping::Linear: tonemapBits = TONEMAP_LINEAR; break;
            case ToneMapping::Reinhard: tonemapBits = TONEMAP_REINHARD; break;
            case ToneMapping::Cineon: tonemapBits = TONEMAP_CINEON; break;
            case ToneMapping::ACESFilmic: tonemapBits = TONEMAP_ACES; break;
            default: break;
        }

        // Render opaque objects (front-to-back, depth write on)
        for (auto* item : renderList_.opaque) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera,
                       fogBits, fogColor, fogNear, fogFar, fogDensity,
                       tonemapBits);
        }

        // Render transparent objects (back-to-front, depth write off)
        for (auto* item : renderList_.transparent) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera,
                       fogBits, fogColor, fogNear, fogFar, fogDensity,
                       tonemapBits);
        }

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // Submit
        WGPUCommandBufferDescriptor cmdDesc{};
        cmdDesc.label = {.data = "cmd_buf", .length = 7};
        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmdBuffer);

        if (useSurface) {
            wgpuSurfacePresent(surface);
        }

        // Cleanup per-frame resources
        wgpuCommandBufferRelease(cmdBuffer);
        wgpuCommandEncoderRelease(encoder);
        if (useSurface) {
            wgpuTextureViewRelease(depthView);
            wgpuTextureRelease(frameDepthTexture);
            if (resolveView) {
                // MSAA: release both the MSAA color texture and the resolve view
                wgpuTextureViewRelease(colorView);       // MSAA view
                wgpuTextureRelease(frameMsaaColorTexture);
                wgpuTextureViewRelease(resolveView);     // surface resolve view
            } else {
                wgpuTextureViewRelease(colorView);
            }
            wgpuTextureRelease(surfaceTexture.texture);
        }
    }

    void renderCustomShaderObject(WGPURenderPassEncoder pass, Mesh* mesh,
                                   ShaderMaterial* sm,
                                   const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                                   const Camera& camera) {
        auto geometry = mesh->geometry();
        if (!geometry || !geometry->hasAttribute("position")) return;

        // Skip GLSL shaders — Dawn requires WGSL
        if (sm->vertexShader.find("gl_Position") != std::string::npos ||
            sm->fragmentShader.find("gl_FragColor") != std::string::npos) {
            return;
        }

        // Combine vertex + fragment shader into one module
        std::string wgsl = sm->vertexShader + "\n" + sm->fragmentShader;

        // Hash the shader source for cache invalidation
        size_t shaderHash = std::hash<std::string>{}(wgsl);

        auto it = customPipelineCache.find(sm);
        bool needRebuild = (it == customPipelineCache.end() || it->second.shaderHash != shaderHash);

        if (needRebuild) {
            // Clean up old entry
            if (it != customPipelineCache.end()) {
                auto& old = it->second;
                if (old.pipeline) wgpuRenderPipelineRelease(old.pipeline);
                if (old.layout) wgpuPipelineLayoutRelease(old.layout);
                if (old.bindGroupLayout) wgpuBindGroupLayoutRelease(old.bindGroupLayout);
                if (old.shader) wgpuShaderModuleRelease(old.shader);
            }

            CustomPipelineEntry entry{};
            entry.shaderHash = shaderHash;

            // Create shader module
            WGPUShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgslSrc.chain.next = nullptr;
            wgslSrc.code = {.data = wgsl.c_str(), .length = wgsl.size()};

            WGPUShaderModuleDescriptor shaderDesc{};
            shaderDesc.nextInChain = &wgslSrc.chain;
            shaderDesc.label = {.data = "custom_shader", .length = 13};
            entry.shader = wgpuDeviceCreateShaderModule(device, &shaderDesc);

            // Build bind group layout entries
            // Binding 0: TransformUniforms (256 bytes) — vertex + fragment
            // Binding 1: LightData (704 bytes) — fragment
            // Bindings 2+: user-defined (discovered from customTextures)
            std::vector<WGPUBindGroupLayoutEntry> bglEntries;

            { WGPUBindGroupLayoutEntry e{}; e.binding = 0;
              e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
              e.buffer.type = WGPUBufferBindingType_Uniform;
              e.buffer.minBindingSize = TRANSFORM_UNIFORM_SIZE;
              bglEntries.push_back(e); }

            { WGPUBindGroupLayoutEntry e{}; e.binding = 1;
              e.visibility = WGPUShaderStage_Fragment;
              e.buffer.type = WGPUBufferBindingType_Uniform;
              e.buffer.minBindingSize = LIGHT_UNIFORM_SIZE;
              bglEntries.push_back(e); }

            // Custom uniform buffer at binding 2 (for ocean params etc.)
            bool hasCustomUniforms = false;
            for (auto& [name, uniform] : sm->uniforms) {
                if (uniform.hasValue()) {
                    hasCustomUniforms = true;
                    break;
                }
            }
            if (hasCustomUniforms) {
                WGPUBindGroupLayoutEntry e{}; e.binding = 2;
                e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = 256;
                bglEntries.push_back(e);
            }

            // GPU texture bindings (texture + sampler pairs)
            std::vector<std::string> texNames;
            for (auto& [name, ptr] : sm->customTextures) {
                texNames.push_back(name);
            }
            std::sort(texNames.begin(), texNames.end());

            uint32_t nextBinding = hasCustomUniforms ? 3 : 2;
            for (auto& name : texNames) {
                { WGPUBindGroupLayoutEntry e{}; e.binding = nextBinding++;
                  e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                  e.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                  e.texture.viewDimension = WGPUTextureViewDimension_2D;
                  bglEntries.push_back(e); }
                { WGPUBindGroupLayoutEntry e{}; e.binding = nextBinding++;
                  e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                  e.sampler.type = WGPUSamplerBindingType_NonFiltering;
                  bglEntries.push_back(e); }
            }

            entry.bglEntries = bglEntries;

            WGPUBindGroupLayoutDescriptor bglDesc{};
            bglDesc.label = {.data = "custom_bgl", .length = 10};
            bglDesc.entryCount = bglEntries.size();
            bglDesc.entries = bglEntries.data();
            entry.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

            WGPUPipelineLayoutDescriptor plDesc{};
            plDesc.label = {.data = "custom_pl", .length = 9};
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = &entry.bindGroupLayout;
            entry.layout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

            // Vertex buffer layout: pos(vec3) + normal(vec3) + uv(vec2) = 32 bytes
            WGPUVertexAttribute attrs[3]{};
            attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0; attrs[0].shaderLocation = 0;
            attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
            attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;

            WGPUVertexBufferLayout vbLayout{};
            vbLayout.arrayStride = dawn::VERTEX_STRIDE;
            vbLayout.stepMode = WGPUVertexStepMode_Vertex;
            vbLayout.attributeCount = 3;
            vbLayout.attributes = attrs;

            WGPUBlendState blendState{};
            blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
            blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blendState.color.operation = WGPUBlendOperation_Add;
            blendState.alpha.srcFactor = WGPUBlendFactor_One;
            blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blendState.alpha.operation = WGPUBlendOperation_Add;

            WGPUColorTargetState colorTarget{};
            colorTarget.format = surfaceFormat;
            colorTarget.writeMask = WGPUColorWriteMask_All;
            colorTarget.blend = &blendState;

            WGPUStringView fsEntry = {.data = "fs_main", .length = 7};
            WGPUFragmentState fragmentState{};
            fragmentState.module = entry.shader;
            fragmentState.entryPoint = fsEntry;
            fragmentState.targetCount = 1;
            fragmentState.targets = &colorTarget;

            WGPUDepthStencilState depthStencil{};
            depthStencil.format = WGPUTextureFormat_Depth24Plus;
            depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
            depthStencil.depthCompare = WGPUCompareFunction_Less;

            WGPURenderPipelineDescriptor pipelineDesc{};
            pipelineDesc.label = {.data = "custom_pipeline", .length = 15};
            pipelineDesc.layout = entry.layout;

            WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
            pipelineDesc.vertex.module = entry.shader;
            pipelineDesc.vertex.entryPoint = vsEntry;
            pipelineDesc.vertex.bufferCount = 1;
            pipelineDesc.vertex.buffers = &vbLayout;
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
            pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
            pipelineDesc.primitive.cullMode = WGPUCullMode_None;
            pipelineDesc.depthStencil = &depthStencil;
            pipelineDesc.multisample.count = sampleCount_;
            pipelineDesc.multisample.mask = 0xFFFFFFFF;
            pipelineDesc.fragment = &fragmentState;

            entry.pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);

            customPipelineCache[sm] = entry;
        }

        auto& pe = customPipelineCache[sm];

        // Create per-draw transform buffer (each draw needs its own buffer because
        // wgpuQueueWriteBuffer writes are batched before the render pass executes,
        // so a shared buffer would only contain the last write's data).
        float transformData[TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, mesh->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix4 modelView;
        modelView.multiplyMatrices(viewMatrix, *mesh->matrixWorld);
        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelView);
        normalMatrix.invert();
        normalMatrix.transpose();
        auto& ne = normalMatrix.elements;
        transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
        transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
        transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        transformData[60] = camPos.x;
        transformData[61] = camPos.y;
        transformData[62] = camPos.z;
        transformData[63] = 0;

        WGPUBufferDescriptor xfBufDesc{};
        xfBufDesc.label = {.data = "custom_xform", .length = 12};
        xfBufDesc.size = TRANSFORM_UNIFORM_SIZE;
        xfBufDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransformBuf = wgpuDeviceCreateBuffer(device, &xfBufDesc);
        wgpuQueueWriteBuffer(queue, perDrawTransformBuf, 0, transformData, TRANSFORM_UNIFORM_SIZE);

        // Build bind group entries
        std::vector<WGPUBindGroupEntry> entries;
        { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = perDrawTransformBuf; e.offset = 0; e.size = TRANSFORM_UNIFORM_SIZE; entries.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = lightBuffer; e.offset = 0; e.size = LIGHT_UNIFORM_SIZE; entries.push_back(e); }

        // Custom uniforms at binding 2
        WGPUBuffer customUniformBuf = nullptr;
        bool hasCustomUniforms = false;
        for (auto& [name, uniform] : sm->uniforms) {
            if (uniform.hasValue()) { hasCustomUniforms = true; break; }
        }
        if (hasCustomUniforms) {
            float uboData[64];
            std::memset(uboData, 0, sizeof(uboData));
            int idx = 0;
            for (auto& [name, uniform] : sm->uniforms) {
                if (!uniform.hasValue()) continue;
                auto& val = uniform.value();
                if (auto* f = std::get_if<float>(&val)) {
                    if (idx < 64) uboData[idx++] = *f;
                } else if (auto* i = std::get_if<int>(&val)) {
                    if (idx < 64) { float fi; std::memcpy(&fi, i, 4); uboData[idx++] = fi; }
                } else if (auto* v3 = std::get_if<Vector3>(&val)) {
                    if (idx + 3 <= 64) { uboData[idx] = v3->x; uboData[idx+1] = v3->y; uboData[idx+2] = v3->z; idx += 4; }
                }
            }
            WGPUBufferDescriptor bd{};
            bd.label = {.data = "custom_ubo", .length = 10};
            bd.size = 256;
            bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            customUniformBuf = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, customUniformBuf, 0, uboData, 256);

            WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = customUniformBuf; e.offset = 0; e.size = 256;
            entries.push_back(e);
        }

        // GPU texture bindings
        std::vector<std::string> texNames;
        for (auto& [name, ptr] : sm->customTextures) {
            texNames.push_back(name);
        }
        std::sort(texNames.begin(), texNames.end());

        uint32_t nextBinding = hasCustomUniforms ? 3 : 2;
        for (auto& name : texNames) {
            auto* gpuTex = static_cast<GPUTexture*>(sm->customTextures[name]);
            { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.textureView = gpuTex->view(); entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.sampler = gpuTex->sampler(); entries.push_back(e); }
        }

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = {.data = "custom_bg", .length = 9};
        bgDesc.layout = pe.bindGroupLayout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry.get());
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                  gb.vertexCount * dawn::VERTEX_STRIDE);
            if (gb.indexBuffer) {
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                     WGPUIndexFormat_Uint32, 0,
                                                     gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                renderInfo.calls++;
                renderInfo.triangles += gb.indexCount / 3;
            } else {
                wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                renderInfo.calls++;
                renderInfo.triangles += gb.vertexCount / 3;
            }
        }

        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(perDrawTransformBuf);
        if (customUniformBuf) wgpuBufferRelease(customUniformBuf);
    }

    // Collect all renderable objects into the render list with z-depth for sorting.
    // Mirrors GLRenderer's projectObject with frustum culling, LOD, Sprite, Line, Points support.
    void collectRenderables(Object3D& object, const Matrix4& projScreenMatrix,
                            Camera& camera, unsigned int groupOrder) {
        if (!object.visible) return;

        if (object.is<Group>()) {
            groupOrder = object.renderOrder;
        } else if (auto lod = object.as<LOD>()) {
            if (lod->autoUpdate) lod->update(camera);
        } else if (auto sprite = object.as<Sprite>()) {
            if (!object.frustumCulled || frustum_.intersectsSprite(*sprite)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*sprite->matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }
                auto material = sprite->material().get();
                if (material && material->visible) {
                    renderList_.push(sprite, nullptr, material, groupOrder, _vector3.z, std::nullopt);
                }
            }
        } else if (object.is<Mesh>() || object.is<Line>() || object.is<Points>()) {
            if (!object.frustumCulled || frustum_.intersectsObject(object)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*object.matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }

                auto* owm = object.as<ObjectWithMaterials>();
                if (owm) {
                    auto geometry = owm->geometry();
                    if (geometry && geometry->hasAttribute("position")) {
                        const auto& materials = owm->materials();
                        if (materials.size() > 1) {
                            const auto& groups = geometry->groups;
                            for (const auto& group : groups) {
                                auto groupMat = materials.at(group.materialIndex).get();
                                if (groupMat && groupMat->visible) {
                                    renderList_.push(&object, geometry.get(), groupMat,
                                                     groupOrder, _vector3.z, group);
                                }
                            }
                        } else if (!materials.empty() && materials.front()->visible) {
                            renderList_.push(&object, geometry.get(), materials.front().get(),
                                             groupOrder, _vector3.z, std::nullopt);
                        }
                    }
                }
            }
        }

        children:
        for (auto& child : object.children) {
            collectRenderables(*child, projScreenMatrix, camera, groupOrder);
        }
    }

    // Render a single item from the render list.
    void renderItem(WGPURenderPassEncoder pass, const RenderItem* item,
                    const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                    const Camera& camera,
                    uint64_t fogBits, const Color& fogColor,
                    float fogNear, float fogFar, float fogDensity,
                    uint64_t tonemapBits) {

        auto* object = item->object;
        auto* geometry = item->geometry;
        Material* rawMat = item->material;
        if (!object || !rawMat) return;

        // Determine object type
        bool isMesh = object->is<Mesh>();
        bool isLine = object->is<Line>();
        bool isPoints = object->is<Points>();
        bool isLineSegments = object->is<LineSegments>();
        auto* instancedMesh = object->as<InstancedMesh>();
        auto* skinnedMesh = object->as<SkinnedMesh>();

        // Geometry comes from the render item (set during collection)
        // For sprites without geometry, skip for now
        if (!geometry) return;

        // ShaderMaterial: use the custom rendering path with user-provided WGSL shaders
        if (auto* sm = dynamic_cast<ShaderMaterial*>(rawMat)) {
            if (auto* mesh = object->as<Mesh>()) {
                renderCustomShaderObject(pass, mesh, sm, projectionMatrix, viewMatrix, camera);
                return;
            }
        }

        // Determine features and extract material parameters
        uint64_t features = FEAT_NONE;
        Color diffuse(1, 1, 1);
        float opacity = rawMat->opacity;
        Color specularColor(0, 0, 0);
        float shininess = 30.0f;
        float roughness = 0.5f, metalness = 0.0f;
        Color emissive(0, 0, 0);
        Texture* diffuseMap = nullptr;
        Texture* normalMap = nullptr;
        Texture* emissiveMap = nullptr;
        Texture* roughnessMap = nullptr;
        Texture* metalnessMap = nullptr;
        Texture* aoMap = nullptr;
        Texture* alphaMap = nullptr;
        Texture* specularMap = nullptr;
        Texture* lightMap = nullptr;
        Texture* bumpMap = nullptr;
        Texture* gradientMap = nullptr;
        Texture* displacementMap = nullptr;
        Texture* envMap = nullptr;
        float envMapIntensity = 1.0f;
        float displacementScale = 1.0f;
        Vector2 normalScale(1, 1);
        float aoMapIntensity = 1.0f;
        float bumpScale = 1.0f;

        if (auto m = dynamic_cast<MeshStandardMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING | FEAT_PBR;
            diffuse = m->color; roughness = m->roughness; metalness = m->metalness;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
            if (m->normalMap) {
                normalMap = m->normalMap.get();
                normalScale = m->normalScale;
                features |= FEAT_NORMAL_MAP;
            }
            if (m->emissiveMap) { emissiveMap = m->emissiveMap.get(); features |= FEAT_EMISSIVE_MAP; }
            if (m->roughnessMap) { roughnessMap = m->roughnessMap.get(); features |= FEAT_ROUGHNESS_MAP; }
            if (m->metalnessMap) { metalnessMap = m->metalnessMap.get(); features |= FEAT_METALNESS_MAP; }
            if (m->aoMap) { aoMap = m->aoMap.get(); aoMapIntensity = m->aoMapIntensity; features |= FEAT_AO_MAP; }
            if (m->alphaMap) { alphaMap = m->alphaMap.get(); features |= FEAT_ALPHA_MAP; }
            if (m->lightMap) { lightMap = m->lightMap.get(); features |= FEAT_LIGHT_MAP; }
            if (m->bumpMap) { bumpMap = m->bumpMap.get(); bumpScale = m->bumpScale; features |= FEAT_BUMP_MAP; }
            if (m->displacementMap) { displacementMap = m->displacementMap.get(); displacementScale = m->displacementScale; features |= FEAT_DISPLACEMENT_MAP; }
            if (m->envMap) { envMap = m->envMap.get(); envMapIntensity = m->envMapIntensity; features |= FEAT_ENV_MAP; }
        } else if (auto m = dynamic_cast<MeshPhongMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING | FEAT_SPECULAR;
            diffuse = m->color; specularColor = m->specular; shininess = m->shininess;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
            if (m->normalMap) { normalMap = m->normalMap.get(); normalScale = m->normalScale; features |= FEAT_NORMAL_MAP; }
            if (m->emissiveMap) { emissiveMap = m->emissiveMap.get(); features |= FEAT_EMISSIVE_MAP; }
            if (m->aoMap) { aoMap = m->aoMap.get(); aoMapIntensity = m->aoMapIntensity; features |= FEAT_AO_MAP; }
            if (m->alphaMap) { alphaMap = m->alphaMap.get(); features |= FEAT_ALPHA_MAP; }
            if (m->specularMap) { specularMap = m->specularMap.get(); features |= FEAT_SPECULAR_MAP; }
            if (m->lightMap) { lightMap = m->lightMap.get(); features |= FEAT_LIGHT_MAP; }
            if (m->bumpMap) { bumpMap = m->bumpMap.get(); bumpScale = m->bumpScale; features |= FEAT_BUMP_MAP; }
        } else if (auto m = dynamic_cast<MeshToonMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING;
            diffuse = m->color;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
            if (m->emissiveMap) { emissiveMap = m->emissiveMap.get(); features |= FEAT_EMISSIVE_MAP; }
            if (m->normalMap) { normalMap = m->normalMap.get(); normalScale = m->normalScale; features |= FEAT_NORMAL_MAP; }
            if (m->bumpMap) { bumpMap = m->bumpMap.get(); bumpScale = m->bumpScale; features |= FEAT_BUMP_MAP; }
            if (m->gradientMap) { gradientMap = m->gradientMap.get(); features |= FEAT_GRADIENT_MAP; }
        } else if (auto m = dynamic_cast<MeshLambertMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING;
            diffuse = m->color;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto m = dynamic_cast<MeshBasicMaterial*>(rawMat)) {
            diffuse = m->color;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto m = dynamic_cast<LineBasicMaterial*>(rawMat)) {
            diffuse = m->color;
        } else if (auto m = dynamic_cast<PointsMaterial*>(rawMat)) {
            diffuse = m->color;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (dynamic_cast<ShadowMaterial*>(rawMat)) {
            // ShadowMaterial: without proper shadow support on this material,
            // skip rendering entirely. In Three.js, ShadowMaterial's alpha comes
            // from the shadow factor; without it, the material should be invisible.
            return;
        } else if (auto m = dynamic_cast<ShaderMaterial*>(rawMat)) {
            // Basic ShaderMaterial support: extract color from uniforms if available
            if (m->uniforms.count("uColor") && m->uniforms.at("uColor").hasValue()) {
                try {
                    auto& val = const_cast<Uniform&>(m->uniforms.at("uColor")).value();
                    if (auto* c = std::get_if<Color>(&val)) {
                        diffuse = *c;
                    }
                } catch (...) {}
            } else if (m->uniforms.count("color") && m->uniforms.at("color").hasValue()) {
                try {
                    auto& val = const_cast<Uniform&>(m->uniforms.at("color")).value();
                    if (auto* c = std::get_if<Color>(&val)) {
                        diffuse = *c;
                    }
                } catch (...) {}
            }
        } else if (auto cm = dynamic_cast<MaterialWithColor*>(rawMat)) {
            diffuse = cm->color;
        }

        // Set topology based on object type
        bool isLineLoop = object->is<LineLoop>();
        if (isLine) {
            if (object->is<LineSegments>() || isLineLoop) {
                // LineLoop uses LineList with a generated index buffer that closes the loop
                features |= TOPO_LINE_LIST;
            } else {
                features |= TOPO_LINE_STRIP;
            }
        } else if (isPoints) {
            features |= TOPO_POINT_LIST;
        }

        // Face culling based on material.side
        switch (rawMat->side) {
            case Side::Front: features |= CULL_BACK; break;
            case Side::Back:  features |= CULL_FRONT; break;
            case Side::Double: features |= CULL_NONE; break;
        }

        // Wireframe mode (only for mesh objects)
        bool useWireframe = false;
        if (isMesh) {
            if (auto wf = dynamic_cast<MaterialWithWireframe*>(rawMat)) {
                if (wf->wireframe) {
                    features |= WIREFRAME_BIT;
                    useWireframe = true;
                }
            }
        }

        // Blend mode
        auto blendVal = static_cast<int>(rawMat->blending);
        if (blendVal == 0)              features |= BLEND_DISABLED;
        else if (blendVal == 2)         features |= BLEND_ADDITIVE;
        else if (blendVal == 3)         features |= BLEND_SUBTRACTIVE;
        else if (blendVal == 4)         features |= BLEND_MULTIPLY;
        else                            features |= BLEND_NORMAL;
        if (rawMat->transparent) {
            features |= DEPTH_WRITE_OFF;
        }

        // Vertex colors
        if (rawMat->vertexColors && geometry->hasAttribute("color")) {
            features |= FEAT_VERTEX_COLORS;
        }

        // Instancing
        if (instancedMesh) {
            features |= FEAT_INSTANCED;
            if (instancedMesh->instanceColor()) {
                features |= FEAT_INSTANCE_COLOR;
            }
        }

        // Skinning
        if (skinnedMesh && skinnedMesh->skeleton &&
            geometry->hasAttribute("skinIndex") && geometry->hasAttribute("skinWeight")) {
            features |= FEAT_SKINNING;
        }

        // Morph targets
        auto* morphMat = dynamic_cast<MaterialWithMorphTargets*>(rawMat);
        if (morphMat && morphMat->morphTargets && geometry->getMorphAttributes().count("position") > 0) {
            auto mesh = object->as<Mesh>();
            if (mesh && !mesh->morphTargetInfluences().empty()) {
                features |= FEAT_MORPH_TARGETS;
            }
        }

        // Shadow (mesh objects with lighting only — shadow code is inside the lighting block)
        if (isMesh && shadowState.active && object->receiveShadow &&
            (features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR))) {
            features |= FEAT_SHADOW;
        }

        // Fog, tone mapping, and output encoding
        if (rawMat->fog) {
            features |= fogBits;
        }
        features |= tonemapBits;
        if (scope.outputEncoding == Encoding::sRGB) {
            features |= FEAT_SRGB_OUTPUT;
        }

        // Get/create pipeline for this feature set
        auto& pe = getOrCreatePipeline(features);
        if (!pe.pipeline) return;

        // Upload transform uniforms
        float transformData[TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, object->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix4 modelView;
        modelView.multiplyMatrices(viewMatrix, *object->matrixWorld);
        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelView);
        normalMatrix.invert();
        normalMatrix.transpose();
        auto& ne = normalMatrix.elements;
        transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
        transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
        transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        transformData[60] = camPos.x;
        transformData[61] = camPos.y;
        transformData[62] = camPos.z;
        transformData[63] = 0;

        // Create per-draw uniform buffers (wgpuQueueWriteBuffer writes are batched
        // before the render pass executes, so a shared buffer across draws would only
        // contain the last write's data).
        WGPUBufferDescriptor xfDesc{};
        xfDesc.label = {.data = "xform_buf", .length = 9};
        xfDesc.size = TRANSFORM_UNIFORM_SIZE;
        xfDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransform = wgpuDeviceCreateBuffer(device, &xfDesc);
        wgpuQueueWriteBuffer(queue, perDrawTransform, 0, transformData, TRANSFORM_UNIFORM_SIZE);

        // Upload material uniforms (now includes fog and tone mapping data)
        float matData[MATERIAL_UNIFORM_SIZE / sizeof(float)];
        std::memset(matData, 0, sizeof(matData));
        matData[0] = diffuse.r; matData[1] = diffuse.g; matData[2] = diffuse.b; matData[3] = 1.0f;
        matData[4] = specularColor.r; matData[5] = specularColor.g; matData[6] = specularColor.b; matData[7] = shininess;
        matData[8] = roughness; matData[9] = metalness; matData[10] = opacity; matData[11] = displacementScale;
        matData[12] = emissive.r; matData[13] = emissive.g; matData[14] = emissive.b; matData[15] = envMapIntensity;
        matData[16] = aoMapIntensity;  // flags.x: aoMapIntensity
        matData[17] = bumpScale;       // flags.y: bumpScale
        matData[18] = normalScale.x;
        matData[19] = normalScale.y;
        // fogColor (vec4, offset 20)
        matData[20] = fogColor.r; matData[21] = fogColor.g; matData[22] = fogColor.b; matData[23] = 1.0f;
        // fogParams: x=near, y=far, z=density, w=toneMappingExposure (vec4, offset 24)
        matData[24] = fogNear; matData[25] = fogFar; matData[26] = fogDensity;
        matData[27] = scope.toneMappingExposure;
        // clipPlane (vec4, offset 28): normal(xyz) + constant(w)
        if (scope.localClippingEnabled && !rawMat->clippingPlanes.empty()) {
            auto& cp = rawMat->clippingPlanes[0];
            matData[28] = cp.normal.x;
            matData[29] = cp.normal.y;
            matData[30] = cp.normal.z;
            matData[31] = cp.constant;
        }

        WGPUBufferDescriptor matDesc{};
        matDesc.label = {.data = "mat_buf", .length = 7};
        matDesc.size = MATERIAL_UNIFORM_SIZE;
        matDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawMaterial = wgpuDeviceCreateBuffer(device, &matDesc);
        wgpuQueueWriteBuffer(queue, perDrawMaterial, 0, matData, MATERIAL_UNIFORM_SIZE);

        // Build bind group dynamically
        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);
        bool tex = features & FEAT_TEXTURE;

        std::vector<WGPUBindGroupEntry> entries;
        { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = perDrawTransform; e.offset = 0; e.size = TRANSFORM_UNIFORM_SIZE; entries.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = perDrawMaterial; e.offset = 0; e.size = MATERIAL_UNIFORM_SIZE; entries.push_back(e); }

        if (lit) {
            WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = lightBuffer; e.offset = 0; e.size = LIGHT_UNIFORM_SIZE; entries.push_back(e);
        }

        auto* texEntry = &textures->getDummyTexture();
        if (tex && diffuseMap) {
            texEntry = &textures->getOrCreateTexture(diffuseMap);
        }
        if (tex) {
            { WGPUBindGroupEntry e{}; e.binding = 3; e.textureView = texEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 4; e.sampler = texEntry->sampler; entries.push_back(e); }
        }

        if (features & FEAT_NORMAL_MAP) {
            auto* nmEntry = &textures->getDummyTexture();
            if (normalMap) {
                nmEntry = &textures->getOrCreateTexture(normalMap);
            }
            { WGPUBindGroupEntry e{}; e.binding = 5; e.textureView = nmEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 6; e.sampler = nmEntry->sampler; entries.push_back(e); }
        }

        if (features & FEAT_SHADOW) {
            { WGPUBindGroupEntry e{}; e.binding = 7; e.buffer = shadowState.uniformBuffer; e.offset = 0; e.size = SHADOW_UNIFORM_SIZE; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 8; e.textureView = shadowState.depthArrayView; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 9; e.sampler = shadowState.comparisonSampler; entries.push_back(e); }
        }

        // Helper lambda to add texture bind group entries
        auto addTexEntries = [&](uint32_t texBinding, uint32_t sampBinding, Texture* tex) {
            auto* te = tex ? &textures->getOrCreateTexture(tex) : &textures->getDummyTexture();
            { WGPUBindGroupEntry e{}; e.binding = texBinding; e.textureView = te->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = sampBinding; e.sampler = te->sampler; entries.push_back(e); }
        };

        if (features & FEAT_EMISSIVE_MAP)  addTexEntries(10, 11, emissiveMap);
        if (features & FEAT_ROUGHNESS_MAP) addTexEntries(12, 13, roughnessMap);
        if (features & FEAT_METALNESS_MAP) addTexEntries(14, 15, metalnessMap);
        if (features & FEAT_AO_MAP)        addTexEntries(16, 17, aoMap);
        if (features & FEAT_ALPHA_MAP)     addTexEntries(18, 19, alphaMap);
        if (features & FEAT_SPECULAR_MAP)  addTexEntries(20, 21, specularMap);
        if (features & FEAT_LIGHT_MAP)     addTexEntries(22, 23, lightMap);
        if (features & FEAT_BUMP_MAP)      addTexEntries(24, 25, bumpMap);
        if (features & FEAT_GRADIENT_MAP)  addTexEntries(26, 27, gradientMap);
        if (features & FEAT_DISPLACEMENT_MAP) addTexEntries(30, 31, displacementMap);

        if (features & FEAT_ENV_MAP) {
            auto* te = envMap ? &textures->getOrCreateCubeTexture(envMap) : &textures->getDummyCubeTexture();
            { WGPUBindGroupEntry e{}; e.binding = 32; e.textureView = te->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 33; e.sampler = te->sampler; entries.push_back(e); }
        }

        // Instance data buffer
        WGPUBuffer instanceBuffer = nullptr;
        if ((features & FEAT_INSTANCED) && instancedMesh) {
            bool hasColor = features & FEAT_INSTANCE_COLOR;
            size_t instanceCount = instancedMesh->count();
            // Each instance: mat4x4 (16 floats) + optional vec4 color (4 floats)
            size_t floatsPerInstance = hasColor ? 20 : 16;
            size_t bufSize = instanceCount * floatsPerInstance * sizeof(float);
            std::vector<float> instanceData(instanceCount * floatsPerInstance, 0.0f);

            auto* matAttr = instancedMesh->instanceMatrix();
            auto* colAttr = instancedMesh->instanceColor();
            for (size_t i = 0; i < instanceCount; i++) {
                // Copy 4x4 matrix (already column-major, matching WGSL mat4x4)
                const float* src = &matAttr->array()[i * 16];
                float* dst = &instanceData[i * floatsPerInstance];
                std::memcpy(dst, src, 16 * sizeof(float));
                if (hasColor && colAttr) {
                    const float* csrc = &colAttr->array()[i * 3];
                    dst[16] = csrc[0];
                    dst[17] = csrc[1];
                    dst[18] = csrc[2];
                    dst[19] = 1.0f;
                }
            }

            WGPUBufferDescriptor bd{};
            bd.label = {.data = "instance_data", .length = 13};
            bd.size = bufSize;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            instanceBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, instanceBuffer, 0, instanceData.data(), bufSize);

            WGPUBindGroupEntry e{};
            e.binding = 28;
            e.buffer = instanceBuffer;
            e.offset = 0;
            e.size = bufSize;
            entries.push_back(e);
        }

        // Morph target data buffer
        WGPUBuffer morphBuffer = nullptr;
        if (features & FEAT_MORPH_TARGETS) {
            auto mesh = object->as<Mesh>();
            auto& morphAttrs = geometry->getMorphAttributes().at("position");
            uint32_t numTargets = static_cast<uint32_t>(morphAttrs.size());
            uint32_t vertexCount = static_cast<uint32_t>(geometry->getAttribute<float>("position")->count());
            auto& influences = mesh->morphTargetInfluences();

            // Layout: header (4 u32 = numTargets + 3 padding) +
            //         influences (2 vec4 = 8 floats) +
            //         positions (numTargets * vertexCount vec4s)
            size_t headerSize = 4;  // 4 u32s = 1 vec4
            size_t influenceSize = 8; // 2 vec4s = 8 floats
            size_t posSize = numTargets * vertexCount * 4; // vec4 per vertex per target
            size_t totalFloats = headerSize + influenceSize + posSize;
            std::vector<float> morphData(totalFloats, 0.0f);

            auto* u32Data = reinterpret_cast<uint32_t*>(morphData.data());
            u32Data[0] = numTargets;

            // Pack influences
            for (uint32_t t = 0; t < numTargets && t < 8; t++) {
                if (t < influences.size()) {
                    morphData[headerSize + t] = influences[t];
                }
            }

            // Pack morph target positions as vec4s
            size_t posOffset = headerSize + influenceSize;
            for (uint32_t t = 0; t < numTargets; t++) {
                auto* attr = dynamic_cast<TypedBufferAttribute<float>*>(morphAttrs[t].get());
                if (!attr) continue;
                for (uint32_t v = 0; v < vertexCount; v++) {
                    size_t idx = posOffset + (t * vertexCount + v) * 4;
                    morphData[idx + 0] = attr->getX(v);
                    morphData[idx + 1] = attr->getY(v);
                    morphData[idx + 2] = attr->getZ(v);
                    morphData[idx + 3] = 0.0f;
                }
            }

            size_t bufSize = totalFloats * sizeof(float);
            WGPUBufferDescriptor bd{};
            bd.label = {.data = "morph_data", .length = 10};
            bd.size = bufSize;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            morphBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, morphBuffer, 0, morphData.data(), bufSize);

            WGPUBindGroupEntry e{};
            e.binding = 29;
            e.buffer = morphBuffer;
            e.offset = 0;
            e.size = bufSize;
            entries.push_back(e);
        }

        // Skinning data buffers
        WGPUBuffer skinBuffer = nullptr;
        WGPUBuffer skinVertexBuffer = nullptr;
        if ((features & FEAT_SKINNING) && skinnedMesh && skinnedMesh->skeleton) {
            auto& skel = *skinnedMesh->skeleton;
            skel.update(); // compute bone matrices

            uint32_t boneCount = static_cast<uint32_t>(skel.bones.size());
            // Layout: bindMatrix(16) + bindMatrixInverse(16) + boneCount(1 u32) + pad(3 u32) + boneMatrices(boneCount * 16)
            size_t headerFloats = 16 + 16 + 4; // 2 matrices + 1 vec4 header
            size_t totalFloats = headerFloats + boneCount * 16;
            std::vector<float> skinData(totalFloats, 0.0f);

            std::memcpy(skinData.data(), skinnedMesh->bindMatrix.elements.data(), 64);
            std::memcpy(skinData.data() + 16, skinnedMesh->bindMatrixInverse.elements.data(), 64);
            auto* u32ptr = reinterpret_cast<uint32_t*>(skinData.data() + 32);
            u32ptr[0] = boneCount;
            // Copy bone matrices
            if (!skel.boneMatrices.empty()) {
                std::memcpy(skinData.data() + headerFloats, skel.boneMatrices.data(),
                            boneCount * 16 * sizeof(float));
            }

            size_t bufSize = totalFloats * sizeof(float);
            WGPUBufferDescriptor bd{};
            bd.label = {.data = "skin_data", .length = 9};
            bd.size = bufSize;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            skinBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, skinBuffer, 0, skinData.data(), bufSize);

            { WGPUBindGroupEntry e{}; e.binding = 34; e.buffer = skinBuffer; e.offset = 0; e.size = bufSize;
              entries.push_back(e); }

            // Pack per-vertex skinIndex + skinWeight
            uint32_t vertexCount = static_cast<uint32_t>(geometry->getAttribute<float>("position")->count());
            auto* skinIdxAttr = geometry->getAttribute<float>("skinIndex");
            auto* skinWgtAttr = geometry->getAttribute<float>("skinWeight");
            size_t vertFloats = vertexCount * 8; // vec4 index + vec4 weight per vertex
            std::vector<float> vertData(vertFloats, 0.0f);
            for (uint32_t v = 0; v < vertexCount; v++) {
                vertData[v * 8 + 0] = skinIdxAttr->getX(v);
                vertData[v * 8 + 1] = skinIdxAttr->getY(v);
                vertData[v * 8 + 2] = skinIdxAttr->getZ(v);
                vertData[v * 8 + 3] = skinIdxAttr->getW(v);
                vertData[v * 8 + 4] = skinWgtAttr->getX(v);
                vertData[v * 8 + 5] = skinWgtAttr->getY(v);
                vertData[v * 8 + 6] = skinWgtAttr->getZ(v);
                vertData[v * 8 + 7] = skinWgtAttr->getW(v);
            }
            size_t vertBufSize = vertFloats * sizeof(float);
            WGPUBufferDescriptor vbd{};
            vbd.label = {.data = "skin_verts", .length = 10};
            vbd.size = vertBufSize;
            vbd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            skinVertexBuffer = wgpuDeviceCreateBuffer(device, &vbd);
            wgpuQueueWriteBuffer(queue, skinVertexBuffer, 0, vertData.data(), vertBufSize);

            { WGPUBindGroupEntry e{}; e.binding = 35; e.buffer = skinVertexBuffer; e.offset = 0; e.size = vertBufSize;
              entries.push_back(e); }
        }

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = {.data = "obj_bg", .length = 6};
        bgDesc.layout = pe.bindGroupLayout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry);
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                     gb.vertexCount * dawn::VERTEX_STRIDE);

            uint32_t instanceCount = instancedMesh ? static_cast<uint32_t>(instancedMesh->count()) : 1;

            if (useWireframe) {
                auto& wb = geometries->getOrCreateWireframeBuffers(geometry);
                if (wb.indexBuffer) {
                    wgpuRenderPassEncoderSetIndexBuffer(pass, wb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         wb.indexCount * sizeof(uint32_t));
                    wgpuRenderPassEncoderDrawIndexed(pass, wb.indexCount, instanceCount, 0, 0, 0);
                    renderInfo.calls++;
                    renderInfo.lines += wb.indexCount / 2;
                }
            } else if (isLineLoop) {
                // WebGPU has no line loop primitive — generate line-list indices
                // that include the closing edge (last vertex → first vertex)
                uint32_t n = gb.vertexCount;
                std::vector<uint32_t> loopIndices;
                loopIndices.reserve(n * 2);
                for (uint32_t i = 0; i < n; i++) {
                    loopIndices.push_back(i);
                    loopIndices.push_back((i + 1) % n);
                }
                WGPUBufferDescriptor bd{};
                bd.label = {.data = "lineloop_idx", .length = 12};
                bd.size = loopIndices.size() * sizeof(uint32_t);
                bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                WGPUBuffer loopBuf = wgpuDeviceCreateBuffer(device, &bd);
                wgpuQueueWriteBuffer(queue, loopBuf, 0, loopIndices.data(), bd.size);
                wgpuRenderPassEncoderSetIndexBuffer(pass, loopBuf,
                                                     WGPUIndexFormat_Uint32, 0, bd.size);
                uint32_t drawCount = static_cast<uint32_t>(loopIndices.size());
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, 0, 0, 0);
                wgpuBufferRelease(loopBuf);
                renderInfo.calls++;
                renderInfo.lines += n;
            } else if (gb.indexBuffer) {
                // Determine draw range from geometry group if present
                uint32_t drawStart = 0;
                uint32_t drawCount = gb.indexCount;
                if (item->group.has_value()) {
                    drawStart = static_cast<uint32_t>(item->group->start);
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, drawStart, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            } else {
                uint32_t drawCount = gb.vertexCount;
                if (item->group.has_value()) {
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderDraw(pass, drawCount, instanceCount, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            }
        }

        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(perDrawTransform);
        wgpuBufferRelease(perDrawMaterial);
        if (instanceBuffer) {
            wgpuBufferRelease(instanceBuffer);
        }
        if (morphBuffer) {
            wgpuBufferRelease(morphBuffer);
        }
        if (skinBuffer) {
            wgpuBufferRelease(skinBuffer);
        }
        if (skinVertexBuffer) {
            wgpuBufferRelease(skinVertexBuffer);
        }
    }

    void dispose() {
        if (!initialized) return;

        // Release geometry and texture subsystems
        if (geometries) geometries->dispose();
        if (textures) textures->dispose();

        // Release render target cache
        for (auto& [id, rt] : rtCache) {
            releaseRTEntry(rt);
        }
        rtCache.clear();

        // Release pipeline cache
        for (auto& [feat, pe] : pipelineCache) {
            if (pe.pipeline) wgpuRenderPipelineRelease(pe.pipeline);
            if (pe.layout) wgpuPipelineLayoutRelease(pe.layout);
            if (pe.bindGroupLayout) wgpuBindGroupLayoutRelease(pe.bindGroupLayout);
            if (pe.shader) wgpuShaderModuleRelease(pe.shader);
        }
        pipelineCache.clear();

        // Release custom shader pipeline cache
        for (auto& [mat, pe] : customPipelineCache) {
            if (pe.pipeline) wgpuRenderPipelineRelease(pe.pipeline);
            if (pe.layout) wgpuPipelineLayoutRelease(pe.layout);
            if (pe.bindGroupLayout) wgpuBindGroupLayoutRelease(pe.bindGroupLayout);
            if (pe.shader) wgpuShaderModuleRelease(pe.shader);
        }
        customPipelineCache.clear();

        disposeShadowMap();

        if (lightBuffer) wgpuBufferRelease(lightBuffer);
        if (queue) wgpuQueueRelease(queue);
        if (device) wgpuDeviceRelease(device);
        if (adapter) wgpuAdapterRelease(adapter);
        if (surface) wgpuSurfaceRelease(surface);
        if (instance) wgpuInstanceRelease(instance);

        initialized = false;
    }

    ~Impl() {
        dispose();
    }
};


// --- DawnRenderer public API ---

DawnRenderer::DawnRenderer(Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(*this, canvas)) {}

void DawnRenderer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

WindowSize DawnRenderer::size() const {
    return pimpl_->size_;
}

void DawnRenderer::setSize(const std::pair<int, int>& size) {
    pimpl_->canvas.setSize(size);
    pimpl_->size_ = {size.first, size.second};
    setViewport(0, 0, size.first, size.second);
    if (pimpl_->initialized) {
        pimpl_->configureSurface();
    }
}

float DawnRenderer::getTargetPixelRatio() const {
    return pimpl_->pixelRatio_;
}

void DawnRenderer::setPixelRatio(float value) {
    pimpl_->pixelRatio_ = value;
    setSize({pimpl_->size_.width(), pimpl_->size_.height()});
}

void DawnRenderer::setViewport(const Vector4& v) {
    pimpl_->viewport_.x = v.x; pimpl_->viewport_.y = v.y;
    pimpl_->viewport_.w = v.z; pimpl_->viewport_.h = v.w;
}

void DawnRenderer::setViewport(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->viewport_.x = std::floor(x * pr);
    pimpl_->viewport_.y = std::floor(y * pr);
    pimpl_->viewport_.w = std::floor(width * pr);
    pimpl_->viewport_.h = std::floor(height * pr);
}

void DawnRenderer::setScissor(const Vector4& v) {
    pimpl_->scissor_.x = static_cast<uint32_t>(v.x);
    pimpl_->scissor_.y = static_cast<uint32_t>(v.y);
    pimpl_->scissor_.w = static_cast<uint32_t>(v.z);
    pimpl_->scissor_.h = static_cast<uint32_t>(v.w);
}

void DawnRenderer::setScissor(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->scissor_.x = static_cast<uint32_t>(std::floor(x * pr));
    pimpl_->scissor_.y = static_cast<uint32_t>(std::floor(y * pr));
    pimpl_->scissor_.w = static_cast<uint32_t>(std::floor(width * pr));
    pimpl_->scissor_.h = static_cast<uint32_t>(std::floor(height * pr));
}

void DawnRenderer::getViewport(Vector4& target) const {
    target.set(pimpl_->viewport_.x, pimpl_->viewport_.y, pimpl_->viewport_.w, pimpl_->viewport_.h);
}

void DawnRenderer::setScissorTest(bool boolean) {
    pimpl_->scissorTest_ = boolean;
}

bool DawnRenderer::getScissorTest() const {
    return pimpl_->scissorTest_;
}

void DawnRenderer::getScissor(Vector4& target) const {
    target.set(static_cast<float>(pimpl_->scissor_.x), static_cast<float>(pimpl_->scissor_.y),
               static_cast<float>(pimpl_->scissor_.w), static_cast<float>(pimpl_->scissor_.h));
}

void DawnRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::getClearColor(Color& target) const {
    target = pimpl_->clearColor_;
}

float DawnRenderer::getClearAlpha() const {
    return pimpl_->clearAlpha_;
}

void DawnRenderer::setClearAlpha(float alpha) {
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::clear(bool /*color*/, bool /*depth*/, bool /*stencil*/) {
    // Clearing happens at render pass begin via loadOp = Clear
}

void DawnRenderer::clearColor() {
    clear(true, false, false);
}

void DawnRenderer::clearDepth() {
    clear(false, true, false);
}

void DawnRenderer::clearStencil() {
    clear(false, false, true);
}

RenderTarget* DawnRenderer::getRenderTarget() {
    return pimpl_->currentRenderTarget_;
}

void DawnRenderer::setRenderTarget(RenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {
    pimpl_->currentRenderTarget_ = renderTarget;
    pimpl_->activeCubeFace_ = activeCubeFace;
    pimpl_->activeMipmapLevel_ = activeMipmapLevel;
}

int DawnRenderer::getActiveCubeFace() const {
    return pimpl_->activeCubeFace_;
}

int DawnRenderer::getActiveMipmapLevel() const {
    return pimpl_->activeMipmapLevel_;
}

const DawnInfo& DawnRenderer::info() const {
    static DawnInfo di;
    di.render.frame = pimpl_->renderInfo.frame;
    di.render.calls = pimpl_->renderInfo.calls;
    di.render.triangles = pimpl_->renderInfo.triangles;
    di.render.lines = pimpl_->renderInfo.lines;
    di.render.points = pimpl_->renderInfo.points;
    di.memory.geometries = pimpl_->renderInfo.geometries;
    di.memory.textures = pimpl_->renderInfo.textures;
    return di;
}

void* DawnRenderer::nativeDevice() const {
    return pimpl_->device;
}

void* DawnRenderer::nativeQueue() const {
    return pimpl_->queue;
}

void* DawnRenderer::nativeInstance() const {
    return pimpl_->instance;
}

void DawnRenderer::setSampleCount(uint32_t count) {
    if (count != 1 && count != 4) {
        std::cerr << "DawnRenderer::setSampleCount: unsupported count " << count
                  << " (must be 1 or 4)" << std::endl;
        return;
    }
    if (pimpl_->sampleCount_ == count) return;

    pimpl_->sampleCount_ = count;

    // Invalidate pipeline cache — pipelines encode the multisample count
    for (auto& [feat, pe] : pimpl_->pipelineCache) {
        if (pe.pipeline) wgpuRenderPipelineRelease(pe.pipeline);
        if (pe.layout) wgpuPipelineLayoutRelease(pe.layout);
        if (pe.bindGroupLayout) wgpuBindGroupLayoutRelease(pe.bindGroupLayout);
        if (pe.shader) wgpuShaderModuleRelease(pe.shader);
    }
    pimpl_->pipelineCache.clear();

    // Invalidate custom shader pipeline cache
    for (auto& [mat, pe] : pimpl_->customPipelineCache) {
        if (pe.pipeline) wgpuRenderPipelineRelease(pe.pipeline);
        if (pe.layout) wgpuPipelineLayoutRelease(pe.layout);
        if (pe.bindGroupLayout) wgpuBindGroupLayoutRelease(pe.bindGroupLayout);
        if (pe.shader) wgpuShaderModuleRelease(pe.shader);
    }
    pimpl_->customPipelineCache.clear();

    // Invalidate render target cache — textures have wrong sample count
    for (auto& [id, rt] : pimpl_->rtCache) {
        pimpl_->releaseRTEntry(rt);
    }
    pimpl_->rtCache.clear();
}

uint32_t DawnRenderer::getSampleCount() const {
    return pimpl_->sampleCount_;
}

void DawnRenderer::resetState() {
    // Dawn manages its own state; this is a no-op for API compatibility
}

std::vector<unsigned char> DawnRenderer::readRGBPixels() {
    if (!pimpl_->initialized || !pimpl_->currentRenderTarget_) return {};

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
    uint32_t w = rt.width;
    uint32_t h = rt.height;

    // Row alignment: WebGPU requires bytesPerRow to be a multiple of 256
    uint32_t bytesPerPixel = 4; // BGRA8
    uint32_t unpaddedBytesPerRow = w * bytesPerPixel;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;
    uint32_t bufferSize = paddedBytesPerRow * h;

    // Create staging buffer
    WGPUBufferDescriptor bd{};
    bd.label = {.data = "readback_buf", .length = 12};
    bd.size = bufferSize;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer stagingBuf = wgpuDeviceCreateBuffer(pimpl_->device, &bd);

    // Copy texture to buffer
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = {.data = "readback_enc", .length = 12};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pimpl_->device, &encDesc);

    WGPUTexelCopyTextureInfo src{};
    src.texture = rt.colorTexture;

    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = stagingBuf;
    dst.layout.bytesPerRow = paddedBytesPerRow;
    dst.layout.rowsPerImage = h;

    WGPUExtent3D extent = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = {.data = "readback_cmd", .length = 12};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(pimpl_->queue, 1, &cmd);

    // Map buffer synchronously
    struct MapData { bool done = false; WGPUMapAsyncStatus status; } mapData;

    WGPUBufferMapCallbackInfo mapCb{};
    mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
    mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
        auto* d = static_cast<MapData*>(ud1);
        d->status = status;
        d->done = true;
    };
    mapCb.userdata1 = &mapData;
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, bufferSize, mapCb);

    auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
    while (!mapData.done) {
        if (std::chrono::steady_clock::now() > deadline) {
            wgpuBufferRelease(stagingBuf);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            throw std::runtime_error("DawnRenderer: readRGBPixels buffer map timed out");
        }
        wgpuDevicePoll(pimpl_->device, true, nullptr);
    }

    std::vector<unsigned char> result;
    if (mapData.status == WGPUMapAsyncStatus_Success) {
        auto* mapped = static_cast<const unsigned char*>(wgpuBufferGetConstMappedRange(stagingBuf, 0, bufferSize));
        result.resize(w * h * 3);
        for (uint32_t row = 0; row < h; row++) {
            for (uint32_t col = 0; col < w; col++) {
                const auto* px = mapped + row * paddedBytesPerRow + col * 4;
                size_t outIdx = (row * w + col) * 3;
                // BGRA -> RGB
                result[outIdx + 0] = px[2];
                result[outIdx + 1] = px[1];
                result[outIdx + 2] = px[0];
            }
        }
        wgpuBufferUnmap(stagingBuf);
    }

    wgpuBufferRelease(stagingBuf);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return result;
}

void DawnRenderer::readPixels(const Vector2& position, const std::pair<int, int>& sz,
                              std::vector<unsigned char>& data) {
    auto allPixels = readRGBPixels();
    if (allPixels.empty()) return;

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
    int rtW = static_cast<int>(rt.width);
    int rtH = static_cast<int>(rt.height);
    int x0 = static_cast<int>(position.x);
    int y0 = static_cast<int>(position.y);
    int w = sz.first;
    int h = sz.second;

    data.resize(w * h * 3);
    for (int row = 0; row < h; row++) {
        int srcY = y0 + row;
        if (srcY < 0 || srcY >= rtH) continue;
        for (int col = 0; col < w; col++) {
            int srcX = x0 + col;
            if (srcX < 0 || srcX >= rtW) continue;
            size_t srcIdx = (srcY * rtW + srcX) * 3;
            size_t dstIdx = (row * w + col) * 3;
            data[dstIdx + 0] = allPixels[srcIdx + 0];
            data[dstIdx + 1] = allPixels[srcIdx + 1];
            data[dstIdx + 2] = allPixels[srcIdx + 2];
        }
    }
}

void DawnRenderer::writeFramebuffer(const std::filesystem::path& filename) {
    auto ext = filename.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
        throw std::runtime_error("Unsupported file format: " + ext);
    }

    auto pixels = readRGBPixels();
    if (pixels.empty()) return;

    int w = pimpl_->size_.width();
    int h = pimpl_->size_.height();
    // WebGPU origin is top-left (no flip needed unlike GL)

    if (filename.has_parent_path() && !std::filesystem::exists(filename.parent_path())) {
        std::error_code ec;
        std::filesystem::create_directories(filename.parent_path(), ec);
    }

    bool success = false;
    if (ext == ".png") {
        success = stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), w * 3);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(filename.string().c_str(), w, h, 3, pixels.data(), 100);
    } else if (ext == ".bmp") {
        success = stbi_write_bmp(filename.string().c_str(), w, h, 3, pixels.data());
    }
    if (!success) {
        throw std::runtime_error("DawnRenderer: failed to write framebuffer to " + filename.string());
    }
}

void DawnRenderer::dispose() {
    pimpl_->dispose();
}

DawnRenderer::~DawnRenderer() = default;
