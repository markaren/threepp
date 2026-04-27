
#include "WgpuShaders.hpp"
#include "WgpuState.hpp"

#include <sstream>
#include <string>

using namespace threepp::wgpu;

std::string ShaderFeatures::describe(uint64_t features) {
    std::ostringstream s;
    auto append = [&](const char* name) {
        if (s.tellp() > 0) s << ", ";
        s << name;
    };

    if (features & Texture)         append("Texture");
    if (features & Lighting)        append("Lighting");
    if (features & Specular)        append("Specular");
    if (features & PBR)             append("PBR");
    if (features & NormalMap)       append("NormalMap");
    if (features & Shadow)          append("Shadow");
    if (features & FogLinear)       append("FogLinear");
    if (features & FogExp2)         append("FogExp2");
    if (features & InstanceColor)   append("InstanceColor");
    if (features & DisplacementMap) append("DisplacementMap");
    if (features & MorphTargets)    append("MorphTargets");
    if (features & Instanced)       append("Instanced");
    if (features & VertexColors)    append("VertexColors");
    if (features & EmissiveMap)     append("EmissiveMap");
    if (features & RoughnessMap)    append("RoughnessMap");
    if (features & MetalnessMap)    append("MetalnessMap");
    if (features & AOMap)           append("AOMap");
    if (features & AlphaMap)        append("AlphaMap");
    if (features & SpecularMap)     append("SpecularMap");
    if (features & LightMap)        append("LightMap");
    if (features & SRGBOutput)      append("SRGBOutput");
    if (features & BumpMap)         append("BumpMap");
    if (features & GradientMap)     append("GradientMap");
    if (features & EnvMap)          append("EnvMap");
    if (features & EnvMapCube)      append("EnvMapCube");
    if (features & Sheen)           append("Sheen");
    if (features & Skinning)        append("Skinning");
    if (features & ShadowMat)       append("ShadowMat");
    if (features & LineDashed)      append("LineDashed");
    if (features & NormalVis)       append("NormalVis");
    if (features & DepthVis)        append("DepthVis");
    if (features & DepthWriteOff)   append("DepthWriteOff");
    if (features & Wireframe)       append("Wireframe");

    // Multi-bit fields
    switch (features & CullMask) {
        case CullNone:  append("CullNone");  break;
        case CullFront: append("CullFront"); break;
        case CullBack:  append("CullBack");  break;
        default:        append("CullUnknown"); break;
    }

    switch (features & BlendMask) {
        case BlendNormal:      append("BlendNormal");      break;
        case BlendDisabled:    append("BlendDisabled");     break;
        case BlendAdditive:    append("BlendAdditive");     break;
        case BlendSubtractive: append("BlendSubtractive");  break;
        case BlendMultiply:    append("BlendMultiply");     break;
        default:               append("BlendUnknown");      break;
    }

    switch (features & TopoMask) {
        case TopoTriangle:  append("TopoTriangle");  break;
        case TopoLineList:  append("TopoLineList");  break;
        case TopoLineStrip: append("TopoLineStrip"); break;
        case TopoPointList: append("TopoPointList");  break;
        default:            append("TopoUnknown");    break;
    }

    switch (features & TonemapMask) {
        case TonemapNone:     append("TonemapNone");     break;
        case TonemapLinear:   append("TonemapLinear");   break;
        case TonemapReinhard: append("TonemapReinhard"); break;
        case TonemapCineon:   append("TonemapCineon");   break;
        case TonemapACES:     append("TonemapACES");     break;
        default:              append("TonemapUnknown");  break;
    }

    return s.str();
}

std::string threepp::wgpu::buildWGSL(uint64_t features, const LightLimits& limits, const ShadowLimits& shadowLimits) {
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
    transmissionParams: vec4<f32>,  // x=transmission, y=ior, z=thickness, w=samplerW
    attenuationParams: vec4<f32>,   // xyz=attenuationColor, w=attenuationDistance
    sheenColorAndRoughness: vec4<f32>, // xyz=sheenColor, w=sheenRoughness
    // Per-map KHR_texture_transform. Each pair (T0, T1) holds row0/row1 of the
    // 3x3 affine, so transformed UV = (dot(T0.xyz, vec3(uv,1)), dot(T1.xyz, vec3(uv,1))).
    // Identity = T0=(1,0,0,0), T1=(0,1,0,0).
    mapT0: vec4<f32>, mapT1: vec4<f32>,
    normalMapT0: vec4<f32>, normalMapT1: vec4<f32>,
    roughnessMapT0: vec4<f32>, roughnessMapT1: vec4<f32>,
    metalnessMapT0: vec4<f32>, metalnessMapT1: vec4<f32>,
    emissiveMapT0: vec4<f32>, emissiveMapT1: vec4<f32>,
    aoMapT0: vec4<f32>, aoMapT1: vec4<f32>,
    alphaMapT0: vec4<f32>, alphaMapT1: vec4<f32>,
    lightMapT0: vec4<f32>, lightMapT1: vec4<f32>,
    specularMapT0: vec4<f32>, specularMapT1: vec4<f32>,
    bumpMapT0: vec4<f32>, bumpMapT1: vec4<f32>,
    displacementMapT0: vec4<f32>, displacementMapT1: vec4<f32>,
};
@group(0) @binding(1) var<uniform> material: MaterialUniforms;
)";

    bool lit = ShaderFeatures::isLit(features);
    if (lit) {
        s << "struct DirectionalLightGPU { direction: vec3<f32>, _p0: f32, color: vec3<f32>, _p1: f32, };\n";
        s << "struct PointLightGPU { position: vec3<f32>, _p0: f32, color: vec3<f32>, distance: f32, decay: f32, _p1: f32, _p2: f32, _p3: f32, };\n";
        s << "struct SpotLightGPU { position: vec3<f32>, _p0: f32, direction: vec3<f32>, _p1: f32, color: vec3<f32>, distance: f32, decay: f32, coneCos: f32, penumbraCos: f32, _p2: f32, };\n";
        s << "struct HemisphereLightGPU { direction: vec3<f32>, _p0: f32, skyColor: vec3<f32>, _p1: f32, groundColor: vec3<f32>, _p2: f32, };\n";
        s << "struct RectAreaLightGPU { position: vec3<f32>, _p0: f32, color: vec3<f32>, _p1: f32, halfWidth: vec3<f32>, _p2: f32, halfHeight: vec3<f32>, _p3: f32, };\n";
        s << "struct LightData {\n";
        s << "  numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,\n";
        s << "  ambient: vec3<f32>, numRect: u32,\n";
        s << "  useLegacyLights: u32, _padHdr0: u32, _padHdr1: u32, _padHdr2: u32,\n";
        s << "  directional: array<DirectionalLightGPU, " << limits.maxDirLights << ">,\n";
        s << "  point: array<PointLightGPU, " << limits.maxPointLights << ">,\n";
        s << "  spot: array<SpotLightGPU, " << limits.maxSpotLights << ">,\n";
        s << "  hemi: array<HemisphereLightGPU, " << limits.maxHemiLights << ">,\n";
        s << "  rect: array<RectAreaLightGPU, " << limits.maxRectAreaLights << ">,\n";
        s << "};\n";
        s << "@group(0) @binding(2) var<uniform> lights: LightData;\n";
    }

    if (features & ShaderFeatures::Texture) {
        s << "@group(0) @binding(3) var t_diffuse: texture_2d<f32>;\n";
        s << "@group(0) @binding(4) var s_diffuse: sampler;\n";
    }
    if (features & ShaderFeatures::NormalMap) {
        s << "@group(0) @binding(5) var t_normalMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(6) var s_normalMap: sampler;\n";
    }

    if (features & ShaderFeatures::EmissiveMap) {
        s << "@group(0) @binding(10) var t_emissiveMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(11) var s_emissiveMap: sampler;\n";
    }
    if (features & ShaderFeatures::RoughnessMap) {
        s << "@group(0) @binding(12) var t_roughnessMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(13) var s_roughnessMap: sampler;\n";
    }
    if (features & ShaderFeatures::MetalnessMap) {
        s << "@group(0) @binding(14) var t_metalnessMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(15) var s_metalnessMap: sampler;\n";
    }
    if (features & ShaderFeatures::AOMap) {
        s << "@group(0) @binding(16) var t_aoMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(17) var s_aoMap: sampler;\n";
    }
    if (features & ShaderFeatures::AlphaMap) {
        s << "@group(0) @binding(18) var t_alphaMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(19) var s_alphaMap: sampler;\n";
    }
    if (features & ShaderFeatures::SpecularMap) {
        s << "@group(0) @binding(20) var t_specularMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(21) var s_specularMap: sampler;\n";
    }
    if (features & ShaderFeatures::LightMap) {
        s << "@group(0) @binding(22) var t_lightMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(23) var s_lightMap: sampler;\n";
    }
    if (features & ShaderFeatures::BumpMap) {
        s << "@group(0) @binding(24) var t_bumpMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(25) var s_bumpMap: sampler;\n";
    }
    if (features & ShaderFeatures::GradientMap) {
        s << "@group(0) @binding(26) var t_gradientMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(27) var s_gradientMap: sampler;\n";
    }
    if (features & ShaderFeatures::DisplacementMap) {
        s << "@group(0) @binding(30) var t_displacementMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(31) var s_displacementMap: sampler;\n";
    }
    if (features & ShaderFeatures::EnvMap) {
        s << "@group(0) @binding(32) var t_envMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(33) var s_envMap: sampler;\n";
    }
    if (features & ShaderFeatures::EnvMapCube) {
        s << "@group(0) @binding(32) var t_envMap: texture_cube<f32>;\n";
        s << "@group(0) @binding(33) var s_envMap: sampler;\n";
    }
    if (features & ShaderFeatures::Transmission) {
        s << "@group(0) @binding(38) var t_transmissionMap: texture_2d<f32>;\n";
        s << "@group(0) @binding(39) var s_transmissionMap: sampler;\n";
    }
    if (features & ShaderFeatures::RectAreaLights) {
        s << "@group(0) @binding(40) var t_ltc_1: texture_2d<f32>;\n";
        s << "@group(0) @binding(41) var t_ltc_2: texture_2d<f32>;\n";
        s << "@group(0) @binding(42) var s_ltc: sampler;\n";
    }

    if (features & ShaderFeatures::Instanced) {
        s << "struct InstanceData {\n";
        s << "    model: mat4x4<f32>,\n";
        if (features & ShaderFeatures::InstanceColor) {
            s << "    color: vec4<f32>,\n";
        }
        s << "};\n";
        s << "@group(0) @binding(28) var<storage, read> instances: array<InstanceData>;\n";
    }

    if (features & ShaderFeatures::MorphTargets) {
        s << "struct MorphData {\n";
        s << "    numTargets: u32,\n";
        s << "    _pad0: u32, _pad1: u32, _pad2: u32,\n";
        s << "    influences: array<vec4<f32>, 2>,\n";
        s << "    positions: array<vec4<f32>>,\n";
        s << "};\n";
        s << "@group(0) @binding(29) var<storage, read> morph: MorphData;\n";
    }

    if (features & ShaderFeatures::Skinning) {
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

    if (features & ShaderFeatures::Shadow) {
        s << "const MAX_SHADOW_LIGHTS: u32 = " << shadowLimits.maxShadowLights << "u;\n";
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
    numDirShadows: u32,
    numSpotShadows: u32,
    _pad0: u32,
    lights: array<ShadowLightData, MAX_SHADOW_LIGHTS>,
};
@group(0) @binding(7) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(8) var t_shadowMaps: texture_depth_2d_array;
@group(0) @binding(9) var s_shadowMap: sampler_comparison;

fn sampleShadow(si: u32, worldPos: vec3<f32>) -> f32 {
    let worldPos4 = vec4<f32>(worldPos, 1.0);
    let lightSpacePos = shadow.lights[si].lightVP * worldPos4;
    let projCoords = lightSpacePos.xyz / lightSpacePos.w;
    let shadowUV = vec2<f32>(projCoords.x * 0.5 + 0.5, 1.0 - (projCoords.y * 0.5 + 0.5));
    let currentDepth = projCoords.z - shadow.lights[si].bias;
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0 || currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }
    let texelSize = 1.0 / f32(textureDimensions(t_shadowMaps).x);
    var shadow_sum = 0.0;
    for (var sy = -1; sy <= 1; sy++) {
        for (var sx = -1; sx <= 1; sx++) {
            let offset = vec2<f32>(f32(sx), f32(sy)) * texelSize;
            shadow_sum += textureSampleCompare(t_shadowMaps, s_shadowMap, shadowUV + offset, i32(si), currentDepth);
        }
    }
    return shadow_sum / 9.0;
}
)";
        // Point light shadow structs and sampling function
        s << "const MAX_SHADOW_POINT_LIGHTS: u32 = " << shadowLimits.maxShadowPointLights << "u;\n";
        s << R"(
struct PointShadowData {
    position: vec3<f32>,
    near: f32,
    bias: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
};
struct PointShadowUniforms {
    count: u32,
    _p0: u32, _p1: u32, _p2: u32,
    lights: array<PointShadowData, MAX_SHADOW_POINT_LIGHTS>,
};
@group(0) @binding(36) var<uniform> pointShadow: PointShadowUniforms;
@group(0) @binding(37) var t_ptShadowMaps: texture_depth_2d_array;

fn samplePointShadow(pi: u32, worldPos: vec3<f32>) -> f32 {
    let psd = pointShadow.lights[pi];
    let L = worldPos - psd.position;
    let aL = abs(L);
    var d: f32;
    var uv: vec2<f32>;
    var faceIdx: i32;
    if (aL.x >= aL.y && aL.x >= aL.z) {
        d = aL.x;
        if (L.x > 0.0) {
            faceIdx = 0;
            uv = vec2<f32>(L.z / (2.0*d) + 0.5, 0.5 - L.y / (2.0*d));
        } else {
            faceIdx = 1;
            uv = vec2<f32>(0.5 - L.z / (2.0*d), 0.5 - L.y / (2.0*d));
        }
    } else if (aL.y >= aL.x && aL.y >= aL.z) {
        d = aL.y;
        if (L.y > 0.0) {
            faceIdx = 4;
            uv = vec2<f32>(L.x / (2.0*d) + 0.5, 0.5 - L.z / (2.0*d));
        } else {
            faceIdx = 5;
            uv = vec2<f32>(L.x / (2.0*d) + 0.5, L.z / (2.0*d) + 0.5);
        }
    } else {
        d = aL.z;
        if (L.z > 0.0) {
            faceIdx = 2;
            uv = vec2<f32>(0.5 - L.x / (2.0*d), 0.5 - L.y / (2.0*d));
        } else {
            faceIdx = 3;
            uv = vec2<f32>(L.x / (2.0*d) + 0.5, 0.5 - L.y / (2.0*d));
        }
    }
    let near = psd.near;
    let far = psd.far;
    let depthRef = far * (d - near) / (d * (far - near)) - psd.bias;
    if (depthRef < 0.0 || depthRef > 1.0) { return 1.0; }
    let layer = i32(pi) * 6 + faceIdx;
    let texelSize = 1.0 / f32(textureDimensions(t_ptShadowMaps).x);
    var shadow_sum = 0.0;
    for (var sy = -1; sy <= 1; sy++) {
        for (var sx = -1; sx <= 1; sx++) {
            let offset = vec2<f32>(f32(sx), f32(sy)) * texelSize;
            shadow_sum += textureSampleCompare(t_ptShadowMaps, s_shadowMap, uv + offset, layer, depthRef);
        }
    }
    return shadow_sum / 9.0;
}
)";
    }

    // LTC helper functions for RectAreaLight (Heitz/Hill 2016).
    if (features & ShaderFeatures::RectAreaLights) {
        s << R"(
fn ltc_edgeVectorFormFactor(v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let x = dot(v1, v2);
    let y = abs(x);
    let a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    let b = 3.4175940 + (4.1616724 + y) * y;
    let v = a / b;
    var theta_sintheta: f32;
    if (x > 0.0) {
        theta_sintheta = v;
    } else {
        theta_sintheta = 0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - v;
    }
    return cross(v1, v2) * theta_sintheta;
}

fn ltc_clippedSphereFormFactor(f: vec3<f32>) -> f32 {
    let l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
}

fn ltc_evaluate(N: vec3<f32>, V: vec3<f32>, P: vec3<f32>, mInv: mat3x3<f32>, rectCoords: array<vec3<f32>, 4>) -> vec3<f32> {
    // bail if point is on the back side of the light plane (CCW winding)
    let v1 = rectCoords[1] - rectCoords[0];
    let v2 = rectCoords[3] - rectCoords[0];
    let lightNormal = cross(v1, v2);
    if (dot(lightNormal, P - rectCoords[0]) < 0.0) {
        return vec3<f32>(0.0);
    }

    // orthonormal basis around N
    let T1 = normalize(V - N * dot(V, N));
    let T2 = -cross(N, T1);

    // world -> cosine-space transform
    let tbn = mat3x3<f32>(T1, T2, N);
    // transpose(tbn) = inverse(tbn) since it's orthonormal.
    let mat = mInv * transpose(tbn);

    var coords: array<vec3<f32>, 4>;
    coords[0] = normalize(mat * (rectCoords[0] - P));
    coords[1] = normalize(mat * (rectCoords[1] - P));
    coords[2] = normalize(mat * (rectCoords[2] - P));
    coords[3] = normalize(mat * (rectCoords[3] - P));

    var vff = vec3<f32>(0.0);
    vff += ltc_edgeVectorFormFactor(coords[0], coords[1]);
    vff += ltc_edgeVectorFormFactor(coords[1], coords[2]);
    vff += ltc_edgeVectorFormFactor(coords[2], coords[3]);
    vff += ltc_edgeVectorFormFactor(coords[3], coords[0]);

    let result = ltc_clippedSphereFormFactor(vff);
    return vec3<f32>(result);
}
)";
    }

    s << "\nstruct VertexInput {\n";
    s << "    @location(0) position: vec3<f32>,\n";
    s << "    @location(1) normal: vec3<f32>,\n";
    s << "    @location(2) uv: vec2<f32>,\n";
    s << "    @location(3) color: vec3<f32>,\n";
    s << "    @location(4) uv2: vec2<f32>,\n";
    s << "};\n";

    s << "struct VertexOutput {\n";
    s << "    @builtin(position) clipPos: vec4<f32>,\n";
    s << "    @location(0) worldPos: vec3<f32>,\n";
    s << "    @location(1) worldNormal: vec3<f32>,\n";
    s << "    @location(2) uv: vec2<f32>,\n";
    s << "    @location(3) uv2: vec2<f32>,\n";
    if (features & ShaderFeatures::VertexColors) {
        s << "    @location(4) vertexColor: vec3<f32>,\n";
    }
    if (features & ShaderFeatures::InstanceColor) {
        s << "    @location(5) instanceColor: vec3<f32>,\n";
    }
    s << "};\n";

    // Build vertex function signature with optional builtins
    s << "\n@vertex\n";
    {
        bool needInstanceIdx = features & ShaderFeatures::Instanced;
        bool needVertexIdx = (features & ShaderFeatures::MorphTargets) || (features & ShaderFeatures::Skinning);
        s << "fn vs_main(in: VertexInput";
        if (needInstanceIdx) s << ", @builtin(instance_index) iid: u32";
        if (needVertexIdx) s << ", @builtin(vertex_index) vid: u32";
        s << ") -> VertexOutput {\n";
    }
    s << "    var out: VertexOutput;\n";

    // Morph target blending
    if (features & ShaderFeatures::MorphTargets) {
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

    // Displacement map
    if (features & ShaderFeatures::DisplacementMap) {
        std::string posVar = (features & ShaderFeatures::MorphTargets) ? "morphedPos" : "in.position";
        s << "    let _dispUv3 = vec3<f32>(select(in.uv, in.uv2, material.displacementMapT1.w > 0.5), 1.0);\n";
        s << "    let _dispUv = vec2<f32>(dot(material.displacementMapT0.xyz, _dispUv3), dot(material.displacementMapT1.xyz, _dispUv3));\n";
        s << "    let dispAmount = textureSampleLevel(t_displacementMap, s_displacementMap, _dispUv, 0.0).r;\n";
        s << "    let displacedPos = " << posVar << " + in.normal * dispAmount * material.roughnessMetalnessOpacity.w;\n";
    }

    // Determine which position variable to use
    std::string posExpr = "in.position";
    if (features & ShaderFeatures::MorphTargets) posExpr = "morphedPos";
    if (features & ShaderFeatures::DisplacementMap) posExpr = "displacedPos";

    // Skinning
    if (features & ShaderFeatures::Skinning) {
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
    if (features & ShaderFeatures::Skinning) normalExpr = "finalSkinNormal";

    if (features & ShaderFeatures::Instanced) {
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
    s << "    out.uv2 = in.uv2;\n";
    s << "    out.clipPos = transform.proj * transform.view * worldPos4;\n";

    if (features & ShaderFeatures::VertexColors) {
        s << "\n    out.vertexColor = in.color;\n";
    }
    if (features & ShaderFeatures::InstanceColor) {
        s << "    out.instanceColor = instances[iid].color.rgb;\n";
    }

    s << R"(
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) isFrontFacing: bool) -> @location(0) vec4<f32> {
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
    // Dashed line discard: lineDistance in uv.x, dash params in specularAndShininess
    if (features & ShaderFeatures::LineDashed) {
        s << "    // LineDashedMaterial: discard fragments in gaps\n";
        s << "    let dashSize = material.specularAndShininess.x;\n";
        s << "    let totalSize = material.specularAndShininess.y;\n";
        s << "    let dashScale = material.specularAndShininess.z;\n";
        s << "    let vLineDistance = in.uv.x * dashScale;\n";
        s << "    if (totalSize > 0.0 && (vLineDistance - floor(vLineDistance / totalSize) * totalSize) > dashSize) { discard; }\n";
    }
    if (features & ShaderFeatures::VertexColors) {
        s << "    baseColor = baseColor * in.vertexColor;\n";
    }
    if (features & ShaderFeatures::InstanceColor) {
        s << "    baseColor = baseColor * in.instanceColor;\n";
    }

    if (features & ShaderFeatures::Texture) {
        s << "    let _mapSrc = vec3<f32>(select(in.uv, in.uv2, material.mapT1.w > 0.5), 1.0);\n";
        s << "    let _mapUv = vec2<f32>(dot(material.mapT0.xyz, _mapSrc), dot(material.mapT1.xyz, _mapSrc));\n";
        s << "    let texColor = textureSample(t_diffuse, s_diffuse, _mapUv);\n";
        s << "    baseColor = baseColor * texColor.rgb;\n";
        s << "    opacity = opacity * texColor.a;\n";
    }
    if (features & ShaderFeatures::RoughnessMap) {
        s << "    let _roughSrc = vec3<f32>(select(in.uv, in.uv2, material.roughnessMapT1.w > 0.5), 1.0);\n";
        s << "    let _roughUv = vec2<f32>(dot(material.roughnessMapT0.xyz, _roughSrc), dot(material.roughnessMapT1.xyz, _roughSrc));\n";
        s << "    roughness = roughness * textureSample(t_roughnessMap, s_roughnessMap, _roughUv).g;\n";
    }
    if (features & ShaderFeatures::MetalnessMap) {
        s << "    let _metalSrc = vec3<f32>(select(in.uv, in.uv2, material.metalnessMapT1.w > 0.5), 1.0);\n";
        s << "    let _metalUv = vec2<f32>(dot(material.metalnessMapT0.xyz, _metalSrc), dot(material.metalnessMapT1.xyz, _metalSrc));\n";
        s << "    metalness = metalness * textureSample(t_metalnessMap, s_metalnessMap, _metalUv).b;\n";
    }
    if (features & ShaderFeatures::AlphaMap) {
        s << "    let _alphaSrc = vec3<f32>(select(in.uv, in.uv2, material.alphaMapT1.w > 0.5), 1.0);\n";
        s << "    let _alphaUv = vec2<f32>(dot(material.alphaMapT0.xyz, _alphaSrc), dot(material.alphaMapT1.xyz, _alphaSrc));\n";
        s << "    opacity = opacity * textureSample(t_alphaMap, s_alphaMap, _alphaUv).r;\n";
    }

    // ShadowMaterial: output shadow attenuation only, skip full lighting
    if (features & ShaderFeatures::ShadowMat) {
        if (features & ShaderFeatures::Shadow) {
            // Compute combined shadow factor from all shadow-casting lights
            s << "    var shadowFactor = 1.0;\n";
            s << "    for (var si = 0u; si < shadow.count; si++) {\n";
            s << "        shadowFactor = min(shadowFactor, sampleShadow(si, in.worldPos));\n";
            s << "    }\n";
            // Where shadowFactor == 1.0 (lit): fully transparent
            // Where shadowFactor < 1.0 (shadow): show shadow color with opacity
            s << "    return vec4<f32>(baseColor, opacity * (1.0 - shadowFactor));\n";
        } else {
            // No shadows active: fully transparent (nothing to show)
            s << "    return vec4<f32>(baseColor, 0.0);\n";
        }
        s << "}\n";
        return s.str();
    }

    // MeshNormalMaterial: map world-space normal to RGB color
    if (features & ShaderFeatures::NormalVis) {
        s << "    var N = normalize(in.worldNormal);\n";
        s << "    if (!isFrontFacing) { N = -N; }\n";
        s << "    baseColor = N * 0.5 + vec3<f32>(0.5);\n";
        s << "    return vec4<f32>(baseColor, opacity);\n}\n";
        return s.str();
    }

    // MeshDepthMaterial: map framebuffer depth to brightness.
    // In the fragment stage @builtin(position).z is already the [0,1]
    // framebuffer depth — dividing by .w (which is 1/clip.w) is wrong.
    if (features & ShaderFeatures::DepthVis) {
        s << "    let brightness = 1.0 - in.clipPos.z;\n";
        s << "    baseColor = vec3<f32>(brightness);\n";
        s << "    return vec4<f32>(baseColor, opacity);\n}\n";
        return s.str();
    }

    if (lit) {
        if (features & ShaderFeatures::NormalMap) {
            s << R"(
    // Screen-space normal map perturbation (no tangent attributes needed)
    let _normalSrc = vec3<f32>(select(in.uv, in.uv2, material.normalMapT1.w > 0.5), 1.0);
    let _normalUv = vec2<f32>(dot(material.normalMapT0.xyz, _normalSrc), dot(material.normalMapT1.xyz, _normalSrc));
    let dPdx = dpdx(in.worldPos);
    let dPdy = dpdy(in.worldPos);
    let dUVdx = dpdx(_normalUv);
    let dUVdy = dpdy(_normalUv);
    let T = normalize(dPdx * dUVdy.y - dPdy * dUVdx.y);
    let B = normalize(dPdy * dUVdx.x - dPdx * dUVdy.x);
    let geomN = normalize(in.worldNormal);
    let TBN = mat3x3<f32>(T, B, geomN);
    let nmSample = textureSample(t_normalMap, s_normalMap, _normalUv).rgb * 2.0 - vec3<f32>(1.0);
    let normalScale = material.flags.zw;
    let scaledNm = vec3<f32>(nmSample.xy * normalScale, nmSample.z);
    var N = normalize(TBN * scaledNm);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
        if ((features & ShaderFeatures::CullMask) == ShaderFeatures::CullNone) {
            s << "    if (!isFrontFacing) { N = -N; }\n";
        }
        } else if (features & ShaderFeatures::BumpMap) {
            s << R"(
    let _bumpSrc = vec3<f32>(select(in.uv, in.uv2, material.bumpMapT1.w > 0.5), 1.0);
    let _bumpUv = vec2<f32>(dot(material.bumpMapT0.xyz, _bumpSrc), dot(material.bumpMapT1.xyz, _bumpSrc));
    let bmp_dPdx = dpdx(in.worldPos);
    let bmp_dPdy = dpdy(in.worldPos);
    let bmp_dUVdx = dpdx(_bumpUv);
    let bmp_dUVdy = dpdy(_bumpUv);
    let Hll = textureSample(t_bumpMap, s_bumpMap, _bumpUv).r;
    let dBx = textureSample(t_bumpMap, s_bumpMap, _bumpUv + bmp_dUVdx).r - Hll;
    let dBy = textureSample(t_bumpMap, s_bumpMap, _bumpUv + bmp_dUVdy).r - Hll;
    let bumpScale = material.flags.y;
    let geomN = normalize(in.worldNormal);
    let crossX = cross(bmp_dPdy, geomN);
    let crossY = cross(geomN, bmp_dPdx);
    let surfGrad = (crossX * dBx + crossY * dBy) * bumpScale;
    var N = normalize(in.worldNormal - surfGrad);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
        if ((features & ShaderFeatures::CullMask) == ShaderFeatures::CullNone) {
            s << "    if (!isFrontFacing) { N = -N; }\n";
        }
        } else {
            s << R"(
    var N = normalize(in.worldNormal);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
        if ((features & ShaderFeatures::CullMask) == ShaderFeatures::CullNone) {
            s << "    if (!isFrontFacing) { N = -N; }\n";
        }
        }
        s << R"(
    var diffuseLight = lights.ambient;
    var specularLight = vec3<f32>(0.0, 0.0, 0.0);
)";
        if (features & ShaderFeatures::Sheen) {
            s << "    var sheenLight = vec3<f32>(0.0, 0.0, 0.0);\n";
            s << "    let sheenColor = material.sheenColorAndRoughness.rgb;\n";
            s << "    let sheenAlpha = max(material.sheenColorAndRoughness.w * material.sheenColorAndRoughness.w, 0.0049);\n";
        }
        s << R"(
    for (var i = 0u; i < lights.numDir; i++) {
        let L = normalize(lights.directional[i].direction);
        var NdotL = max(dot(N, L), 0.0);
)";
        if (features & ShaderFeatures::GradientMap) {
            s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
        }
        // Per-light shadow attenuation for directional lights
        if (features & ShaderFeatures::Shadow) {
            s << "        var dirShadow = 1.0;\n";
            s << "        if (i < shadow.numDirShadows) { dirShadow = sampleShadow(i, in.worldPos); }\n";
            s << "        diffuseLight += lights.directional[i].color * NdotL * dirShadow;\n";
        } else {
            s << "        diffuseLight += lights.directional[i].color * NdotL;\n";
        }
        if (features & ShaderFeatures::Specular) {
            s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.directional[i].color * material.specularAndShininess.rgb * s * dirShadow; }\n";
            } else {
                s << "          specularLight += lights.directional[i].color * material.specularAndShininess.rgb * s; }\n";
            }
        }
        if (features & ShaderFeatures::PBR) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let NdotV = max(dot(N, V), 0.001);\n";
            s << "          let r = roughness; let m = metalness;\n";
            s << "          let a = r * r; let a2 = a * a;\n";
            s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
            s << "          let D = a2 / (3.14159265 * denom * denom);\n";
            s << "          let F0 = mix(vec3<f32>(0.04) * material.specularAndShininess.rgb * material.specularAndShininess.w, baseColor, m);\n";
            s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.directional[i].color * F * D * NdotL * dirShadow; }\n";
            } else {
                s << "          specularLight += lights.directional[i].color * F * D * NdotL; }\n";
            }
        }
        if (features & ShaderFeatures::Sheen) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let NdotV = max(dot(N, V), 0.001);\n";
            s << "          let oneMinusNdotH2 = max(1.0 - NdotH * NdotH, 0.0);\n";
            s << "          let Ds = (2.0 + 1.0 / sheenAlpha) * pow(oneMinusNdotH2, 0.5 / sheenAlpha) * 0.15915494;\n"; // /(2π)
            s << "          let Vs = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV + 0.0001));\n";
            s << "          let sTerm = sheenColor * Ds * Vs * NdotL;\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          sheenLight += lights.directional[i].color * sTerm * dirShadow; }\n";
            } else {
                s << "          sheenLight += lights.directional[i].color * sTerm; }\n";
            }
        }
        s << "    }\n";

        s << R"(
    for (var i = 0u; i < lights.numPoint; i++) {
        let lv = lights.point[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        var NdotL = max(dot(N, L), 0.0);
        // Distance falloff: legacy = pow(saturate(1 - d/cutoff), decay);
        // physical = Frostbite 1/d^decay * smooth window.
        var att = 1.0;
        if (lights.useLegacyLights == 1u) {
            if (lights.point[i].distance > 0.0 && lights.point[i].decay > 0.0) {
                att = pow(saturate(-d / lights.point[i].distance + 1.0), lights.point[i].decay);
            }
        } else {
            att = 1.0 / max(pow(d, lights.point[i].decay), 0.01);
            if (lights.point[i].distance > 0.0) {
                let r = d / lights.point[i].distance;
                let r4 = r * r * r * r;
                let w = saturate(1.0 - r4);
                att = att * (w * w);
            }
        }
)";
        if (features & ShaderFeatures::GradientMap) {
            s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
        }
        // Per-light shadow attenuation for point lights
        if (features & ShaderFeatures::Shadow) {
            s << "        var ptShadow = 1.0;\n";
            s << "        if (i < pointShadow.count) { ptShadow = samplePointShadow(i, in.worldPos); }\n";
            s << "        diffuseLight += lights.point[i].color * NdotL * att * ptShadow;\n";
        } else {
            s << "        diffuseLight += lights.point[i].color * NdotL * att;\n";
        }
        if (features & ShaderFeatures::Specular) {
            s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.point[i].color * material.specularAndShininess.rgb * s * att * ptShadow; }\n";
            } else {
                s << "          specularLight += lights.point[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
        }
        if (features & ShaderFeatures::PBR) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let r = roughness; let m = metalness;\n";
            s << "          let a = r * r; let a2 = a * a;\n";
            s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
            s << "          let D = a2 / (3.14159265 * denom * denom);\n";
            s << "          let F0 = mix(vec3<f32>(0.04) * material.specularAndShininess.rgb * material.specularAndShininess.w, baseColor, m);\n";
            s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.point[i].color * F * D * NdotL * att * ptShadow; }\n";
            } else {
                s << "          specularLight += lights.point[i].color * F * D * NdotL * att; }\n";
            }
        }
        if (features & ShaderFeatures::Sheen) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let NdotV = max(dot(N, V), 0.001);\n";
            s << "          let oneMinusNdotH2 = max(1.0 - NdotH * NdotH, 0.0);\n";
            s << "          let Ds = (2.0 + 1.0 / sheenAlpha) * pow(oneMinusNdotH2, 0.5 / sheenAlpha) * 0.15915494;\n";
            s << "          let Vs = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV + 0.0001));\n";
            s << "          let sTerm = sheenColor * Ds * Vs * NdotL;\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          sheenLight += lights.point[i].color * sTerm * att * ptShadow; }\n";
            } else {
                s << "          sheenLight += lights.point[i].color * sTerm * att; }\n";
            }
        }
        s << "    }\n";

        s << R"(
    for (var i = 0u; i < lights.numSpot; i++) {
        let lv = lights.spot[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        var NdotL = max(dot(N, L), 0.0);
        let ac = dot(L, lights.spot[i].direction);
        let se = smoothstep(lights.spot[i].coneCos, lights.spot[i].penumbraCos, ac);
        var att = se;
        // Distance falloff: same legacy/physical branch as point lights.
        if (lights.useLegacyLights == 1u) {
            if (lights.spot[i].distance > 0.0 && lights.spot[i].decay > 0.0) {
                att = att * pow(saturate(-d / lights.spot[i].distance + 1.0), lights.spot[i].decay);
            }
        } else {
            var dFall = 1.0 / max(pow(d, lights.spot[i].decay), 0.01);
            if (lights.spot[i].distance > 0.0) {
                let r = d / lights.spot[i].distance;
                let r4 = r * r * r * r;
                let w = saturate(1.0 - r4);
                dFall = dFall * (w * w);
            }
            att = att * dFall;
        }
)";
        if (features & ShaderFeatures::GradientMap) {
            s << "        NdotL = textureSample(t_gradientMap, s_gradientMap, vec2<f32>(NdotL, 0.5)).r;\n";
        }
        // Per-light shadow attenuation for spot lights
        if (features & ShaderFeatures::Shadow) {
            s << "        var spotShadow = 1.0;\n";
            s << "        if (i < shadow.numSpotShadows) { spotShadow = sampleShadow(shadow.numDirShadows + i, in.worldPos); }\n";
            s << "        diffuseLight += lights.spot[i].color * NdotL * att * spotShadow;\n";
        } else {
            s << "        diffuseLight += lights.spot[i].color * NdotL * att;\n";
        }
        if (features & ShaderFeatures::Specular) {
            s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.spot[i].color * material.specularAndShininess.rgb * s * att * spotShadow; }\n";
            } else {
                s << "          specularLight += lights.spot[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
        }
        if (features & ShaderFeatures::PBR) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let r = roughness; let m = metalness;\n";
            s << "          let a = r * r; let a2 = a * a;\n";
            s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
            s << "          let D = a2 / (3.14159265 * denom * denom);\n";
            s << "          let F0 = mix(vec3<f32>(0.04) * material.specularAndShininess.rgb * material.specularAndShininess.w, baseColor, m);\n";
            s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          specularLight += lights.spot[i].color * F * D * NdotL * att * spotShadow; }\n";
            } else {
                s << "          specularLight += lights.spot[i].color * F * D * NdotL * att; }\n";
            }
        }
        if (features & ShaderFeatures::Sheen) {
            s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
            s << "          let NdotV = max(dot(N, V), 0.001);\n";
            s << "          let oneMinusNdotH2 = max(1.0 - NdotH * NdotH, 0.0);\n";
            s << "          let Ds = (2.0 + 1.0 / sheenAlpha) * pow(oneMinusNdotH2, 0.5 / sheenAlpha) * 0.15915494;\n";
            s << "          let Vs = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV + 0.0001));\n";
            s << "          let sTerm = sheenColor * Ds * Vs * NdotL;\n";
            if (features & ShaderFeatures::Shadow) {
                s << "          sheenLight += lights.spot[i].color * sTerm * att * spotShadow; }\n";
            } else {
                s << "          sheenLight += lights.spot[i].color * sTerm * att; }\n";
            }
        }
        s << R"(
    }
    for (var i = 0u; i < lights.numHemi; i++) {
        let w = 0.5 * dot(N, lights.hemi[i].direction) + 0.5;
        diffuseLight += mix(lights.hemi[i].groundColor, lights.hemi[i].skyColor, w);
    }
)";
        // RectAreaLight LTC evaluation (Heitz/Hill 2016). Requires PBR.
        if ((features & ShaderFeatures::RectAreaLights) && (features & ShaderFeatures::PBR)) {
            s << R"(
    for (var i = 0u; i < lights.numRect; i++) {
        let lightPos = lights.rect[i].position;
        let halfW = lights.rect[i].halfWidth;
        let halfH = lights.rect[i].halfHeight;
        let lightColor = lights.rect[i].color;

        // CCW winding; emissive face points -Z in local space
        var rectCoords: array<vec3<f32>, 4>;
        rectCoords[0] = lightPos + halfW - halfH;
        rectCoords[1] = lightPos - halfW - halfH;
        rectCoords[2] = lightPos - halfW + halfH;
        rectCoords[3] = lightPos + halfW + halfH;

        let dotNV = clamp(dot(N, V), 0.0, 1.0);
        let lutUV = vec2<f32>(roughness, sqrt(1.0 - dotNV)) * (63.0 / 64.0) + vec2<f32>(0.5 / 64.0);
        let t1 = textureSampleLevel(t_ltc_1, s_ltc, lutUV, 0.0);
        let t2 = textureSampleLevel(t_ltc_2, s_ltc, lutUV, 0.0);

        let mInv = mat3x3<f32>(
            vec3<f32>(t1.x, 0.0, t1.y),
            vec3<f32>(0.0,  1.0, 0.0 ),
            vec3<f32>(t1.z, 0.0, t1.w)
        );

        let F0 = mix(vec3<f32>(0.04) * material.specularAndShininess.rgb * material.specularAndShininess.w, baseColor, metalness);
        let ltcFresnel = F0 * t2.x + (vec3<f32>(1.0) - F0) * t2.y;

        let ltcSpec = ltc_evaluate(N, V, in.worldPos, mInv, rectCoords);
        let ltcDiff = ltc_evaluate(N, V, in.worldPos, mat3x3<f32>(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0), rectCoords);

        specularLight += lightColor * ltcFresnel * ltcSpec;
        diffuseLight  += lightColor * (1.0 - metalness) * ltcDiff;
    }
)";
        }
        // Apply Lambert /π divisor to direct + ambient diffuse light in
        // physical mode. Legacy mode keeps the historical no-divisor scale.
        // Mirrors three.js BRDF_Lambertian = diffuseColor / π.
        s << "    diffuseLight = diffuseLight * select(0.31830989, 1.0, lights.useLegacyLights == 1u);\n";
        // Emissive
        if (features & ShaderFeatures::EmissiveMap) {
            s << "    let _emissiveSrc = vec3<f32>(select(in.uv, in.uv2, material.emissiveMapT1.w > 0.5), 1.0);\n";
            s << "    let _emissiveUv = vec2<f32>(dot(material.emissiveMapT0.xyz, _emissiveSrc), dot(material.emissiveMapT1.xyz, _emissiveSrc));\n";
            s << "    var emissiveColor = material.emissive.rgb * textureSample(t_emissiveMap, s_emissiveMap, _emissiveUv).rgb;\n";
        } else {
            s << "    var emissiveColor = material.emissive.rgb;\n";
        }

        // SpecularMap
        if (features & ShaderFeatures::SpecularMap) {
            s << "    let _specularSrc = vec3<f32>(select(in.uv, in.uv2, material.specularMapT1.w > 0.5), 1.0);\n";
            s << "    let _specularUv = vec2<f32>(dot(material.specularMapT0.xyz, _specularSrc), dot(material.specularMapT1.xyz, _specularSrc));\n";
            s << "    specularLight = specularLight * textureSample(t_specularMap, s_specularMap, _specularUv).rgb;\n";
        }

        if (features & ShaderFeatures::PBR) {
            if (features & ShaderFeatures::Transmission) {
                s << "    let albedoTint = baseColor;\n";
            }
            s << "    let F0 = mix(vec3<f32>(0.04) * material.specularAndShininess.rgb * material.specularAndShininess.w, baseColor, metalness);\n";
            if (features & (ShaderFeatures::EnvMap | ShaderFeatures::EnvMapCube)) {
                s << "    let R = reflect(-V, N);\n";
                s << "    let maxMip    = f32(textureNumLevels(t_envMap) - 1u);\n";
                s << "    let envMip    = clamp(roughness * maxMip, 0.0, maxMip);\n";
                if (features & ShaderFeatures::EnvMapCube) {
                    s << "    let envColor  = textureSampleLevel(t_envMap, s_envMap, R, envMip).rgb;\n";
                    s << "    let envDiff   = textureSampleLevel(t_envMap, s_envMap, N, maxMip).rgb;\n";
                } else {
                    s << "    let envPhi    = atan2(R.z, R.x) * 0.15915494 + 0.5;\n";  // 1/(2π), three.js convention: atan2(z, x)
                    s << "    let envTheta  = asin(clamp(R.y, -1.0, 1.0)) * 0.31830989 + 0.5;\n";  // 1/π
                    s << "    let envColor  = textureSampleLevel(t_envMap, s_envMap, vec2<f32>(envPhi, envTheta), envMip).rgb;\n";
                    s << "    let nPhi      = atan2(N.z, N.x) * 0.15915494 + 0.5;\n";
                    s << "    let nTheta    = asin(clamp(N.y, -1.0, 1.0)) * 0.31830989 + 0.5;\n";
                    s << "    let envDiff   = textureSampleLevel(t_envMap, s_envMap, vec2<f32>(nPhi, nTheta), maxMip).rgb;\n";
                }
                s << "    let envFresnel   = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);\n";
                s << "    let indirSpec    = envFresnel * envColor;\n";
                s << "    let indirDiff    = envDiff * baseColor * (1.0 - metalness);\n";
                s << "    let ambientSpecOnly = indirSpec * material.emissive.w;\n";
                s << "    let ambientDiff     = indirDiff * material.emissive.w;\n";
                if (features & ShaderFeatures::Sheen) {
                    s << "    let sheenIBL = envDiff * sheenColor * material.emissive.w;\n";
                    s << "    sheenLight += sheenIBL;\n";
                }
            } else {
                s << "    let ambientSpecOnly = F0 * lights.ambient * (1.0 - roughness);\n";
                s << "    let ambientDiff     = vec3<f32>(0.0);\n";
            }
            if (features & ShaderFeatures::Sheen) {
                s << "    let sheenScale = max(max(sheenColor.r, sheenColor.g), sheenColor.b);\n";
                s << "    let sheenAtt = clamp(1.0 - 0.157 * sheenScale, 0.0, 1.0);\n";
                if (features & ShaderFeatures::Transmission) {
                    s << "    let totalDiffuse  = (baseColor * (1.0 - metalness) * diffuseLight + ambientDiff) * sheenAtt;\n";
                    s << "    let totalSpecular = (specularLight + ambientSpecOnly) * sheenAtt + sheenLight;\n";
                    s << "    baseColor = totalDiffuse + totalSpecular;\n";
                } else {
                    s << "    baseColor = (baseColor * (1.0 - metalness) * diffuseLight + specularLight + ambientSpecOnly + ambientDiff) * sheenAtt + sheenLight;\n";
                }
            } else {
                if (features & ShaderFeatures::Transmission) {
                    s << "    let totalDiffuse  = baseColor * (1.0 - metalness) * diffuseLight + ambientDiff;\n";
                    s << "    let totalSpecular = specularLight + ambientSpecOnly;\n";
                    s << "    baseColor = totalDiffuse + totalSpecular;\n";
                } else {
                    s << "    baseColor = baseColor * (1.0 - metalness) * diffuseLight + specularLight + ambientSpecOnly + ambientDiff;\n";
                }
            }
        } else {
            if (features & (ShaderFeatures::EnvMap | ShaderFeatures::EnvMapCube)) {
                s << "    let R = reflect(-V, N);\n";
                if (features & ShaderFeatures::EnvMapCube) {
                    s << "    let envColor = textureSample(t_envMap, s_envMap, R).rgb;\n";
                } else {
                    s << "    let envPhi   = atan2(R.z, R.x) * 0.15915494 + 0.5;\n";  // three.js convention: atan2(z, x)
                    s << "    let envTheta = asin(clamp(R.y, -1.0, 1.0)) * 0.31830989 + 0.5;\n";
                    s << "    let envColor = textureSample(t_envMap, s_envMap, vec2<f32>(envPhi, envTheta)).rgb;\n";
                }
                s << "    baseColor = baseColor * diffuseLight + specularLight + envColor * material.emissive.w;\n";
            } else {
                s << "    baseColor = baseColor * diffuseLight + specularLight;\n";
            }
        }

        // LightMap
        if (features & ShaderFeatures::LightMap) {
            s << "    let _lightSrc = vec3<f32>(select(in.uv, in.uv2, material.lightMapT1.w > 0.5), 1.0);\n";
            s << "    let _lightUv = vec2<f32>(dot(material.lightMapT0.xyz, _lightSrc), dot(material.lightMapT1.xyz, _lightSrc));\n";
            s << "    baseColor = baseColor + textureSample(t_lightMap, s_lightMap, _lightUv).rgb;\n";
        }

        // Transmission: sample the opaque framebuffer copy through refraction
        if (features & ShaderFeatures::Transmission) {
            s << R"(    {
        let transmission = material.transmissionParams.x;
        let ior = material.transmissionParams.y;
        let thicknessFactor = material.transmissionParams.z;
        let samplerW = material.transmissionParams.w;
        let attColor = material.attenuationParams.xyz;
        let attDist = material.attenuationParams.w;

        // Refraction ray through the volume (thickness in local space, scaled by model)
        let refrDir = refract(-V, N, 1.0 / ior);
        let modelScale = vec3<f32>(
            length(transform.model[0].xyz),
            length(transform.model[1].xyz),
            length(transform.model[2].xyz));
        let transmissionRay = normalize(refrDir) * thicknessFactor * modelScale;
        let refractedPos = in.worldPos + transmissionRay;

        // Project to screen space
        let clipPos = transform.proj * transform.view * vec4<f32>(refractedPos, 1.0);
        let refrNDC = clipPos.xy / clipPos.w * 0.5 + 0.5;
        let refrUV = vec2<f32>(refrNDC.x, 1.0 - refrNDC.y);
        let framebufferLod = log2(samplerW) * roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
        var transmittedLight = textureSampleLevel(t_transmissionMap, s_transmissionMap, refrUV, framebufferLod).rgb;

        // Beer's law volume attenuation
        if (attDist > 0.0) {
            let attCoeff = -log(max(attColor, vec3<f32>(1e-6))) / attDist;
            transmittedLight = transmittedLight * exp(-attCoeff * length(transmissionRay));
        }

        // Fresnel
        let NdotV_t = max(dot(N, V), 0.0);
        let f0_t = pow((ior - 1.0) / (ior + 1.0), 2.0);
        let fresnel_t = f0_t + (1.0 - f0_t) * pow(1.0 - NdotV_t, 5.0);
        let transmissionFactor = transmission * (1.0 - fresnel_t);
        baseColor = totalSpecular + mix(totalDiffuse, transmittedLight * albedoTint, transmissionFactor);
    }
)";
        }

        // AO map — applied before emissive so emissive is not darkened by occlusion
        if (features & ShaderFeatures::AOMap) {
            s << "    {\n";
            s << "        let _aoSrc = vec3<f32>(select(in.uv, in.uv2, material.aoMapT1.w > 0.5), 1.0);\n";
            s << "        let _aoUv = vec2<f32>(dot(material.aoMapT0.xyz, _aoSrc), dot(material.aoMapT1.xyz, _aoSrc));\n";
            s << "        let ao = textureSample(t_aoMap, s_aoMap, _aoUv).r;\n";
            s << "        let aoIntensity = material.flags.x;\n";
            s << "        baseColor = baseColor * mix(1.0, ao, aoIntensity);\n";
            s << "    }\n";
        }

        // Add emissive after AO (matches three.js outgoingLight composition)
        s << "    baseColor = baseColor + emissiveColor;\n";
    } else {
        // Unlit path
        if (features & ShaderFeatures::EmissiveMap) {
            s << "    let _emissiveSrcU = vec3<f32>(select(in.uv, in.uv2, material.emissiveMapT1.w > 0.5), 1.0);\n";
            s << "    let _emissiveUvU = vec2<f32>(dot(material.emissiveMapT0.xyz, _emissiveSrcU), dot(material.emissiveMapT1.xyz, _emissiveSrcU));\n";
            s << "    baseColor = baseColor + material.emissive.rgb * textureSample(t_emissiveMap, s_emissiveMap, _emissiveUvU).rgb;\n";
        }
    }

    // Per-object tone mapping — used when rendering to a render target
    // (where the post-process blit doesn't apply). When rendering to the
    // surface, these bits are zero and the renderer-level post-process handles it.
    if ((features & ShaderFeatures::TonemapMask) != ShaderFeatures::TonemapNone) {
        s << "    let exposure = material.fogParams.w;\n";
        s << "    baseColor = baseColor * exposure;\n";
        if ((features & ShaderFeatures::TonemapMask) == ShaderFeatures::TonemapReinhard) {
            s << "    baseColor = baseColor / (vec3<f32>(1.0) + baseColor);\n";
        } else if ((features & ShaderFeatures::TonemapMask) == ShaderFeatures::TonemapCineon) {
            s << "    let x = max(vec3<f32>(0.0), baseColor - vec3<f32>(0.004));\n";
            s << "    baseColor = (x * (6.2 * x + vec3<f32>(0.5))) / (x * (6.2 * x + vec3<f32>(1.7)) + vec3<f32>(0.06));\n";
        } else if ((features & ShaderFeatures::TonemapMask) == ShaderFeatures::TonemapACES) {
            s << "    let a = baseColor * (baseColor * 2.51 + vec3<f32>(0.03));\n";
            s << "    let b = baseColor * (baseColor * 2.43 + vec3<f32>(0.59)) + vec3<f32>(0.14);\n";
            s << "    baseColor = clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));\n";
        }
        // TonemapLinear: just exposure multiplication (done above)
    }

    // Fog
    if (features & ShaderFeatures::FogLinear) {
        s << "    {\n";
        s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
        s << "        let fogFactor = clamp((material.fogParams.y - fogDist) / (material.fogParams.y - material.fogParams.x), 0.0, 1.0);\n";
        s << "        baseColor = mix(material.fogColor.rgb, baseColor, fogFactor);\n";
        s << "    }\n";
    } else if (features & ShaderFeatures::FogExp2) {
        s << "    {\n";
        s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
        s << "        let fogDensity = material.fogParams.z;\n";
        s << "        let fogFactor = exp(-fogDensity * fogDensity * fogDist * fogDist);\n";
        s << "        baseColor = mix(material.fogColor.rgb, baseColor, clamp(fogFactor, 0.0, 1.0));\n";
        s << "    }\n";
    }

    // Per-object sRGB output — used when rendering to a render target.
    // Use the proper piecewise sRGB OETF (matches GL's LinearTosRGB) — gamma 2.2 is
    // brighter in midtones and would diverge from GL.
    if (features & ShaderFeatures::SRGBOutput) {
        s << "    {\n"
          << "        let bcClamped = max(baseColor, vec3<f32>(0.0));\n"
          << "        let bcLo = bcClamped * 12.92;\n"
          << "        let bcHi = pow(bcClamped, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);\n"
          << "        baseColor = select(bcHi, bcLo, bcClamped <= vec3<f32>(0.0031308));\n"
          << "    }\n";
    }

    s << "    return vec4<f32>(baseColor, opacity);\n}\n";
    return s.str();
}

std::string threepp::wgpu::buildDepthWGSL() {
    return R"(struct DepthUniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: DepthUniforms;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec3<f32> };
@vertex fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}
@fragment fn fs_main() {}
)";
}

std::string threepp::wgpu::buildSkinnedDepthWGSL() {
    return R"(
struct DepthUniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: DepthUniforms;

struct SkinData {
    bindMatrix: mat4x4<f32>,
    bindMatrixInverse: mat4x4<f32>,
    boneCount: u32, _pad0: u32, _pad1: u32, _pad2: u32,
    bones: array<mat4x4<f32>>,
};
@group(0) @binding(1) var<storage, read> skin: SkinData;

struct SkinVertex { index: vec4<f32>, weight: vec4<f32> };
@group(0) @binding(2) var<storage, read> skinVertices: array<SkinVertex>;

struct VI { @location(0) position: vec3<f32> };

@vertex fn vs_main(in: VI, @builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let sv = skinVertices[vid];
    var sp = vec4<f32>(0.0);
    let bp = skin.bindMatrix * vec4<f32>(in.position, 1.0);
    for (var i = 0u; i < 4u; i++) {
        let w = sv.weight[i];
        if (w > 0.0) {
            sp += (skin.bones[u32(sv.index[i])] * bp) * w;
        }
    }
    let finalPos = (skin.bindMatrixInverse * sp).xyz;
    return u.mvp * vec4<f32>(finalPos, 1.0);
}
@fragment fn fs_main() {}
)";
}

std::string threepp::wgpu::buildMorphDepthWGSL() {
    return R"(
struct DepthUniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: DepthUniforms;

struct MorphData {
    numTargets: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    influences: array<vec4<f32>, 2>,
    positions: array<vec4<f32>>,
};
@group(0) @binding(1) var<storage, read> morph: MorphData;

struct VI { @location(0) position: vec3<f32> };

@vertex fn vs_main(in: VI, @builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = in.position;
    let numT = morph.numTargets;
    let totalVerts = arrayLength(&morph.positions) / max(numT, 1u);
    for (var t = 0u; t < numT; t++) {
        let inf = morph.influences[t / 4u][t % 4u];
        if (inf > 0.0) {
            let mp = morph.positions[t * totalVerts + vid];
            pos = pos + (mp.xyz - in.position) * inf;
        }
    }
    return u.mvp * vec4<f32>(pos, 1.0);
}
@fragment fn fs_main() {}
)";
}
