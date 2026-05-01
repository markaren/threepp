#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Phase 9: this shader fills a struct payload (radiance leaving the hit +
// sampled bounce direction + throughput multiplier). raygen handles the
// path loop; we just shade the hit and pick where the path goes next.
//
// Direct lighting: Cook-Torrance GGX for specular, energy-conserving
// Lambert for diffuse, gated by per-light shadow rays. Specular env IBL
// is sampled directly from the equirect at the reflection direction
// (matches WGPU PT and standard raster IBL).
//
// Indirect bounce: cosine-weighted hemisphere sample. BRDF·cos/pdf for
// Lambert collapses to `albedo·(1-metalness)`, so metals get throughput=0
// and naturally do not contribute to diffuse GI. Specular indirect
// (mirror reflections of nearby geometry) is a follow-up — for now metals
// reflect the env map only.

struct Payload {
    vec3 radiance;
    vec3 brdfWeight;
    vec3 nextOrigin;
    vec3 nextDir;
    uint flags;
    uint seed;
};

layout(buffer_reference, scalar) readonly buffer VertexBuf { float p[]; };
layout(buffer_reference, scalar) readonly buffer NormalBuf  { float n[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf   { uint  i[]; };
layout(buffer_reference, scalar) readonly buffer UvBuf      { float u[]; };

struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;// 0 == no UV attribute
    uint     indexed;
    uint     _pad;
};

struct MaterialDesc {
    vec3  albedo;
    float roughness;
    float metalness;
    vec3  emissive;
    float emissiveIntensity;
    int   albedoTexIndex;   // -1 == no albedo texture
    int   roughnessTexIndex;// -1 == no roughness texture; sampled .g
    int   metalnessTexIndex;// -1 == no metalness texture; sampled .b
    int   normalTexIndex;   // -1 == no normal map; tangent-space (glTF)
    vec2  normalScale;      // (sx, sy) applied to sampled XY before re-deriving Z
    float alphaCutoff;      // <= 0 == disabled (used by the any-hit shader)
    float transmission;     // 0..1 probability the bounce takes the transmission lobe
    float ior;              // index of refraction (only consulted when transmission > 0)
    int   transmissionTexIndex;// -1 == no transmission map; sampled .r (glTF KHR_materials_transmission)
    float clearcoat;        // 0..1 layer intensity; ccF0 fixed at 0.04 (dielectric IOR ~1.5)
    float clearcoatRoughness;// 0..1; clamped in shader to keep VNDF non-singular
    int   clearcoatTexIndex;          // -1 == none; sampled .r (glTF KHR_materials_clearcoat)
    int   clearcoatRoughnessTexIndex; // -1 == none; sampled .g
    vec3  attenuationColor;   // Beer-Lambert tint per attenuationDistance (default white)
    float attenuationDistance;// 0 = no Beer-Lambert; world-space mean-free path
    int   emissiveTexIndex;   // -1 == no emissive map; sampled .rgb (sRGB decode via format)
    float specularIntensity;  // KHR_materials_specular: scales dielectric F0 (default 1)
    vec3  specularColor;      // KHR_materials_specular: tints dielectric F0 (default white)
    vec3  sheenColor;         // KHR_materials_sheen: Charlie lobe color (default black = off)
    float sheenRoughness;     // KHR_materials_sheen: Charlie roughness (default 0)
};

const uint kMaxMaterialTextures = 256;

struct DirLight {
    vec3 direction;
    vec3 color;
};
struct PointLight {
    vec3  position;
    float range;  // 0 = infinite
    vec3  color;
    float decay;
};
struct SpotLight {
    vec3  position;
    float range;
    vec3  color;
    float decay;
    vec3  direction;      // toward target (emission direction)
    float cosAngleOuter;  // cos(angle)          — hard cutoff
    float cosAngleInner;  // cos(angle*(1-pen))  — full-brightness edge
};
struct RectLight {
    vec3 position;
    vec3 halfU;   // world right  * width/2
    vec3 halfV;   // world up     * height/2
    vec3 normal;  // emission direction into scene
    vec3 color;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topAS;
layout(set = 0, binding = 3, scalar) readonly buffer GeomDescBuf {
    GeometryDesc geoms[];
};
layout(set = 0, binding = 4, scalar) readonly buffer MatDescBuf {
    MaterialDesc mats[];
};
layout(set = 0, binding = 5, scalar) uniform LightsUbo {
    vec3       ambient;
    uint       dirCount;
    DirLight   dirLights[8];
    uint       pointCount;
    uint       spotCount;
    uint       rectCount;
    PointLight pointLights[8];
    SpotLight  spotLights[8];
    RectLight  rectLights[4];
} lights;
layout(set = 0, binding = 6) uniform sampler2D envTex;
// Bindless material albedo array. Indexed by mdesc.albedoTexIndex; slot 0 is
// the host's 1×1 white default, used implicitly via -1→0 fallback below.
layout(set = 0, binding = 8) uniform sampler2D albedoMaps[kMaxMaterialTextures];

// Phase 11: PMREM mip count comes via the same push-constant block used by
// raygen. .x is raygen's sampleIndex (not read here); .y is envMipCount.
layout(push_constant) uniform Pc {
    uint sampleIndex;
    uint envMipCount;
    uint _pad1;
    uint _pad2;
} pc;

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT Payload payload;
// Shadow visibility: 0 = occluded (default), 1 = clear (set by shadow_miss).
layout(location = 1) rayPayloadEXT float shadowVisibility;

const float PI = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

// GL / three.js equirect convention (matches miss.rmiss + WGPU PT).
vec3 sampleEquirect(vec3 dir) {
    const float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return texture(envTex, vec2(u, v)).rgb;
}

float distGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geomSmithG1(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// KHR_materials_sheen: Charlie NDF, Neubelt visibility, and IBL energy approximation.
// Matches three.js physical_pars_fragment.glsl.
float D_Charlie(float NdotH, float roughness) {
    float invAlpha = 1.0 / max(roughness * roughness, 1e-4);
    float sin2h = max(1.0 - NdotH * NdotH, 1e-7);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}
float V_Neubelt(float NdotV, float NdotL) {
    return clamp(1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
}
float IBLSheenBRDF(float dotNV, float roughness) {
    float r2 = roughness * roughness;
    float a = roughness < 0.25 ? -339.2*r2 + 161.4*roughness - 25.9
                                : -8.48*r2  + 14.3*roughness  - 9.95;
    float b = roughness < 0.25 ?  44.0*r2  - 23.7*roughness  +  3.26
                                :  1.97*r2  -  3.27*roughness  +  0.72;
    float DG = exp(a * dotNV + b) + (roughness < 0.25 ? 0.0 : 0.1*(roughness - 0.25));
    return clamp(DG / PI, 0.0, 1.0);
}

// PCG XSH-RR. Cheap, well-mixed, single-uint state — perfect for shaders.
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float urand(inout uint state) {
    return float(pcgNext(state)) / 4294967296.0;
}

// Tangent-space cosine-weighted hemisphere sample (Z up).
vec3 cosineHemisphere(vec2 u) {
    const float r = sqrt(u.x);
    const float phi = TWO_PI * u.y;
    return vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// Build an arbitrary orthonormal basis around N.
mat3 makeTBN(vec3 N) {
    const vec3 up = abs(N.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3 T = normalize(cross(up, N));
    const vec3 B = cross(N, T);
    return mat3(T, B, N);
}

// Analytic Smith G1 for GGX (alpha = roughness²). Matches the form used
// by VNDF importance sampling — must NOT be swapped with the Schlick-Smith
// `geomSmithG1` above, which is a different approximation.
float smithG1(float NdotX, float alpha) {
    const float a2 = alpha * alpha;
    return 2.0 * NdotX /
           max(NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX), 1e-6);
}

// GGX VNDF microfacet normal (half-vector) in world space.
// Dupuy & Benyoub 2023 spherical-cap parameterization — guaranteed
// above-horizon result, no path deaths. Same Dv distribution as
// Heitz 2018 so PDF expressions are unchanged.
vec3 sampleVNDF_H(vec3 wo, vec3 n, float alpha, vec2 u) {
    const vec3 nt = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0)
                                     : vec3(1.0, 0.0, 0.0);
    const vec3 t1 = normalize(cross(nt, n));
    const vec3 t2 = cross(n, t1);

    // Stretch wo into the isotropic configuration.
    const vec3 woLocal = vec3(dot(wo, t1), dot(wo, t2), dot(wo, n));
    const vec3 wiStr   = normalize(vec3(alpha * woLocal.x,
                                        alpha * woLocal.y,
                                        woLocal.z));

    // Sample a point on the spherical cap above wiStr.z.
    // z ∈ (-wiStr.z, 1] guarantees the half-vector is above horizon.
    const float phi      = TWO_PI * u.x;
    const float z        = (1.0 - u.y) * (1.0 + wiStr.z) - wiStr.z;
    const float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    const vec3  c        = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);

    // Half-vector: sum cap sample + stretched wo, then unstretch.
    const vec3 h      = c + wiStr;
    const vec3 hLocal = normalize(vec3(alpha * h.x, alpha * h.y, h.z));
    return hLocal.x * t1 + hLocal.y * t2 + hLocal.z * n;
}

vec3 sampleVNDF(vec3 wo, vec3 n, float alpha, vec2 u) {
    return reflect(-wo, sampleVNDF_H(wo, n, alpha, u));
}

void main() {
    const float w = 1.0 - attribs.x - attribs.y;

    const GeometryDesc gdesc = geoms[gl_InstanceCustomIndexEXT];
    const MaterialDesc mdesc = mats [gl_InstanceCustomIndexEXT];

    uvec3 idx;
    if (gdesc.indexed != 0u) {
        IndexBuf ib = IndexBuf(gdesc.indexAddress);
        idx = uvec3(ib.i[gl_PrimitiveID * 3 + 0],
                    ib.i[gl_PrimitiveID * 3 + 1],
                    ib.i[gl_PrimitiveID * 3 + 2]);
    } else {
        idx = uvec3(gl_PrimitiveID * 3 + 0,
                    gl_PrimitiveID * 3 + 1,
                    gl_PrimitiveID * 3 + 2);
    }

    NormalBuf nb = NormalBuf(gdesc.normalAddress);
    const vec3 n0 = vec3(nb.n[idx.x * 3 + 0], nb.n[idx.x * 3 + 1], nb.n[idx.x * 3 + 2]);
    const vec3 n1 = vec3(nb.n[idx.y * 3 + 0], nb.n[idx.y * 3 + 1], nb.n[idx.y * 3 + 2]);
    const vec3 n2 = vec3(nb.n[idx.z * 3 + 0], nb.n[idx.z * 3 + 1], nb.n[idx.z * 3 + 2]);

    const vec3 nObj = normalize(w * n0 + attribs.x * n1 + attribs.y * n2);
    const vec3 Nworld = normalize(transpose(mat3(gl_WorldToObjectEXT)) * nObj);

    const vec3 V = normalize(-gl_WorldRayDirectionEXT);
    // Front-face = ray came from outside the surface (V on the same side as the
    // outward normal). Captured before flipping so the transmission lobe below
    // can pick the correct refraction η and origin offset on back-facing hits.
    const bool isFront = dot(Nworld, V) >= 0.0;
    vec3 N = isFront ? Nworld : -Nworld;// double-sided shading for thin meshes / cull-disabled

    // UV interpolation (only when the geometry has a uv attribute). The
    // outer fallback is vec2(0) — harmless for materials without an albedo
    // texture; the slot-0 white default would just sample (0,0) anyway.
    vec2 uv = vec2(0.0);
    if (gdesc.uvAddress != 0ul) {
        UvBuf ub = UvBuf(gdesc.uvAddress);
        const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
        const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
        const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
        uv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;

        // Tangent-space normal map (glTF convention). The TBN frame is
        // derived per-pixel from triangle position + UV deltas, so we don't
        // have to upload a tangent attribute. Skipped when the UV layout
        // is degenerate (det ≈ 0) or the tangent collapses after Gram-
        // Schmidt against N. Sampled RGB is unpacked from [0,1]→[-1,1];
        // xy are scaled by mdesc.normalScale and z is re-derived to keep
        // the perturbed normal a unit vector. Mirrored UV charts (det < 0)
        // would ideally flip B's sign — not handled here; revisit if a
        // specific asset shows inverted bumps. albedoMaps is the generic
        // bindless pool (the name is slightly stale).
        if (mdesc.normalTexIndex >= 0 && gdesc.vertexAddress != 0ul) {
            VertexBuf vb = VertexBuf(gdesc.vertexAddress);
            const vec3 p0 = vec3(vb.p[idx.x * 3 + 0], vb.p[idx.x * 3 + 1], vb.p[idx.x * 3 + 2]);
            const vec3 p1 = vec3(vb.p[idx.y * 3 + 0], vb.p[idx.y * 3 + 1], vb.p[idx.y * 3 + 2]);
            const vec3 p2 = vec3(vb.p[idx.z * 3 + 0], vb.p[idx.z * 3 + 1], vb.p[idx.z * 3 + 2]);
            const vec3 e1 = p1 - p0;
            const vec3 e2 = p2 - p0;
            const vec2 duv1 = uv1 - uv0;
            const vec2 duv2 = uv2 - uv0;
            const float det = duv1.x * duv2.y - duv2.x * duv1.y;
            if (abs(det) > 1e-8) {
                const float inv = 1.0 / det;
                const vec3 Tobj = inv * (e1 * duv2.y - e2 * duv1.y);
                vec3 Tworld = mat3(gl_ObjectToWorldEXT) * Tobj;
                Tworld = Tworld - dot(Tworld, N) * N;// Gram-Schmidt vs shading N
                const float Tlen = length(Tworld);
                if (Tlen > 1e-6) {
                    const vec3 T = Tworld / Tlen;
                    const vec3 B = cross(N, T);
                    const int nidx = clamp(mdesc.normalTexIndex, 0, int(kMaxMaterialTextures) - 1);
                    vec3 ns = texture(albedoMaps[nidx], uv).rgb * 2.0 - 1.0;
                    ns.xy *= mdesc.normalScale;
                    ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
                    N = normalize(T * ns.x + B * ns.y + N * ns.z);
                }
            }
        }
    }
    const float NdotV = max(dot(N, V), 0.0);

    // Albedo: scalar PBR colour modulated by the bound albedo map (sRGB
    // decode is hardware-side via the VK_FORMAT_R8G8B8A8_SRGB view).
    vec3 albedoSample = vec3(1.0);
    if (mdesc.albedoTexIndex >= 0) {
        const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedoSample = texture(albedoMaps[idxClamped], uv).rgb;
    }
    const vec3  albedo    = mdesc.albedo * albedoSample;

    // glTF packs roughness in .g and metalness in .b; threepp's metalnessMap /
    // roughnessMap typically point at the same packed texture, so the bindless
    // cache dedupes to a single slot. Multiplicative — matches three.js.
    float roughness = mdesc.roughness;
    float metalness = mdesc.metalness;
    if (mdesc.roughnessTexIndex >= 0) {
        const int i = clamp(mdesc.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        roughness *= texture(albedoMaps[i], uv).g;
    }
    if (mdesc.metalnessTexIndex >= 0) {
        const int i = clamp(mdesc.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        metalness *= texture(albedoMaps[i], uv).b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metalness = clamp(metalness, 0.0,  1.0);

    const vec3  F0        = mix(vec3(0.04) * mdesc.specularIntensity * mdesc.specularColor, albedo, metalness);
    const float k         = (roughness + 1.0) * (roughness + 1.0) / 8.0;

    // World-space hit point, used as origin for the shadow rays. Offset
    // along the geometric normal to avoid self-intersection (acne).
    const vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    uint seed = payload.seed;
    const float pSpec = mix(0.5, 0.98, metalness);
    const float alpha = roughness * roughness;

    // === Clearcoat layer params ===
    // ccF0 fixed at 0.04 (dielectric IOR ≈ 1.5 — three.js / WGPU PT convention).
    // ccWeight = clearcoat·F_cc(N·V) both attenuates the base layer (energy
    // conservation: base receives 1−ccWeight) and weights the clearcoat GGX
    // lobe in eval/sampling. ccProb floors the cc sampling rate at 0.15·cc so
    // face-on viewing (where ccFresnel ≈ 0.04) doesn't starve the cc lobe of
    // samples. Mutually exclusive with transmission via the early-return below.
    // Out of scope for v1: separate clearcoatNormalMap — clearcoat uses the
    // shading normal (post base normal-map), matching three.js raster.
    float ccScalar = mdesc.clearcoat;
    float ccRough  = mdesc.clearcoatRoughness;
    if (mdesc.clearcoatTexIndex >= 0) {
        const int i = clamp(mdesc.clearcoatTexIndex, 0, int(kMaxMaterialTextures) - 1);
        ccScalar *= texture(albedoMaps[i], uv).r;
    }
    if (mdesc.clearcoatRoughnessTexIndex >= 0) {
        const int i = clamp(mdesc.clearcoatRoughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        ccRough *= texture(albedoMaps[i], uv).g;
    }
    ccScalar = clamp(ccScalar, 0.0, 1.0);
    ccRough  = clamp(ccRough,  0.04, 1.0);
    const float ccF0       = 0.04;
    const float ccFresnel  = ccF0 + (1.0 - ccF0) * pow(1.0 - NdotV, 5.0);
    const float ccWeight   = ccScalar * ccFresnel;
    const float ccAlpha    = ccRough * ccRough;
    const float ccProb     = max(ccWeight, 0.15 * ccScalar);
    const float baseScale  = 1.0 - ccWeight;
    // Inverse of the base-branch probability for the stochastic 3-way split
    // below; clamped to avoid blowups when ccProb → 1.
    const float invBaseProb = 1.0 / max(1.0 - ccProb, 1e-6);

    // === Transmission lobe ===
    // Russian-roulette gate by mdesc.transmission: with probability `transmission`
    // this hit acts as a glass interface (Schlick-Fresnel weighted reflect /
    // refract) and skips direct lighting + env NEE entirely. The path continues
    // with the chosen bounce direction. Matches the WGPU PT pattern: no
    // 1/transmission inverse-prob scaling, so `transmission` doubles as a
    // stylised reflect-vs-transmit blend factor (artist control), not a physical
    // mixing weight. Out-of-scope: dispersion, Beer-Lambert volumetric attenuation.
    //
    // The outgoing payload sets bit 2 ("NEE skipped at this hit") so raygen
    // doesn't half-weight the env on the next bounce-into-miss — the MIS
    // partner (NEE shadow ray) wasn't fired here, so the env at the next miss
    // needs full weight to stay unbiased.
    //
    // glTF KHR_materials_transmission: scalar `transmission` is multiplied by
    // the transmissionMap red channel (linear texture, no sRGB decode).
    float transmission = mdesc.transmission;
    if (mdesc.transmissionTexIndex >= 0) {
        const int i = clamp(mdesc.transmissionTexIndex, 0, int(kMaxMaterialTextures) - 1);
        transmission *= texture(albedoMaps[i], uv).r;
    }
    if (transmission > 0.0 && urand(seed) < transmission) {
        const float ior = max(mdesc.ior, 1.0);
        const float eta = isFront ? (1.0 / ior) : ior;
        const vec3  I    = gl_WorldRayDirectionEXT;

        // Sample a GGX microfacet normal so roughness scatters the refracted
        // ray. α=0 (smooth glass) degenerates to H=N (mirror). Fresnel and
        // reflect/refract all use H rather than N.
        const vec2  u2   = vec2(urand(seed), urand(seed));
        const vec3  H    = sampleVNDF_H(V, N, alpha, u2);
        const float cosH = max(dot(V, H), 0.0);

        // Schlick Fresnel at the microfacet half-vector. On exit use the
        // transmitted-side cosine so TIR raises F→1 smoothly.
        const float r0    = pow((1.0 - ior) / (1.0 + ior), 2.0);
        const float sin2H = max(0.0, 1.0 - cosH * cosH);
        const float cosSchlick = isFront
                ? cosH
                : sqrt(max(0.0, 1.0 - ior * ior * sin2H));
        const float F = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);

        vec3 wDir;
        vec3 wOrigin;
        vec3 tWeight = vec3(1.0);
        if (urand(seed) < F) {
            wDir    = reflect(I, H);
            wOrigin = hitPos + N * 1e-3;
        } else {
            const vec3 refr = refract(I, H, eta);
            if (dot(refr, refr) < 1e-6) {
                // Total internal reflection — fall back to mirror reflect.
                wDir    = reflect(I, H);
                wOrigin = hitPos + N * 1e-3;
            } else {
                wDir    = normalize(refr);
                wOrigin = hitPos - N * 1e-3;
                // Base color tints the transmitted light. G1 masking for the
                // transmitted direction (matches WGPU PT: albedo * ggxG1(cosOut)).
                const float cosOut = abs(dot(wDir, H));
                const float G1out  = smithG1(cosOut, alpha);
                vec3 glassTint = albedo * G1out;
                if (mdesc.attenuationDistance > 0.0) {
                    glassTint *= pow(max(mdesc.attenuationColor, vec3(1e-6)),
                                     vec3(gl_HitTEXT / mdesc.attenuationDistance));
                }
                tWeight = glassTint / (eta * eta);
            }
        }

        vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
        if (mdesc.emissiveTexIndex >= 0) {
            const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
            emissiveOut *= texture(albedoMaps[ei], uv).rgb;
        }
        const float emLum0 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
        if (emLum0 > 20.0) emissiveOut *= 20.0 / emLum0;
        payload.radiance   = emissiveOut;
        payload.brdfWeight = tWeight;
        payload.nextOrigin = wOrigin;
        payload.nextDir    = wDir;
        payload.flags      = 4u;// bit 2: NEE skipped — raygen must not half-weight the next env miss
        payload.seed       = seed;
        return;
    }

    // Sum direct contribution from each scene-driven directional light.
    // Physical lights (three.js useLegacyLights = false): light.color
    // already encodes intensity, no 1/PI cancel. Each light is gated by a
    // shadow ray; closest_hit is skipped on those rays so only the shadow
    // miss handler matters. Linear HDR radiance is returned to raygen,
    // which handles the sRGB encode.
    // KHR_materials_sheen energy conservation: sheen absorbs some energy from the
    // base BRDF. Scale factor is 1 - max(sheenColor) * IBL_sheen(NdotV, roughness).
    const bool hasSheen = mdesc.sheenRoughness > 0.0 &&
                          any(greaterThan(mdesc.sheenColor, vec3(0.0)));
    float sheenScaling = 1.0;
    if (hasSheen) {
        const float sheenMax = max(mdesc.sheenColor.r, max(mdesc.sheenColor.g, mdesc.sheenColor.b));
        sheenScaling = 1.0 - sheenMax * IBLSheenBRDF(NdotV, mdesc.sheenRoughness);
    }

    vec3 lit = vec3(0.0);
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3 L = normalize(lights.dirLights[i].direction);
        const float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        // Shadow ray: any non-alpha-tested hit blocks the light. Without
        // OpaqueEXT the any-hit shader fires and lets cutout foliage etc.
        // pass light through their transparent texels.
        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 0, 0, 1,// missIndex = 1 (shadow miss)
                    hitPos + N * 1e-3, 0.0, L, 1e4, 1);
        if (shadowVisibility <= 0.0) continue;

        const vec3  H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);

        const vec3  F        = fresnelSchlick(VdotH, F0);
        const float D        = distGGX(NdotH, roughness);
        const float G        = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd       = (vec3(1.0) - F) * (1.0 - metalness);
        const vec3  diffuse  = kd * albedo / PI;

        // Layered material: base lobes carry (1−ccWeight) of the energy and
        // the clearcoat lobe contributes its own GGX. Single-scatter Schlick-
        // Smith G is used for the cc lobe to stay consistent with the rest of
        // the dir-light eval — the bounce/NEE paths use analytic Smith G1
        // (matching VNDF's PDF), so dir-light eval is a stylistic shortcut.
        vec3 perLight = (diffuse + specular) * baseScale * sheenScaling;
        if (hasSheen) {
            perLight += mdesc.sheenColor * D_Charlie(NdotH, mdesc.sheenRoughness)
                                        * V_Neubelt(NdotV, NdotL);
        }
        if (ccWeight > 0.0) {
            const float k_cc    = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
            const float D_cc    = distGGX(NdotH, ccRough);
            const float G_cc    = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
            const float spec_cc = (D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4);
            perLight += vec3(spec_cc * ccWeight);
        }
        lit += perLight * NdotL * lights.dirLights[i].color;
    }

    // === Point lights ===
    for (uint i = 0u; i < lights.pointCount; ++i) {
        vec3        toL  = lights.pointLights[i].position - hitPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float NdotL = max(dot(N, toL), 0.0);
        if (NdotL <= 0.0) continue;

        float atten = 1.0 / max(dist * dist, 0.01);
        const float range = lights.pointLights[i].range;
        if (range > 0.0) {
            const float t = min(dist / range, 1.0);
            atten *= (1.0 - t * t) * (1.0 - t * t);
        }

        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 0, 0, 1,
                    hitPos + N * 1e-3, 0.0, toL, dist - 1e-2, 1);
        if (shadowVisibility <= 0.0) continue;

        const vec3  H_p   = normalize(V + toL);
        const float NdotH = max(dot(N, H_p), 0.0);
        const float VdotH = max(dot(V, H_p), 0.0);
        const vec3  F_p   = fresnelSchlick(VdotH, F0);
        const float D_p   = distGGX(NdotH, roughness);
        const float G_p   = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  spec  = (D_p * G_p * F_p) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd_p  = (vec3(1.0) - F_p) * (1.0 - metalness);
        const vec3  diff  = kd_p * albedo / PI;
        vec3 perPt = (diff + spec) * baseScale * sheenScaling;
        if (hasSheen) {
            perPt += mdesc.sheenColor * D_Charlie(NdotH, mdesc.sheenRoughness)
                                      * V_Neubelt(NdotV, NdotL);
        }
        if (ccWeight > 0.0) {
            const float k_cc    = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
            const float D_cc    = distGGX(NdotH, ccRough);
            const float G_cc    = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
            perPt += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4) * ccWeight);
        }
        lit += perPt * NdotL * lights.pointLights[i].color * atten;
    }

    // === Spot lights ===
    for (uint i = 0u; i < lights.spotCount; ++i) {
        vec3        toL  = lights.spotLights[i].position - hitPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float NdotL = max(dot(N, toL), 0.0);
        if (NdotL <= 0.0) continue;

        // Cone check: dot(-toL, emissionDir) = cos of angle at light between hitPoint and target.
        const float spotCos  = dot(-toL, lights.spotLights[i].direction);
        const float spotAtten = smoothstep(lights.spotLights[i].cosAngleOuter,
                                           lights.spotLights[i].cosAngleInner,
                                           spotCos);
        if (spotAtten <= 0.0) continue;

        float atten = 1.0 / max(dist * dist, 0.01);
        const float range = lights.spotLights[i].range;
        if (range > 0.0) {
            const float t = min(dist / range, 1.0);
            atten *= (1.0 - t * t) * (1.0 - t * t);
        }
        atten *= spotAtten;

        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 0, 0, 1,
                    hitPos + N * 1e-3, 0.0, toL, dist - 1e-2, 1);
        if (shadowVisibility <= 0.0) continue;

        const vec3  H_s   = normalize(V + toL);
        const float NdotH = max(dot(N, H_s), 0.0);
        const float VdotH = max(dot(V, H_s), 0.0);
        const vec3  F_s   = fresnelSchlick(VdotH, F0);
        const float D_s   = distGGX(NdotH, roughness);
        const float G_s   = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  spec  = (D_s * G_s * F_s) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd_s  = (vec3(1.0) - F_s) * (1.0 - metalness);
        const vec3  diff  = kd_s * albedo / PI;
        vec3 perSp = (diff + spec) * baseScale * sheenScaling;
        if (hasSheen) {
            perSp += mdesc.sheenColor * D_Charlie(NdotH, mdesc.sheenRoughness)
                                      * V_Neubelt(NdotV, NdotL);
        }
        if (ccWeight > 0.0) {
            const float k_cc    = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
            const float D_cc    = distGGX(NdotH, ccRough);
            const float G_cc    = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
            perSp += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4) * ccWeight);
        }
        lit += perSp * NdotL * lights.spotLights[i].color * atten;
    }

    // === Rect area lights ===
    // One random sample per light per hit. The Monte Carlo estimator divides
    // out the pdf (1/area) and multiplies the geometric coupling term
    // (cosEmitter/dist²) so this is unbiased in expectation.
    for (uint i = 0u; i < lights.rectCount; ++i) {
        // Sample uniform random point on the emitter rectangle.
        const float ru = urand(seed) * 2.0 - 1.0;
        const float rv = urand(seed) * 2.0 - 1.0;
        const vec3 samplePos = lights.rectLights[i].position
                             + ru * lights.rectLights[i].halfU
                             + rv * lights.rectLights[i].halfV;

        vec3        toL  = samplePos - hitPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float NdotL = max(dot(N, toL), 0.0);
        if (NdotL <= 0.0) continue;

        // Only the front face (normal pointing toward scene) emits.
        const float cosEmitter = max(dot(-toL, lights.rectLights[i].normal), 0.0);
        if (cosEmitter <= 0.0) continue;

        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 0, 0, 1,
                    hitPos + N * 1e-3, 0.0, toL, dist - 1e-2, 1);
        if (shadowVisibility <= 0.0) continue;

        // area = |2·halfU × 2·halfV| = 4·|halfU × halfV|
        const float area = length(cross(lights.rectLights[i].halfU,
                                        lights.rectLights[i].halfV)) * 4.0;
        const float geomTerm = cosEmitter * area / (dist * dist);

        const vec3  H_r   = normalize(V + toL);
        const float NdotH = max(dot(N, H_r), 0.0);
        const float VdotH = max(dot(V, H_r), 0.0);
        const vec3  F_r   = fresnelSchlick(VdotH, F0);
        const float D_r   = distGGX(NdotH, roughness);
        const float G_r   = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  spec  = (D_r * G_r * F_r) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd_r  = (vec3(1.0) - F_r) * (1.0 - metalness);
        const vec3  diff  = kd_r * albedo / PI;
        vec3 perRect = (diff + spec) * baseScale;
        if (ccWeight > 0.0) {
            const float k_cc    = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
            const float D_cc    = distGGX(NdotH, ccRough);
            const float G_cc    = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
            perRect += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4) * ccWeight);
        }
        lit += perRect * NdotL * lights.rectLights[i].color * geomTerm;
    }

    // === Env NEE + MIS ===
    // Visibility-tested env sample at this shade point, combined with the
    // bounce-into-miss estimator via Multiple Importance Sampling. Both
    // strategies currently use BSDF-importance sampling, so pdf_nee(x) ==
    // pdf_bsdf(x) at every direction and the balance-heuristic weights
    // collapse to a constant 0.5 / 0.5 split:
    //
    //   w_nee  = pdf_nee  / (pdf_nee + pdf_bsdf) = 0.5
    //   w_bsdf = pdf_bsdf / (pdf_nee + pdf_bsdf) = 0.5
    //
    // The NEE term gets its 0.5 multiplier here; the bounce term gets it
    // in miss.rmiss (gated on the non-primary flag). Variance vs the old
    // "suppress env on miss" pattern is halved (~√2 std-dev reduction)
    // since both estimators contribute independent samples averaged with
    // equal weight. Cost is unchanged — same NEE shadow ray, same bounce
    // ray. The constant 0.5 collapses out cleanly when an env-luminance
    // CDF NEE upgrade lands; that path will pass bsdfPdf in the payload
    // so miss can compute pdf_env_at_dir and apply pdf-based weights.
    {
        // 3-way stochastic lobe selection: clearcoat with prob ccProb, then
        // base spec / base diffuse with the existing pSpec split inside the
        // remaining (1−ccProb). The throughput multipliers absorb both the
        // branch-selection inverse and the (1−ccWeight) energy split so the
        // estimator stays unbiased and matches dir-light's analytic eval.
        const float xiCC = urand(seed);
        vec3 nDir;
        vec3 nWeight;
        bool nValid = true;
        if (ccProb > 0.0 && xiCC < ccProb) {
            const vec2 u2 = vec2(urand(seed), urand(seed));
            nDir = sampleVNDF(V, N, ccAlpha, u2);
            const float NdotL = dot(N, nDir);
            if (NdotL <= 0.0) {
                nValid = false;
            } else {
                const float G1L = smithG1(NdotL, ccAlpha);
                nWeight = vec3(ccWeight * G1L / ccProb);
            }
        } else {
            const float xiN = urand(seed);
            if (xiN < pSpec) {
                const vec2 u2 = vec2(urand(seed), urand(seed));
                nDir = sampleVNDF(V, N, alpha, u2);
                const float NdotL = dot(N, nDir);
                if (NdotL <= 0.0) {
                    nValid = false;
                } else {
                    const vec3  H_n   = normalize(V + nDir);
                    const float VdotH = max(0.0, dot(V, H_n));
                    const vec3  F_n   = fresnelSchlick(VdotH, F0);
                    const float G1L   = smithG1(NdotL, alpha);
                    nWeight = F_n * G1L * baseScale * invBaseProb / pSpec;
                }
            } else {
                const vec2 u2 = vec2(urand(seed), urand(seed));
                const vec3 localDir = cosineHemisphere(u2);
                const mat3 tbn = makeTBN(N);
                nDir = normalize(tbn * localDir);
                nWeight = albedo * (1.0 - metalness) * baseScale * invBaseProb / (1.0 - pSpec);
            }
        }
        if (nValid) {
            shadowVisibility = 0.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT,
                        0xff, 0, 0, 1,
                        hitPos + N * 1e-3, 0.0, nDir, 1e4, 1);
            if (shadowVisibility > 0.0) {
                vec3 envSample = sampleEquirect(nDir);
                const float envLum = dot(envSample, vec3(0.2126, 0.7152, 0.0722));
                if (envLum > 20.0) envSample *= 20.0 / envLum;
                lit += 0.5 * nWeight * envSample;
            }
        }
    }

    // Flat ambient irradiance — only the diffuse lobe receives it (metals
    // have no diffuse). No PI cancel under physical lights. Scaled by the
    // base-layer fraction so a 100%-clearcoat surface only shows the cc
    // contribution (which is dir-light + env NEE only — no flat ambient
    // term for clearcoat).
    const vec3 ambient = albedo * (1.0 - metalness) * lights.ambient * baseScale;

    // Phase 9 v2: probabilistic spec/diffuse lobe selection so polished
    // metals reflect nearby geometry, not just the env probe. p_spec mirrors
    // the WGPU PT (mix(0.5, 0.98, metalness)). The selected lobe's BRDF·cos
    // is divided by its sampling pdf and by p_spec / (1-p_spec) so the
    // estimator stays unbiased.
    //
    //   spec branch — VNDF half-vector → reflect → wi.
    //     BRDF·cos / vndfPdf collapses to F * G1(L);
    //     final weight = F * G1(L) / p_spec.
    //   diff branch — cosine-weighted hemisphere.
    //     BRDF·cos / cosPdf for Lambert collapses to diffuseAlbedo;
    //     final weight = diffuseAlbedo / (1 - p_spec).
    //
    // Spherical-cap VNDF (Dupuy 2023) guarantees above-horizon wi; the
    // dot(N,wi)<=0 guards below are retained for numerical edge cases only.
    // 3-way stochastic split mirrors the env-NEE sampler so MIS pairs match.
    const float xiCCB = urand(seed);
    vec3 bounceDir;
    vec3 brdfWeight;
    uint pathFlags = 0u;
    if (ccProb > 0.0 && xiCCB < ccProb) {
        const vec2 u2 = vec2(urand(seed), urand(seed));
        bounceDir = sampleVNDF(V, N, ccAlpha, u2);
        const float NdotL = dot(N, bounceDir);
        if (NdotL <= 0.0) {
            brdfWeight = vec3(0.0);
            pathFlags |= 1u;// numerical edge case — terminate path
        } else {
            const float G1L = smithG1(NdotL, ccAlpha);
            brdfWeight = vec3(ccWeight * G1L / ccProb);
        }
    } else {
        const float xi = urand(seed);
        if (xi < pSpec) {
            const vec2 u2 = vec2(urand(seed), urand(seed));
            bounceDir = sampleVNDF(V, N, alpha, u2);
            const float NdotL = dot(N, bounceDir);
            if (NdotL <= 0.0) {
                brdfWeight = vec3(0.0);
                pathFlags |= 1u;// numerical edge case — terminate path
            } else {
                const vec3  H_b   = normalize(V + bounceDir);
                const float VdotH = max(0.0, dot(V, H_b));
                const vec3  F_b   = fresnelSchlick(VdotH, F0);
                const float G1L   = smithG1(NdotL, alpha);
                brdfWeight = F_b * G1L * baseScale * invBaseProb / pSpec;
            }
        } else {
            const vec2 u2 = vec2(urand(seed), urand(seed));
            const vec3 localDir = cosineHemisphere(u2);
            const mat3 tbn = makeTBN(N);
            bounceDir = normalize(tbn * localDir);
            brdfWeight = albedo * (1.0 - metalness) * baseScale * invBaseProb / (1.0 - pSpec);
        }
    }

    vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
    if (mdesc.emissiveTexIndex >= 0) {
        const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
        emissiveOut *= texture(albedoMaps[ei], uv).rgb;
    }
    const float emLum1 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
    if (emLum1 > 20.0) emissiveOut *= 20.0 / emLum1;
    payload.radiance   = emissiveOut + ambient + lit;
    payload.brdfWeight = brdfWeight;
    payload.nextOrigin = hitPos + N * 1e-3;
    payload.nextDir    = bounceDir;
    payload.flags      = pathFlags;
    payload.seed       = seed;
}
