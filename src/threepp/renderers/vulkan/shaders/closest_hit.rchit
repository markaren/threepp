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
    vec3 hitWorldPos;  // world-space hit point (for primary-hit reprojection)
    uint hitInstanceId;// gl_InstanceCustomIndexEXT + 1; 0 == miss/sky/pass-through
    float hitRoughness;// post-clamp surface roughness; primary-hit FC cap input
    uint inFlags;      // raygen→here: bit 0 = scatter>0 (suppress emissive output;
                       // emissive NEE on the prev shade already accounted for it)
    float hitMetalness;   // raygen uses for adaptive bounce classification
    float hitTransmission;// raygen uses for adaptive bounce classification
    float bsdfPdf;        // chit→miss: pdf of the BSDF-sampled bounce direction
    float currentIor;     // medium-stack tracking; see raygen Payload for full description
};

layout(buffer_reference, scalar) readonly buffer VertexBuf { float p[]; };
layout(buffer_reference, scalar) readonly buffer NormalBuf  { float n[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf   { uint  i[]; };
layout(buffer_reference, scalar) readonly buffer UvBuf      { float u[]; };
layout(buffer_reference, scalar) readonly buffer FoamBuf    { float f[]; };

struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;// 0 == no UV attribute
    uint64_t foamAddress;// 0 == no foam attribute (per-vertex float; written by water_displace.comp for ocean meshes)
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
    int   doubleSided;        // 1 = shade both faces; 0 = pass through back-face hits
    mat3  uvTransform;              // albedo channel
    int   occlusionTexIndex;        // -1 = none; sampled .r
    mat3  uvTransformNormal;        // for normalTexIndex
    mat3  uvTransformRoughMetal;    // for roughness + metalness
    mat3  uvTransformEmissive;      // for emissiveTexIndex
    mat3  uvTransformOcclusion;     // for occlusionTexIndex
    mat3  uvTransformClearcoat;     // for clearcoatTexIndex
    mat3  uvTransformClearcoatRough;// for clearcoatRoughnessTexIndex
    mat3  uvTransformTransmission;  // for transmissionTexIndex
    float iridescence;              // KHR_materials_iridescence: 0..1 layer factor
    float iridescenceIOR;           // thin-film IOR (1.0..2.5; default 1.3)
    float iridescenceThicknessNm;   // thin-film thickness in nm (default 400)
    float dispersion;               // KHR_materials_dispersion: 0 = off; ~0.05+ visible
    float thickness;                // KHR_materials_volume: in-medium distance proxy for thin/open meshes; 0 = fall back to back-face actual ray distance
    int   thinWalled;               // 1 = treat both faces as entry (thin-shell BSDF); 0 = closed mesh
};

const uint kMaxMaterialTextures = 2048;

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
// Env luminance CDF (Phase A: importance-sample bright env features). Bound
// as a 1×1 dummy with envCdfTotalSum=0 when env is solid color or default.
layout(set = 0, binding = 18) uniform sampler2D envCdfTex;  // conditional, w×h
layout(set = 0, binding = 19) uniform sampler2D envMargTex; // marginal, 1×h

// Homogeneous fog (participating media). sigmaT.xyz = per-channel extinction
// (1/world-unit), enabled = 1.0 when scene.fog is set. Mirrors the WGPU PT
// uniform layout (WgpuPathTracerTypes.hpp: fog/fogColor vec4 pair).
layout(set = 0, binding = 17) uniform FogUbo {
    vec3  sigmaT;
    float enabled;
    vec3  color;
    float anisotropy;
} fog;

// Beer-Lambert transmittance over a finite ray segment. Distance is clamped
// to 1e6 to avoid overflow on directional / env "infinite ray" sentinels —
// callers that pass maxDist >= 1e20 should skip fog attenuation entirely
// (those sources sit "outside" the fog volume, i.e. sun through atmosphere).
bool fogEnabled() { return fog.enabled > 0.5; }
vec3 fogTransmittance(float dist) {
    const float d = clamp(dist, 0.0, 1e6);
    return exp(-fog.sigmaT * d);
}

// Bindless material albedo array. Indexed by mdesc.albedoTexIndex; slot 0 is
// the host's 1×1 white default, used implicitly via -1→0 fallback below.
layout(set = 0, binding = 8) uniform sampler2D albedoMaps[kMaxMaterialTextures];

// Emissive-mesh NEE: each emissive triangle is packed as 4 vec4
// (v0.xyz/area, v1.xyz/cumPower, v2.xyz/power, emission.rgb/_pad).
// The host walks the scene each frame, transforms triangle vertices to
// world space, computes per-triangle area + power = lum*area, and stores
// a running CDF in v1.w (cumPower). Total power lives in pc.emissiveTotalPower.
struct EmTri {
    vec4 v0;        // xyz = pos, w = area
    vec4 v1;        // xyz = pos, w = cumPower (running CDF of power)
    vec4 v2;        // xyz = pos, w = power (per-triangle power = lum * area)
    vec4 emission;  // xyz = emissive*intensity, w = unused
};
layout(set = 0, binding = 14, scalar) readonly buffer EmissiveTriBuf {
    EmTri emissiveTris[];
};

// Caustic photon map (photon_emit.rgen deposits, we gather here).
// Grid constants must match photon_emit.rgen exactly.
const uint  kPhotonGridBits = 16u;
const uint  kPhotonGridSize = 1u << kPhotonGridBits; // 65536
const uint  kPhotonsPerCell = 8u;
const float kGatherRadius   = 0.15;

layout(set = 0, binding = 15, std430) readonly buffer PhotonCountBuf { uint photonCounts[]; };
layout(set = 0, binding = 16, scalar) readonly buffer PhotonDataBuf  { vec3 photonData[];   };

// Phase 11: PMREM mip count comes via the same push-constant block used by
// raygen. .x is raygen's sampleIndex (not read here); .y is envMipCount.
layout(push_constant) uniform Pc {
    uint sampleIndex;
    uint envMipCount;
    uint _pad1;
    uint _pad2;
    uint motionFlags;       // bit 2 = scene has any glass material (gates caustic gather)
    uint emissiveCount;     // # of EmTri entries
    float emissiveTotalPower;// total CDF power (last entry's cumPower)
    uint _padSpp;           // raygen spp (unused here)
    uint envCdfWidth;       // env CDF dimensions (envCdfTotalSum > 0 to enable)
    uint envCdfHeight;
    float envCdfTotalSum;   // pdf normaliser; 0 disables env importance sampling
    float fireflyClamp;     // per-NEE luminance cap; 1e30 disables (gates never fire)
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

vec2 dirToEquirectUV(vec3 dir) {
    return vec2(0.5 + atan(dir.z, dir.x) / TWO_PI,
                0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI);
}

vec3 uvToEquirectDir(vec2 uv) {
    const float phi   = (uv.x - 0.5) * TWO_PI;
    const float theta = (uv.y - 0.5) * PI;
    const float ct    = cos(theta);
    return vec3(ct * cos(phi), sin(theta), ct * sin(phi));
}

// FNV-1a hash on a 3-D cell coordinate — must match photon_emit.rgen.
uint photonCellHash(ivec3 c) {
    uint h = 2166136261u;
    h = (h ^ uint(c.x)) * 16777619u;
    h = (h ^ uint(c.y)) * 16777619u;
    h = (h ^ uint(c.z)) * 16777619u;
    return h & (kPhotonGridSize - 1u);
}

// Gather caustic photons within kGatherRadius of pos. Returns radiance from
// Lambertian evaluation of the photon flux: flux * (albedo/π) * |N·(-dir)| / (π*r²).
// Only called for diffuse-ish (roughness > 0.1) surfaces.
vec3 gatherCaustics(vec3 pos, vec3 N, vec3 albedo, float roughness) {
    if (roughness < 0.1) return vec3(0.0);
    vec3 caustic = vec3(0.0);
    const float r2       = kGatherRadius * kGatherRadius;
    const float invDisk  = 1.0 / (PI * r2);
    const ivec3 center   = ivec3(floor(pos / kGatherRadius));
    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++) {
        const uint h     = photonCellHash(center + ivec3(dx, dy, dz));
        const uint total = photonCounts[h];
        const uint n     = min(total, kPhotonsPerCell);
        // Cells that received more photons than kPhotonsPerCell only stored a
        // subset; scale up so energy is conserved despite the cap.
        const float overflow = (n > 0u) ? float(total) / float(n) : 0.0;
        for (uint s = 0u; s < n; s++) {
            const uint base  = (h * kPhotonsPerCell + s) * 3u;
            const vec3 ppos  = photonData[base + 0u];
            const vec3 pflux = photonData[base + 1u];
            const vec3 pdir  = photonData[base + 2u]; // travel direction toward surface
            if (dot(ppos - pos, ppos - pos) < r2) {
                caustic += pflux * (albedo / PI) * max(dot(N, -pdir), 0.0) * invDisk * overflow;
            }
        }
    }
    return caustic;
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

// ── Kulla-Conty multi-scattering compensation (analytic fit) ──────────────
// WGPU PT precomputes a 32x32 GGX directional-albedo LUT and samples it on
// device. Vulkan currently uses a Karis-style closed-form fit (~5% accuracy)
// to avoid the texture binding scaffolding; promote to a real LUT later if
// the white-furnace error budget tightens. F0=1 conductor matches WGPU's LUT.
float ggxEApprox(float NdotV, float alpha) {
    const float a2 = alpha * alpha;
    return 1.0 - 0.5 / (1.0 + 7.6 * a2 * max(NdotV, 1e-3));
}

// Additive diffuse multi-scattering compensation (Kulla & Conty 2017).
// Restores the energy the spec layer absorbs internally and re-emits to the
// diffuse substrate. Adds ~8-15% energy back on rough dielectrics; matches
// WGPU evalBrdf's kcDiff term (WgpuPathTracerShaders_Rt.cpp:696-700).
vec3 kcDiff(vec3 albedo, float metalness, vec3 F0, float NdotV, float alpha) {
    const float E_v   = ggxEApprox(NdotV, alpha);
    const vec3  F_avg = (20.0 * F0 + vec3(1.0)) / 21.0;
    return albedo * (1.0 - metalness) * F_avg * max(0.0, 1.0 - E_v) / PI;
}

// Throughput boost for diffuse-sampled bounces. The MC sampler picks
// cosine-weighted hemisphere → BRDF·cos/pdf collapses to albedo·(1-metal);
// the kcBoost factor (1 + F_avg·(1-E)) absorbs the ms-comp energy so the
// estimator stays unbiased. Matches WGPU bounce sites (lines 3402-3407 +
// 4717-4727).
vec3 kcBoost(vec3 F0, float NdotV, float alpha) {
    const float E_kc  = ggxEApprox(NdotV, alpha);
    const vec3  F_avg = (20.0 * F0 + vec3(1.0)) / 21.0;
    return vec3(1.0) + F_avg * max(0.0, 1.0 - E_kc);
}

// ── Thin-film iridescence (Belcour & Barla 2017, glTF reference) ────────────
// Modulates dielectric F0 with wavelength-dependent thin-film interference.
// Inputs:
//   outsideIOR     : IOR of the medium above the film (air ≈ 1.0)
//   eta2           : IOR of the thin film itself (mdesc.iridescenceIOR)
//   cosTheta1      : cosine of the angle of incidence at the film top
//   thicknessNm    : film thickness in nanometers (mdesc.iridescenceThicknessNm)
//   baseF0         : F0 of the substrate underneath the film
// Output: spectrally-integrated F0 (RGB) accounting for the thin-film phase
// shifts. Mix with `baseF0` by mdesc.iridescence at the call site.
//
// Implementation matches three.js / Khronos glTF Sample Viewer line-for-line.
vec3 iridFresnel0ToIor(vec3 F0) {
    vec3 sqrtF0 = sqrt(F0);
    return (vec3(1.0) + sqrtF0) / (vec3(1.0) - sqrtF0);
}
vec3 iridIorToFresnel0_v(vec3 transmittedIor, float incidentIor) {
    return pow((transmittedIor - vec3(incidentIor)) / (transmittedIor + vec3(incidentIor)), vec3(2.0));
}
float iridIorToFresnel0_s(float transmittedIor, float incidentIor) {
    return pow((transmittedIor - incidentIor) / (transmittedIor + incidentIor), 2.0);
}
// Spectral-to-RGB sensitivity from Belcour 2017 supplemental. Evaluated once
// per Fourier order; `shift` is the per-RGB phase (vec3) since phi23 is
// substrate-IOR dependent and varies between channels.
vec3 iridSensitivity(float OPD, vec3 shift) {
    float phase = 2.0 * PI * OPD * 1.0e-9;
    vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    vec3 vr  = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    vec3 xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-(phase*phase) * vr);
    xyz.x   += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * (phase*phase));
    xyz     /= 1.0685e-7;
    // XYZ → linear sRGB (D65)
    mat3 XYZ_TO_REC709 = mat3(
         3.2404542, -0.9692660,  0.0556434,
        -1.5371385,  1.8760108, -0.2040259,
        -0.4985314,  0.0415560,  1.0572252);
    return XYZ_TO_REC709 * xyz;
}
vec3 evalIridescence(float outsideIOR, float eta2, float cosTheta1,
                     float thinFilmThickness, vec3 baseF0) {
    // Force iridescenceIOR -> outsideIOR when thinFilmThickness == 0 → reduces to base F0.
    float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    // Snell to the angle inside the film.
    float sinTheta2Sq = (outsideIOR / iridescenceIor) * (outsideIOR / iridescenceIor) *
                       (1.0 - cosTheta1 * cosTheta1);
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) return vec3(1.0);// total internal reflection at the film top
    float cosTheta2 = sqrt(cosTheta2Sq);
    // First interface (above the film).
    float R0   = iridIorToFresnel0_s(iridescenceIor, outsideIOR);
    float R12  = R0 + (1.0 - R0) * pow(1.0 - cosTheta1, 5.0);
    float T121 = 1.0 - R12;
    float phi12 = 0.0;
    if (iridescenceIor < outsideIOR) phi12 = PI;
    float phi21 = PI - phi12;
    // Second interface (substrate). Recover the substrate's IOR from baseF0.
    vec3 baseIOR = iridFresnel0ToIor(clamp(baseF0, vec3(0.0), vec3(0.9999)));
    vec3 R1      = iridIorToFresnel0_v(baseIOR, iridescenceIor);
    vec3 R23     = R1 + (vec3(1.0) - R1) * pow(1.0 - cosTheta2, 5.0);
    vec3 phi23   = vec3(0.0);
    if (baseIOR.x < iridescenceIor) phi23.x = PI;
    if (baseIOR.y < iridescenceIor) phi23.y = PI;
    if (baseIOR.z < iridescenceIor) phi23.z = PI;
    // Optical path difference and phase.
    float OPD     = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    vec3  phi     = vec3(phi21) + phi23;
    // Compound reflectance — exact formulas from Belcour 2017 §3.
    vec3  R123    = clamp(R12 * R23, vec3(1e-5), vec3(0.9999));
    vec3  r123    = sqrt(R123);
    vec3  Rs      = (T121 * T121) * R23 / (vec3(1.0) - R123);
    // First-order Fourier (m=0) — base reflectance.
    vec3 C0 = R12 + Rs;
    vec3 I  = C0;
    // Higher-order Fourier terms: spectral cosine integrals (m=1,2).
    vec3 Cm = Rs - T121;
    for (int m = 1; m <= 2; ++m) {
        Cm     *= r123;
        vec3 Sm = 2.0 * iridSensitivity(float(m) * OPD, float(m) * phi);
        I      += Cm * Sm;
    }
    return max(I, vec3(0.0));
}

// VNDF pdf for a sampled wi given wo, n, alpha = roughness². Walter 2007:
//   PDF_h = D · G1 · VdotH / NdotV
// then Jacobian to wi via half-vector reflection: divide by 4·VdotH.
// VdotH cancels → PDF_wi = D · G1 / (4 · NdotV).  Matches WGPU's vndfPdf.
float vndfPdf(vec3 wo, vec3 wi, vec3 n, float roughness) {
    const vec3 hm = normalize(wo + wi);
    const float NdotH = max(0.0, dot(n, hm));
    const float NdotV = max(1e-6, dot(n, wo));
    const float D     = distGGX(NdotH, roughness);
    const float k     = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float G1v   = geomSmithG1(NdotV, k);
    return D * G1v / (4.0 * NdotV);
}

// Combined BRDF pdf — mixed VNDF spec + cosine diff with the same pSpec
// that the BSDF sampler uses (mix(0.5, 0.98, metalness)).  Used for the MIS
// balance heuristic in NEE.
float brdfPdf(vec3 wo, vec3 wi, vec3 n, float roughness, float metalness) {
    const float NdotL = dot(n, wi);
    if (NdotL <= 0.0) return 0.0;
    const float pSpec   = mix(0.5, 0.98, metalness);
    const float specPdf = vndfPdf(wo, wi, n, roughness);
    const float diffPdf = NdotL * (1.0 / PI);
    return pSpec * specPdf + (1.0 - pSpec) * diffPdf;
}

// 3-lobe BSDF mixture pdf: base-spec + base-diff + clearcoat. Matches the
// actual sampler in main()'s 3-way stochastic split. Used for env-CDF MIS
// (chit's env NEE side and miss's BSDF→env complement). The 2-lobe brdfPdf
// underestimates the mixture pdf on clearcoat materials — the cc lobe is a
// sharp peak at glossy reflection angles — so MIS comes out skewed and
// adds noise. ccProb=0 collapses this to brdfPdf.
float brdfPdf3(vec3 wo, vec3 wi, vec3 n, float roughness, float metalness,
               float ccProb, float ccRough) {
    const float NdotL = dot(n, wi);
    if (NdotL <= 0.0) return 0.0;
    const float pSpec   = mix(0.5, 0.98, metalness);
    const float specPdf = vndfPdf(wo, wi, n, roughness);
    const float diffPdf = NdotL * (1.0 / PI);
    const float basePdf = pSpec * specPdf + (1.0 - pSpec) * diffPdf;
    if (ccProb <= 0.0) return basePdf;
    const float ccPdf = vndfPdf(wo, wi, n, ccRough);
    return ccProb * ccPdf + (1.0 - ccProb) * basePdf;
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

// Binary search a 1-row CDF lookup. row=0 is used for the 1×h marginal CDF
// (treated as a single column laid out top-to-bottom).
int envCdfSearch(sampler2D tex, int row, int size, float xi) {
    int lo = 0;
    int hi = size - 1;
    for (int it = 0; it < 16; ++it) {
        if (lo >= hi) break;
        const int mid = (lo + hi) >> 1;
        const float v = texelFetch(tex, ivec2(mid, row), 0).x;
        if (v < xi) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Importance-sample the env by luminance. Returns vec4(dir, pdfOmega) where
// pdfOmega = lum(pixel) · W·H / (2π² · totalSum). The 2π² factor is the
// equirect uv→ω Jacobian (= 2π²·cos(lat)) cancelling with the cos(lat) in
// the luminance weighting. totalSum is the host-built Σ lum·cos(lat).
// Returns vec4(0) if no CDF is bound (totalSum <= 0).
vec4 sampleEnvImportance(inout uint seed) {
    if (pc.envCdfTotalSum <= 0.0) return vec4(0.0);
    const int W = int(pc.envCdfWidth);
    const int H = int(pc.envCdfHeight);
    if (W <= 0 || H <= 0) return vec4(0.0);

    const int row = envCdfSearch(envMargTex, 0, H, urand(seed));
    const int col = envCdfSearch(envCdfTex, row, W, urand(seed));

    // Sub-pixel jitter inside the chosen cell so pdf_uv = uniform-within-pixel
    // matches the integrand exactly (cell-center snap biases smooth integrands
    // by ~7% per WGPU memory).
    const vec2 uv = vec2((float(col) + urand(seed)) / float(W),
                         (float(row) + urand(seed)) / float(H));
    const vec3 dir = normalize(uvToEquirectDir(uv));

    const vec3 envCol = texelFetch(envTex, ivec2(col, row), 0).rgb;
    const float lum   = dot(envCol, vec3(0.2126, 0.7152, 0.0722)) + 1e-10;
    const float pdfOmega = lum * float(W * H) / (2.0 * PI * PI * max(pc.envCdfTotalSum, 1e-10));
    return vec4(dir, pdfOmega);
}

// Solid-angle pdf for an arbitrary direction under the env CDF. Used for MIS
// when a BSDF-sampled bounce hits the env — needs to know how likely env
// importance sampling would have picked the same direction.
float envImportancePdf(vec3 dir) {
    if (pc.envCdfTotalSum <= 0.0) return 0.0;
    const int W = int(pc.envCdfWidth);
    const int H = int(pc.envCdfHeight);
    if (W <= 0 || H <= 0) return 0.0;
    const vec2 uv = dirToEquirectUV(normalize(dir));
    const int col = clamp(int(uv.x * float(W)), 0, W - 1);
    const int row = clamp(int(uv.y * float(H)), 0, H - 1);
    const vec3 envCol = texelFetch(envTex, ivec2(col, row), 0).rgb;
    const float lum   = dot(envCol, vec3(0.2126, 0.7152, 0.0722)) + 1e-10;
    return lum * float(W * H) / (2.0 * PI * PI * max(pc.envCdfTotalSum, 1e-10));
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

    // Per-vertex foam coverage [0..1] (FFT-Tessendorf Jacobian < 1 → folding
    // → whitewater). Interpolated across the triangle and used downstream to
    // bleach albedo and pin roughness toward 1 where the surface breaks. 0
    // for any non-DisplacedMesh geometry (foamAddress is null).
    float foamCoverage = 0.0;
    if (gdesc.foamAddress != 0ul) {
        FoamBuf fb = FoamBuf(gdesc.foamAddress);
        const float f0 = fb.f[idx.x];
        const float f1 = fb.f[idx.y];
        const float f2 = fb.f[idx.z];
        foamCoverage = clamp(w * f0 + attribs.x * f1 + attribs.y * f2, 0.0, 1.0);
    }

    const vec3 nObj = normalize(w * n0 + attribs.x * n1 + attribs.y * n2);
    const vec3 Nworld = normalize(transpose(mat3(gl_WorldToObjectEXT)) * nObj);

    const vec3 V = normalize(-gl_WorldRayDirectionEXT);
    // Front-face = ray came from outside the surface (V on the same side as the
    // outward normal). Captured before flipping so the transmission lobe below
    // can pick the correct refraction η and origin offset on back-facing hits.
    const bool isFront = dot(Nworld, V) >= 0.0;
    vec3 N = isFront ? Nworld : -Nworld;// double-sided shading for thin meshes / cull-disabled

    // Single-sided material hit from behind: pass the ray through unchanged.
    if (!isFront && mdesc.doubleSided == 0 && mdesc.transmission <= 0.0) {
        payload.radiance      = vec3(0.0);
        payload.brdfWeight    = vec3(1.0);
        // No advance past hitT — raygen uses a 1e-4 tmin for pass-through
        // bounces so the next ray sees a host surface sitting right behind
        // the just-hit pass-through surface (decal-on-mesh case).
        payload.nextOrigin    = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        payload.nextDir       = gl_WorldRayDirectionEXT;
        payload.flags         = 4u;
        // Tag this surface so raygen can record it as the primary if nothing
        // opaque is found later (e.g. pass-through → sky). Without this, sky-
        // facing back-faces never accumulate history and stay permanently noisy.
        payload.hitWorldPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        payload.hitInstanceId = uint(gl_InstanceCustomIndexEXT) + 1u;
        // Pass-through: no shading, so use the material's nominal roughness as
        // a coarse-but-safe FC cap input. The next visible surface upstream
        // overrides this if it lands as the primary hit.
        payload.hitRoughness    = clamp(mdesc.roughness, 0.04, 1.0);
        payload.hitMetalness    = clamp(mdesc.metalness, 0.0, 1.0);
        payload.hitTransmission = clamp(mdesc.transmission, 0.0, 1.0);
        return;
    }

    // MeshBasicMaterial: unlit early-out. Sentinel `roughness < 0` set on the
    // host (VulkanRenderer.cpp materialFromMesh). Emit base color as direct
    // radiance and terminate the path — no lighting, NEE, or bounce. Mirrors
    // WGPU's `shininess == -1` unlit gate.
    if (mdesc.roughness < 0.0) {
        vec2 unlitUv = vec2(0.0);
        if (gdesc.uvAddress != 0ul) {
            UvBuf ub = UvBuf(gdesc.uvAddress);
            const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
            const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
            const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
            unlitUv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;
        }
        vec3 albedoSample = vec3(1.0);
        if (mdesc.albedoTexIndex >= 0) {
            const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
            const vec2 uvA = (mdesc.uvTransform * vec3(unlitUv, 1.0)).xy;
            albedoSample = texture(albedoMaps[idxClamped], uvA).rgb;
        }
        const vec3 hitPosUnlit = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        payload.radiance      = mdesc.albedo * albedoSample;
        payload.brdfWeight    = vec3(0.0);
        payload.nextOrigin    = vec3(0.0);
        payload.nextDir       = vec3(0.0);
        payload.flags         = 1u;// terminate
        payload.hitWorldPos     = hitPosUnlit;
        payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
        payload.hitRoughness    = 1.0;
        payload.hitMetalness    = 0.0;
        payload.hitTransmission = 0.0;
        return;
    }

    // UV interpolation (only when the geometry has a uv attribute). The
    // outer fallback is vec2(0) — harmless for materials without an albedo
    // texture; the slot-0 white default would just sample (0,0) anyway.
    vec2 rawUv = vec2(0.0);
    if (gdesc.uvAddress != 0ul) {
        UvBuf ub = UvBuf(gdesc.uvAddress);
        const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
        const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
        const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
        rawUv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;

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
                    vec3 ns = texture(albedoMaps[nidx], (mdesc.uvTransformNormal * vec3(rawUv, 1.0)).xy).rgb * 2.0 - 1.0;
                    ns.xy *= mdesc.normalScale;
                    ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
                    N = normalize(T * ns.x + B * ns.y + N * ns.z);
                }
            }
        }
    }
    // Per-channel transformed UVs — applied to rawUv with each texture's own matrix.
    const vec2 uvAlbedo         = (mdesc.uvTransform            * vec3(rawUv, 1.0)).xy;
    const vec2 uvRoughMetal     = (mdesc.uvTransformRoughMetal  * vec3(rawUv, 1.0)).xy;
    const vec2 uvEmissive       = (mdesc.uvTransformEmissive    * vec3(rawUv, 1.0)).xy;
    const vec2 uvOcclusion      = (mdesc.uvTransformOcclusion   * vec3(rawUv, 1.0)).xy;
    const vec2 uvClearcoat      = (mdesc.uvTransformClearcoat   * vec3(rawUv, 1.0)).xy;
    const vec2 uvClearcoatRough = (mdesc.uvTransformClearcoatRough * vec3(rawUv, 1.0)).xy;
    const vec2 uvTransmission   = (mdesc.uvTransformTransmission * vec3(rawUv, 1.0)).xy;

    const float NdotV = max(dot(N, V), 0.0);

    // Albedo: scalar PBR colour modulated by the bound albedo map (sRGB
    // decode is hardware-side via the VK_FORMAT_R8G8B8A8_SRGB view).
    vec3 albedoSample = vec3(1.0);
    if (mdesc.albedoTexIndex >= 0) {
        const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedoSample = texture(albedoMaps[idxClamped], uvAlbedo).rgb;
    }
    vec3 albedo = mdesc.albedo * albedoSample;

    // glTF packs roughness in .g and metalness in .b; threepp's metalnessMap /
    // roughnessMap typically point at the same packed texture, so the bindless
    // cache dedupes to a single slot. Multiplicative — matches three.js.
    float roughness = mdesc.roughness;
    float metalness = mdesc.metalness;
    if (mdesc.roughnessTexIndex >= 0) {
        const int i = clamp(mdesc.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        roughness *= texture(albedoMaps[i], uvRoughMetal).g;
    }
    if (mdesc.metalnessTexIndex >= 0) {
        const int i = clamp(mdesc.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        metalness *= texture(albedoMaps[i], uvRoughMetal).b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metalness = clamp(metalness, 0.0,  1.0);

    // Foam application: folded-surface vertices (foamCoverage > 0, set by
    // water_displace.comp via the Tessendorf Jacobian) bleach the albedo
    // toward white and push roughness toward 1.0 (fully diffuse). The
    // transmission lobe also reads `transmission` later in the shader; we
    // suppress it on heavy foam so whitecaps read as opaque whitewater
    // rather than tinted glass-foam. No-op when foamCoverage = 0.
    if (foamCoverage > 0.0) {
        albedo    = mix(albedo,    vec3(1.0), foamCoverage);
        roughness = mix(roughness, 1.0,       foamCoverage);
    }

    vec3 F0 = mix(vec3(0.04) * mdesc.specularIntensity * mdesc.specularColor, albedo, metalness);
    // Thin-film iridescence layer (KHR_materials_iridescence). Modulates F0
    // with wavelength-dependent interference; lobe shape (GGX) is unchanged,
    // only the Fresnel base shifts per channel. Skipped when factor == 0
    // so non-iridescent materials pay nothing beyond the branch.
    if (mdesc.iridescence > 0.0) {
        const vec3 irid = evalIridescence(1.0, mdesc.iridescenceIOR, NdotV,
                                          mdesc.iridescenceThicknessNm, F0);
        F0 = mix(F0, irid, mdesc.iridescence);
    }
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
        ccScalar *= texture(albedoMaps[i], uvClearcoat).r;
    }
    if (mdesc.clearcoatRoughnessTexIndex >= 0) {
        const int i = clamp(mdesc.clearcoatRoughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        ccRough *= texture(albedoMaps[i], uvClearcoatRough).g;
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

    // BLEND-mode (alphaCutoff < 0 sentinel) is now handled stochastically inside
    // the any-hit shader: when the coin says "transparent", the BVH skips the
    // candidate and continues past it. By the time we reach closest_hit on a
    // BLEND material, the texel rolled "opaque" for this sample, so we shade
    // the surface normally. The retrace approach used previously sat fireflies
    // on top of decal-on-mesh setups (LeePerrySmith head decal); the in-BVH
    // rejection mirrors WGPU PT's testTriangle() and converges cleanly.

    // === Transmission lobe ===
    // Russian-roulette gate by mdesc.transmission: with probability `transmission`
    // this hit acts as a glass interface (Schlick-Fresnel weighted reflect /
    // refract) and skips direct lighting + env NEE entirely. The path continues
    // with the chosen bounce direction. Matches the WGPU PT pattern: no
    // 1/transmission inverse-prob scaling, so `transmission` doubles as a
    // stylised reflect-vs-transmit blend factor (artist control), not a physical
    // mixing weight. Dispersion is sampled per-channel below; thin-wall
    // thickness proxy is wired through `mdesc.thickness` in the BL block.
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
        transmission *= texture(albedoMaps[i], uvTransmission).r;
    }
    // Foam suppresses transmission so whitecaps read as opaque whitewater
    // rather than tinted glass. mix toward 0 by foamCoverage.
    transmission *= (1.0 - foamCoverage);
    if (transmission > 0.0 && urand(seed) < transmission) {
        const float ior_base = max(mdesc.ior, 1.0);
        const vec3  I        = gl_WorldRayDirectionEXT;

        // Sample a GGX microfacet normal so roughness scatters the refracted
        // ray. α=0 (smooth glass) degenerates to H=N (mirror). Fresnel and
        // reflect/refract all use H rather than N.
        const vec2  u2   = vec2(urand(seed), urand(seed));
        const vec3  H    = sampleVNDF_H(V, N, alpha, u2);
        const float cosH = max(dot(V, H), 0.0);

        // Thin-shell BSDF: explicit per-material opt-in via
        // MaterialWithThickness::thinWalled. Examples: FFT-displaced ocean
        // plane, sunglasses lens, leaf, single sheet of glass. Both faces
        // are treated as "entering" — refract uses eta=1/ior on both sides
        // — and the ray's medium isn't tracked through the surface (it
        // "exits" instantly at the same point, conceptually).
        const bool isThinShell = mdesc.thinWalled != 0;

        // Medium tracking — fixes camera-inside-closed-glass (e.g. windshield
        // from cabin). Geometric isFront alone would misclassify a back-face
        // hit as "exiting glass" when the ray was actually still in air. We
        // instead determine entry/exit from the ray's current medium IOR
        // carried in the payload: if we're already in this material's
        // medium → exiting; otherwise → entering. For thin shells, force
        // entering-on-both-faces and don't propagate the medium change.
        const float currentIor = max(payload.currentIor, 1.0);
        float targetIor;
        bool isEntering;
        if (isThinShell) {
            targetIor   = ior_base;
            isEntering  = true;
        } else if (abs(currentIor - ior_base) < 1e-3) {
            targetIor   = 1.0;        // exit to air (default outside medium)
            isEntering  = false;
        } else {
            targetIor   = ior_base;
            isEntering  = true;
        }

        // Schlick Fresnel at the microfacet half-vector. Uses base IOR so the
        // reflect/refract branch decision is achromatic — chromatic Fresnel
        // adds visible noise without much benefit at typical dispersion values.
        // Exiting-side path uses the transmitted-side cosine so TIR raises
        // F→1 smoothly.
        const float r0    = pow((1.0 - ior_base) / (1.0 + ior_base), 2.0);
        const float sin2H = max(0.0, 1.0 - cosH * cosH);
        const float cosSchlick = isEntering
                ? cosH
                : sqrt(max(0.0, 1.0 - ior_base * ior_base * sin2H));
        const float F = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);

        // === Reflect / refract — two strategies depending on geometry ===
        // For THIN shells (isThinShell=true: ocean plane, sunglasses lens) we
        // use a deterministic split: reflect lobe is evaluated analytically as
        // F · env(reflectDir) · visibility via a shadow ray, refract continues
        // the path with weight (1−F)·glassTint. Variance from the Fresnel pick
        // drops to zero — major denoiser-off win on smooth water.
        //
        // For CLOSED glass (isThinShell=false: spheres, goblets, vases) we use
        // the original stochastic split — pick reflect with prob F, refract
        // with prob (1−F), single ray continues. The shadow-ray approximation
        // only captures one straight-line through the glass to env, so for
        // concave geometry where a ray TIRs and bounces multiple times inside
        // the glass before exiting, multi-bounce energy is lost (manifests as
        // black bands on goblet stems, etc.). The stochastic path traverses
        // every internal bounce naturally and recovers that energy.
        //
        // Dispersion (KHR_materials_dispersion) stays stochastic per
        // wavelength regardless of mode — three-channel sampling, ×3 boost.
        float ior = ior_base;
        vec3 channelMask = vec3(1.0);
        if (mdesc.dispersion > 0.0) {
            const vec3  lambda    = vec3(0.6563, 0.5500, 0.4861);
            const float refInvSq  = 1.0 / (0.5893 * 0.5893);
            const uint  ch        = uint(urand(seed) * 3.0) % 3u;
            const float invSq     = 1.0 / (lambda[ch] * lambda[ch]);
            const float B         = (ior_base - 1.0) * mdesc.dispersion / 38.2;
            ior = ior_base + B * (invSq - refInvSq);
            channelMask = vec3(0.0);
            channelMask[ch] = 3.0;
        }
        // eta = currentIor / targetIor; for entering it reduces to (1/ior),
        // for exiting to ior. Dispersion adjusts `ior` per channel above; the
        // medium-tracking state itself uses the achromatic ior_base.
        const float eta = isEntering ? (1.0 / ior) : ior;
        const vec3 refr = refract(I, H, eta);
        const bool tir = (dot(refr, refr) < 1e-6);

        // Glass tint (used by both modes). Wraps Beer-Lambert in either the
        // thin-shell proxy (per-crossing thickness) or the closed-mesh actual-
        // distance branch.
        const float cosOut = !tir ? abs(dot(normalize(refr), H)) : 1.0;
        const float G1out  = smithG1(cosOut, alpha);
        vec3 tintBase;
        if (mdesc.ior <= 1.01) {
            const float albedoLum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
            tintBase = mix(vec3(1.0), albedo, smoothstep(0.0, 0.1, albedoLum));
        } else {
            tintBase = albedo;
        }
        vec3 glassTint = tintBase * G1out;
        if (mdesc.attenuationDistance > 0.0) {
            if (isThinShell) {
                // Thin-shell proxy — use the user-supplied `thickness` as
                // in-medium distance. Applied at every entry crossing.
                glassTint *= pow(max(mdesc.attenuationColor, vec3(1e-6)),
                                 vec3(mdesc.thickness / mdesc.attenuationDistance));
            } else if (!isEntering) {
                // Closed-mesh actual ray distance through the medium —
                // matches the original (pre-branch) behaviour. An earlier
                // attempt added a `thickness` fallback when gl_HitTEXT < 1e-2
                // (mirrored from WGPU); it misfired on thin-walled closed
                // glass (goblet walls < 1 cm) by replacing the genuine 5 mm
                // ray distance with an unrelated asset-thickness value, and
                // over-darkened the glass into solid blue. Keep it simple.
                glassTint *= pow(max(mdesc.attenuationColor, vec3(1e-6)),
                                 vec3(gl_HitTEXT / mdesc.attenuationDistance));
            }
        }

        vec3 reflectContrib = vec3(0.0);
        vec3 wDir    = vec3(0.0);
        vec3 wOrigin = vec3(0.0);
        vec3 tWeight = vec3(0.0);
        bool terminate = false;
        bool wasReflect = false;

        if (isThinShell) {
            // ── Deterministic split (variance reduction for thin shells) ──
            const vec3 reflectDir    = reflect(I, H);
            const vec3 reflectOrigin = hitPos + N * 1e-3;
            shadowVisibility = 1.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT |
                        gl_RayFlagsNoOpaqueEXT,
                        0xff, 1, 0, 1,
                        reflectOrigin, 0.0, reflectDir, 1e30, 1);
            vec3 reflectEnv = vec3(0.0);
            if (shadowVisibility > 0.0) {
                reflectEnv = sampleEquirect(reflectDir);
            }
            reflectContrib = F * reflectEnv * shadowVisibility;

            if (!tir) {
                wDir    = normalize(refr);
                wOrigin = hitPos - N * 1e-3;
                tWeight = (1.0 - F) * glassTint * channelMask / (eta * eta);
            } else {
                // TIR — Schlick gave F=1 already, full reflection captured by
                // reflectContrib. Terminate path.
                terminate = true;
            }
        } else {
            // ── Stochastic split (original — multi-bounce intact for closed glass) ──
            if (urand(seed) < F) {
                wDir       = reflect(I, H);
                wOrigin    = hitPos + N * 1e-3;
                tWeight    = vec3(1.0);
                wasReflect = true;
            } else if (tir) {
                // TIR — fall back to mirror reflect.
                wDir       = reflect(I, H);
                wOrigin    = hitPos + N * 1e-3;
                tWeight    = vec3(1.0);
                wasReflect = true;
            } else {
                wDir    = normalize(refr);
                wOrigin = hitPos - N * 1e-3;
                tWeight = glassTint * channelMask / (eta * eta);
                // Closed-mesh refract — update the ray's current medium so
                // the next bounce knows where it is (entering glass: medium
                // becomes glass; exiting: medium becomes air). Reflect /
                // TIR-fallback paths leave currentIor untouched (ray stays
                // in the same medium).
                payload.currentIor = targetIor;
            }
        }

        vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
        if (mdesc.emissiveTexIndex >= 0) {
            const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
            emissiveOut *= texture(albedoMaps[ei], uvEmissive).rgb;
        }
        const float emLum0 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
        if (emLum0 > pc.fireflyClamp) emissiveOut *= pc.fireflyClamp / emLum0;
        if ((payload.inFlags & 1u) != 0u) emissiveOut = vec3(0.0);
        // For thin-shell mode reflectContrib was captured analytically; for
        // stochastic mode it stays vec3(0) and the reflect lobe's contribution
        // arrives via path traversal. raygen adds throughput·radiance to the
        // total each hit and multiplies throughput by brdfWeight.
        payload.radiance      = emissiveOut + reflectContrib;
        payload.brdfWeight    = tWeight;
        payload.nextOrigin    = wOrigin;
        payload.nextDir       = wDir;
        // flags=1 terminates path (only thin-shell + TIR right now);
        // flags=4 continues. Stochastic mode never terminates here (TIR
        // falls back to reflect) so the path always continues.
        payload.flags         = terminate ? 1u : 4u;
        payload.seed          = seed;
        // Primary-tag policy for transmission hits. With deterministic split,
        // every hit captures both reflect and refract contributions, so the
        // decision is purely material-based:
        //   ior > 1.01: real glass — tag glass as primary. Reproject anchors
        //               on the glass surface, which gives stable sky/env
        //               reflections under camera motion (the fix the
        //               vulkan-pt branch was carrying — original symptom:
        //               glass tint disappearing during movement).
        //   ior ≈ 1.0:  alpha-blend / stochastic pass-through. Refraction
        //               returns the incident direction unchanged → the ray
        //               continues to the surface behind, which IS what
        //               should anchor reproject. Skip the primary tag so
        //               the next non-zero hit wins.
        const bool tagPrimary = (ior_base > 1.01);
        if (tagPrimary) {
            payload.hitWorldPos     = hitPos;
            payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
            payload.hitRoughness    = roughness;
            payload.hitMetalness    = metalness;
            payload.hitTransmission = mdesc.transmission;
        } else {
            payload.hitWorldPos     = vec3(0.0);
            payload.hitInstanceId   = 0u;
            payload.hitRoughness    = 1.0;
            payload.hitMetalness    = 0.0;
            payload.hitTransmission = 0.0;
        }
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

        // Shadow ray via hit group 1 (shadow_anyhit.rahit). NoOpaqueEXT forces
        // all geometry through the any-hit so glass/cutout can attenuate.
        // shadowVisibility starts at 1.0; glass multiplies it down; opaque sets
        // it to 0; miss is a no-op so the accumulated transmittance is the result.
        // maxDist = 1e30 is the "infinite ray" sentinel (sun through atmosphere)
        // so the fog attenuation below treats DirLight as outside the fog volume.
        shadowVisibility = 1.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT |
                    gl_RayFlagsNoOpaqueEXT,
                    0xff, 1, 0, 1,// sbtOffset=1 → shadow hit group; missIndex=1
                    hitPos + N * 1e-3, 0.0, L, 1e30, 1);
        if (shadowVisibility <= 0.0) continue;

        const vec3  H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);

        const vec3  F        = fresnelSchlick(VdotH, F0);
        const float D        = distGGX(NdotH, roughness);
        const float G        = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd       = (vec3(1.0) - F) * (1.0 - metalness);
        const vec3  diffuse  = kd * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);

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
        // DirLight uses 1e30 maxDist sentinel (above), so fogAtten is vec3(1)
        // here — the sun is "outside" the fog. The volumeInscatter pass in
        // raygen handles the camera-ray scattering of this light into haze.
        lit += perLight * NdotL * lights.dirLights[i].color * shadowVisibility;
    }

    // === Point lights ===
    for (uint i = 0u; i < lights.pointCount; ++i) {
        vec3        toL  = lights.pointLights[i].position - hitPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float NdotL = max(dot(N, toL), 0.0);
        if (NdotL <= 0.0) continue;

        // Frostbite/three.js physical falloff: 1/d^decay. threepp's PointLight
        // defaults decay=1 (linear); KHR_lights_punctual / modern three.js use
        // decay=2 (inverse square). Hardcoding d² would clamp every PointLight
        // to inverse-square and ignore the user's `pl->decay`. Matches WGPU PT.
        const float decay = lights.pointLights[i].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float range = lights.pointLights[i].range;
        if (range > 0.0) {
            // Three.js / KHR_lights_punctual smooth window: pow(saturate(1 - (d/r)^4), 2).
            // Quartic interior (stays near 1.0 across most of the range)
            // squared at the edge for a soft cutoff. Matches WGPU PT.
            const float t  = dist / range;
            const float t4 = t * t * t * t;
            const float w  = max(1.0 - t4, 0.0);
            atten *= w * w;
        }

        shadowVisibility = 1.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT |
                    gl_RayFlagsNoOpaqueEXT,
                    0xff, 1, 0, 1,
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
        const vec3  diff  = kd_p * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
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
        const vec3 fogAttenPt = fogEnabled() ? fogTransmittance(dist) : vec3(1.0);
        lit += perPt * NdotL * lights.pointLights[i].color * atten * shadowVisibility * fogAttenPt;
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

        // Distance falloff matches PointLight: 1/d^decay (Frostbite physical).
        const float decay = lights.spotLights[i].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float range = lights.spotLights[i].range;
        if (range > 0.0) {
            // Three.js / KHR_lights_punctual smooth window: pow(saturate(1 - (d/r)^4), 2).
            const float t  = dist / range;
            const float t4 = t * t * t * t;
            const float w  = max(1.0 - t4, 0.0);
            atten *= w * w;
        }
        atten *= spotAtten;

        shadowVisibility = 1.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT |
                    gl_RayFlagsNoOpaqueEXT,
                    0xff, 1, 0, 1,
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
        const vec3  diff  = kd_s * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
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
        const vec3 fogAttenSp = fogEnabled() ? fogTransmittance(dist) : vec3(1.0);
        lit += perSp * NdotL * lights.spotLights[i].color * atten * shadowVisibility * fogAttenSp;
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

        shadowVisibility = 1.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT |
                    gl_RayFlagsNoOpaqueEXT,
                    0xff, 1, 0, 1,
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
        const vec3  diff  = kd_r * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
        vec3 perRect = (diff + spec) * baseScale;
        if (ccWeight > 0.0) {
            const float k_cc    = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
            const float D_cc    = distGGX(NdotH, ccRough);
            const float G_cc    = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
            perRect += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4) * ccWeight);
        }
        const vec3 fogAttenRect = fogEnabled() ? fogTransmittance(dist) : vec3(1.0);
        lit += perRect * NdotL * lights.rectLights[i].color * geomTerm * shadowVisibility * fogAttenRect;
    }

    // === Emissive-mesh NEE ===
    // Power-weighted picking from the per-frame emissive-triangle CDF, then
    // uniform area sample on the chosen triangle. Mirrors WGPU's
    // sampleEmissiveTriCdf (WgpuPathTracerShaders_Rt.cpp:1622). This shade's
    // BSDF bounce will not double-count: raygen sets payload.inFlags bit 0 on
    // the next iteration (when emissiveCount > 0), and the closest_hit zeros
    // emissiveOut on that bit.
    if (pc.emissiveCount > 0u && pc.emissiveTotalPower > 0.0) {
        // Binary search the cumulative-power CDF (stored in v1.w of each tri).
        const float xi = urand(seed) * pc.emissiveTotalPower;
        uint lo = 0u;
        uint hi = pc.emissiveCount - 1u;
        for (int s = 0; s < 32; ++s) {
            if (lo >= hi) break;
            const uint mid = (lo + hi) >> 1u;
            if (emissiveTris[mid].v1.w < xi) lo = mid + 1u;
            else                              hi = mid;
        }
        const EmTri t = emissiveTris[lo];
        // sqrt-barycentric for uniform-by-area sampling on the triangle.
        const float r1 = urand(seed);
        const float r2 = urand(seed);
        const float su1 = sqrt(r1);
        const float bA = 1.0 - su1;
        const float bB = su1 * (1.0 - r2);
        const float bC = su1 * r2;
        const vec3 lp = bA * t.v0.xyz + bB * t.v1.xyz + bC * t.v2.xyz;
        vec3 toL = lp - hitPos;
        const float dist2 = dot(toL, toL);
        const float dist  = sqrt(max(dist2, 1e-20));
        if (dist > 1e-4) {
            toL /= dist;
            const float NdotLe = dot(N, toL);
            // Triangle geometric normal (orientation as authored). Two-sided
            // emitters by default — matches WGPU; cosLight uses |dot|.
            const vec3 ge1 = t.v1.xyz - t.v0.xyz;
            const vec3 ge2 = t.v2.xyz - t.v0.xyz;
            const vec3 lnRaw = cross(ge1, ge2);
            const float lnLen = length(lnRaw);
            // Grazing receiver / emitter angles produce arbitrarily-large NEE
            // contributions (small pdfOmega, nearly-zero BRDF cosine compensated
            // by 1/cosLe denominator). Even with MIS these can leak fireflies
            // through brdfPdf's G1V approximation; reject early instead.
            if (NdotLe > 0.01 && lnLen > 1e-20) {
                const vec3 lN = lnRaw / lnLen;
                const float cosLight = abs(dot(-toL, lN));
                if (cosLight > 0.01 && t.v0.w > 1e-20 && t.v2.w > 0.0) {
                    shadowVisibility = 1.0;
                    traceRayEXT(topAS,
                                gl_RayFlagsTerminateOnFirstHitEXT |
                                gl_RayFlagsSkipClosestHitShaderEXT |
                                gl_RayFlagsNoOpaqueEXT,
                                0xff, 1, 0, 1,
                                hitPos + N * 1e-3, 0.0, toL, dist - 1e-2, 1);
                    if (shadowVisibility > 0.0) {
                        // Solid-angle pdf: (power/totalPower) * dist² / (area*cosLight)
                        const float pickPdf = t.v2.w / pc.emissiveTotalPower;
                        const float pdfOmega = pickPdf * dist2 / (t.v0.w * cosLight);
                        // Cook-Torrance + Lambert + (optional) clearcoat at toL.
                        const vec3  H_e   = normalize(V + toL);
                        const float NdotH = max(dot(N, H_e), 0.0);
                        const float VdotH = max(dot(V, H_e), 0.0);
                        const vec3  F_e   = fresnelSchlick(VdotH, F0);
                        const float D_e   = distGGX(NdotH, roughness);
                        const float G_e   = geomSmithG1(NdotV, k) * geomSmithG1(NdotLe, k);
                        const vec3  spec_e = (D_e * G_e * F_e) / max(4.0 * NdotV * NdotLe, 1e-4);
                        const vec3  kd_e   = (vec3(1.0) - F_e) * (1.0 - metalness);
                        const vec3  diff_e = kd_e * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
                        vec3 perEm = (diff_e + spec_e) * baseScale;
                        if (ccWeight > 0.0) {
                            const float k_cc = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
                            const float D_cc = distGGX(NdotH, ccRough);
                            const float G_cc = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotLe, k_cc);
                            perEm += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotLe, 1e-4) * ccWeight);
                        }
                        vec3 emCol = t.emission.rgb;
                        const float emLumE = dot(emCol, vec3(0.2126, 0.7152, 0.0722));
                        if (emLumE > pc.fireflyClamp) emCol *= pc.fireflyClamp / emLumE;
                        // MIS balance heuristic against the BSDF-sampled bounce
                        // estimator. On glossy surfaces pdfBsdf >> pdfOmega →
                        // w_light → 0 (NEE contributes nothing, the BSDF bounce
                        // handles the spec lobe). On rough/diffuse pdfOmega
                        // dominates → w_light → 1. Without this, NEE samples
                        // landing inside a narrow spec lobe spike to enormous
                        // values when divided by pdfOmega.
                        const float pdfBsdfNee = brdfPdf(V, toL, N, roughness, metalness);
                        const float wLight = pdfOmega / max(pdfOmega + pdfBsdfNee, 1e-8);
                        vec3 emContrib = perEm * NdotLe * emCol * wLight / max(pdfOmega, 1e-8);
                        // Per-contribution firefly clamp.  Anything brighter
                        // than the user-tunable cap on a single NEE sample is
                        // either a numerical spike or a legitimate direct-
                        // bright tap that the next sample will reinforce
                        // anyway. Default 20.0 (set via setFireflyClamp).
                        const float emCLum = dot(emContrib, vec3(0.2126, 0.7152, 0.0722));
                        if (emCLum > pc.fireflyClamp) emContrib *= pc.fireflyClamp / emCLum;
                        const vec3 fogAttenEm = fogEnabled() ? fogTransmittance(dist) : vec3(1.0);
                        lit += emContrib * shadowVisibility * fogAttenEm;
                    }
                }
            }
        }
    }

    // === Env NEE + MIS ===
    //   • envCdfTotalSum > 0  → importance-sample the env by luminance, eval
    //     the BSDF at that direction, weight by w_env = pdf_env / (pdf_env +
    //     pdf_brdf). The miss handler computes the BSDF→env complement using
    //     payload.bsdfPdf and the same env CDF. Big variance reduction on
    //     HDRI scenes with bright sun discs / sky bands.
    //   • envCdfTotalSum <= 0 → fallback to BSDF-sampled env NEE with constant
    //     0.5 MIS weight (matches miss's 0.5 in the same branch). Same
    //     behavior as before this feature landed.
    if (pc.envCdfTotalSum > 0.0) {
        const vec4  envPick = sampleEnvImportance(seed);
        const vec3  nDir    = envPick.xyz;
        const float pdfEnv  = envPick.w;
        const float NdotL_e = dot(N, nDir);
        if (pdfEnv > 1e-10 && NdotL_e > 0.0) {
            shadowVisibility = 1.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT |
                        gl_RayFlagsNoOpaqueEXT,
                        0xff, 1, 0, 1,
                        hitPos + N * 1e-3, 0.0, nDir, 1e30, 1);
            if (shadowVisibility > 0.0) {
                // Eval BSDF at the env-importance-sampled direction (Cook-
                // Torrance + Lambert + K-C ms-comp; clearcoat layer too).
                const vec3  H_e   = normalize(V + nDir);
                const float NdotH = max(dot(N, H_e), 0.0);
                const float VdotH = max(dot(V, H_e), 0.0);
                const vec3  F_e   = fresnelSchlick(VdotH, F0);
                const float D_e   = distGGX(NdotH, roughness);
                const float G_e   = geomSmithG1(NdotV, k) * geomSmithG1(NdotL_e, k);
                const vec3  spec_e = (D_e * G_e * F_e) / max(4.0 * NdotV * NdotL_e, 1e-4);
                const vec3  kd_e   = (vec3(1.0) - F_e) * (1.0 - metalness);
                const vec3  diff_e = kd_e * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
                vec3 lobeSum = (diff_e + spec_e) * baseScale;
                if (ccWeight > 0.0) {
                    const float k_cc = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
                    const float D_cc = distGGX(NdotH, ccRough);
                    const float G_cc = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL_e, k_cc);
                    lobeSum += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL_e, 1e-4) * ccWeight);
                }
                vec3 envSample = sampleEquirect(nDir);
                const float envLum = dot(envSample, vec3(0.2126, 0.7152, 0.0722));
                if (envLum > pc.fireflyClamp) envSample *= pc.fireflyClamp / envLum;
                // 3-lobe pdf: matches the cc + base-spec + base-diff sampler
                // in main()'s bounce direction selection. Clearcoat-aware so
                // MIS doesn't add noise on clearcoat materials.
                const float pdfBrdf = brdfPdf3(V, nDir, N, roughness, metalness, ccProb, ccRough);
                const float wEnv    = pdfEnv / max(pdfEnv + pdfBrdf, 1e-8);
                lit += wEnv * lobeSum * NdotL_e * envSample * shadowVisibility / max(pdfEnv, 1e-8);
            }
        }
    } else {
        // Fallback: BSDF-sampled env NEE with constant 0.5 MIS (no CDF). 3-way
        // stochastic lobe selection mirrors the bounce sampler so the implicit
        // pdfs match and the balance-heuristic weight collapses to 0.5 / 0.5.
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
                nWeight = albedo * (1.0 - metalness) * kcBoost(F0, NdotV, alpha) * baseScale * invBaseProb / (1.0 - pSpec);
            }
        }
        if (nValid) {
            shadowVisibility = 1.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT |
                        gl_RayFlagsNoOpaqueEXT,
                        0xff, 1, 0, 1,
                        hitPos + N * 1e-3, 0.0, nDir, 1e30, 1);
            if (shadowVisibility > 0.0) {
                vec3 envSample = sampleEquirect(nDir);
                const float envLum = dot(envSample, vec3(0.2126, 0.7152, 0.0722));
                if (envLum > pc.fireflyClamp) envSample *= pc.fireflyClamp / envLum;
                lit += 0.5 * nWeight * envSample * shadowVisibility;
            }
        }
    }

    // Flat ambient irradiance — only the diffuse lobe receives it (metals
    // have no diffuse). No PI cancel under physical lights. Scaled by the
    // base-layer fraction so a 100%-clearcoat surface only shows the cc
    // contribution (which is dir-light + env NEE only — no flat ambient
    // term for clearcoat). Occlusion map (.r channel) scales ambient.
    float ao = 1.0;
    if (mdesc.occlusionTexIndex >= 0) {
        const int oi = clamp(mdesc.occlusionTexIndex, 0, int(kMaxMaterialTextures) - 1);
        ao = texture(albedoMaps[oi], uvOcclusion).r;
    }
    const vec3 ambient = albedo * (1.0 - metalness) * lights.ambient * baseScale * ao;

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
            brdfWeight = albedo * (1.0 - metalness) * kcBoost(F0, NdotV, alpha) * baseScale * invBaseProb / (1.0 - pSpec);
        }
    }

    vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
    if (mdesc.emissiveTexIndex >= 0) {
        const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
        emissiveOut *= texture(albedoMaps[ei], uvEmissive).rgb;
    }
    const float emLum1 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
    if (emLum1 > pc.fireflyClamp) emissiveOut *= pc.fireflyClamp / emLum1;
    // Suppress emission on indirect shading hits when the prior shade event
    // already accounted for emissive triangles via NEE. Primary hits, hits
    // after pass-through, and hits on frames with no emissive geometry keep
    // the term so the user can see directly-visible glowing surfaces.
    if ((payload.inFlags & 1u) != 0u) emissiveOut = vec3(0.0);
    if ((pc.motionFlags & 4u) != 0u) lit += gatherCaustics(hitPos, N, albedo, roughness);
    payload.radiance   = emissiveOut + ambient + lit;
    payload.brdfWeight = brdfWeight;
    payload.nextOrigin = hitPos + N * 1e-3;
    payload.nextDir    = bounceDir;
    payload.flags      = pathFlags;
    payload.seed       = seed;
    // Tag this surface for raygen's primary-hit reprojection. Pass-through and
    // transmission early-returns above leave hitInstanceId at 0 so raygen waits
    // for the first real shade event to anchor history. +1 so 0 still means
    // miss/sky after gl_InstanceCustomIndexEXT == 0.
    payload.hitWorldPos     = hitPos;
    payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
    payload.hitRoughness    = roughness;// post-clamp; raygen uses for FC cap on motion
    payload.hitMetalness    = metalness;
    payload.hitTransmission = mdesc.transmission;
    // 3-lobe pdf at the chosen bounce direction (cc + base-spec + base-diff).
    // Miss uses this for the BSDF→env MIS weight when env CDF is enabled. Must
    // match the actual sampler mixture above, otherwise MIS over/underweights
    // and adds noise on clearcoat.
    payload.bsdfPdf = brdfPdf3(V, bounceDir, N, roughness, metalness, ccProb, ccRough);
}
