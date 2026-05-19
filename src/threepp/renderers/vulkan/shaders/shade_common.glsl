// Shared shading helpers used by closest_hit.rchit (and, eventually, the
// hybrid bounce-0-on-raster path in raygen.rgen). All function bodies were
// extracted verbatim from closest_hit.rchit in commit f9c6c… — keep the two
// in lockstep when you touch BSDF math or sampling.
//
// The binding-dependent helpers (sampleEquirect, fogEnabled / fogTransmittance,
// gatherCaustics) reference globals declared by the includer:
//   - envTex (binding 6)
//   - fog    (binding 17)
//   - photonCounts (binding 15), photonData (binding 16)
//   - kPhotonGridSize / kPhotonsPerCell / kGatherRadius from vulkan_shared.h
// Include this header AFTER those bindings are declared.

#ifndef THREEPP_VULKAN_SHADE_COMMON_GLSL
#define THREEPP_VULKAN_SHADE_COMMON_GLSL

const float PI = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

// ── RNG ────────────────────────────────────────────────────────────────────
// PCG XSH-RR. Cheap, well-mixed, single-uint state — perfect for shaders.
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float urand(inout uint state) {
    return float(pcgNext(state)) / 4294967296.0;
}

// ── Misc utilities ────────────────────────────────────────────────────────
float lum3(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// 2D→1D hash. Used to break up bilinear-interpolated per-vertex foam into
// crisp speckle (matches the multi-octave hash noise in the WGPU webtide
// raster shader).
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2D value noise: smoothstep-interpolated cell hash. Cheap Perlin
// substitute — gives spatial coherence the raw hash lacks. Cost: 4
// hashes + smoothstep + 3 mixes per sample.
float vnoise21(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 4-octave fBm in [0..1] (approximately, given amplitude geometric sum).
// Used by the foam shading block to give the macro "pool" mask coherent
// large-scale structure instead of pure speckle.
float fbm4(vec2 p) {
    float a = 0.0;
    float w = 0.5;
    for (int i = 0; i < 4; ++i) {
        a += w * vnoise21(p);
        p *= 2.03;     // slight off-2 to break grid alignment
        w *= 0.5;
    }
    return a;
}

// ── BSDF math ──────────────────────────────────────────────────────────────
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

// ── Sampling PDFs (used by NEE MIS) ────────────────────────────────────────
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

// ── Sheen (KHR_materials_sheen) ────────────────────────────────────────────
// Charlie NDF, Neubelt visibility, and IBL energy approximation.
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

// ── Env sampling (binding-dependent: needs `envTex`) ───────────────────────
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

// ── Fog (binding-dependent: needs `fog`) ───────────────────────────────────
// Beer-Lambert transmittance over a finite ray segment. Distance is clamped
// to 1e6 to avoid overflow on directional / env "infinite ray" sentinels —
// callers that pass maxDist >= 1e20 should skip fog attenuation entirely
// (those sources sit "outside" the fog volume, i.e. sun through atmosphere).
bool fogEnabled() { return fog.enabled > 0.5; }
vec3 fogTransmittance(float dist) {
    const float d = clamp(dist, 0.0, 1e6);
    return exp(-fog.sigmaT * d);
}

// ── Photon gather (binding-dependent: needs `photonCounts`, `photonData`) ──
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

// ── Env CDF importance sampling ────────────────────────────────────────────
// Binding-dependent: needs `envCdfTex`, `envMargTex`, `envTex`, and a `pc`
// push constant with envCdfTotalSum / envCdfWidth / envCdfHeight fields
// (chit + raygen both have those — see VulkanRenderer.cpp pc layout).

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

// ── VNDF microfacet sampling ───────────────────────────────────────────────
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

// ── Multi-lobe BSDF sampler ────────────────────────────────────────────────
// Used by env-NEE-fallback and the indirect bounce. Picks between clearcoat /
// base-spec / base-diff via the supplied probabilities, samples a direction
// from the chosen lobe, and returns the throughput weight (already divided
// by the lobe-pick pdf).
//
// `valid == false` means the sampled direction landed at or below the
// surface (rare numerical edge case); callers either skip the trace
// (env-NEE) or terminate the path (bounce).
//
// Both call sites must use this so MIS pdf agreement (brdfPdf3) holds —
// any divergence in lobe weights or urand call order would over-/under-
// count contributions where env-NEE and BSDF-sampled bounce overlap.
struct BsdfSample {
    vec3  dir;
    vec3  weight;
    bool  valid;
};

BsdfSample sampleBsdf(
        vec3  V,         vec3  N,        vec3  F0,         vec3  albedo,
        float alpha,     float ccAlpha,  float metalness,  float NdotV,
        float pSpec,     float ccProb,   float ccWeight,
        float baseScale, float invBaseProb,
        inout uint seed) {
    BsdfSample r;
    r.valid = true;
    const float xiCC = urand(seed);
    if (ccProb > 0.0 && xiCC < ccProb) {
        const vec2 u2 = vec2(urand(seed), urand(seed));
        r.dir = sampleVNDF(V, N, ccAlpha, u2);
        const float NdotL = dot(N, r.dir);
        if (NdotL <= 0.0) {
            r.valid  = false;
            r.weight = vec3(0.0);
            return r;
        }
        const float G1L = smithG1(NdotL, ccAlpha);
        r.weight = vec3(ccWeight * G1L / ccProb);
    } else {
        const float xi = urand(seed);
        if (xi < pSpec) {
            const vec2 u2 = vec2(urand(seed), urand(seed));
            r.dir = sampleVNDF(V, N, alpha, u2);
            const float NdotL = dot(N, r.dir);
            if (NdotL <= 0.0) {
                r.valid  = false;
                r.weight = vec3(0.0);
                return r;
            }
            const vec3  H     = normalize(V + r.dir);
            const float VdotH = max(0.0, dot(V, H));
            const vec3  F     = fresnelSchlick(VdotH, F0);
            const float G1L   = smithG1(NdotL, alpha);
            r.weight = F * G1L * baseScale * invBaseProb / pSpec;
        } else {
            const vec2 u2 = vec2(urand(seed), urand(seed));
            const vec3 localDir = cosineHemisphere(u2);
            const mat3 tbn = makeTBN(N);
            r.dir = normalize(tbn * localDir);
            r.weight = albedo * (1.0 - metalness)
                     * kcBoost(F0, NdotV, alpha)
                     * baseScale * invBaseProb / (1.0 - pSpec);
        }
    }
    return r;
}

// ── Env NEE + MIS for opaque surfaces ─────────────────────────────────────
// Importance-samples the env by luminance when an env CDF is bound, otherwise
// falls back to BSDF-sampled env NEE with constant 0.5 MIS. Fires a shadow
// ray through topAS and returns the env contribution to add into the
// caller's `lit` accumulator.
//
// Required globals (must be declared by the includer):
//   - topAS (binding 0)
//   - shadowVisibility (rayPayloadEXT at location 1)
//   - envTex, envCdfTex, envMargTex
//   - pc with envCdfTotalSum + fireflyClamp fields
//
// Glass / transmission-dominant primaries should skip this entirely — they
// have their own refraction handling that subsumes env contribution.
vec3 envNeeOpaque(
        vec3  V,         vec3  N,        vec3  hitPos,
        vec3  F0,        vec3  albedo,
        float roughness, float metalness, float alpha,
        float NdotV,     float k,
        float baseScale, float invBaseProb, float pSpec,
        float ccProb,    float ccRough,    float ccWeight, float ccAlpha,
        inout uint seed) {
    vec3 contribution = vec3(0.0);
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
                // Post-multiply firefly clamp. The env-sample clamp above caps
                // the env texel's luminance, but on glossy surfaces the BSDF
                // lobe (`lobeSum`) can spike — `D_e = distGGX` at a near-mirror
                // configuration easily hits 1000+ — and when the env CDF picks
                // a low-pdf direction near a bright HDR sun disc, the ratio
                // `lobeSum / pdfEnv` blows up regardless of the per-sample env
                // clamp. The product becomes a visible sky-coloured firefly on
                // glossy/painted surfaces under sunlit HDRI scenes (the long-
                // standing speckle issue on the Ocean demo's ship hull).
                // Matches the post-multiply clamp pattern used in the ReSTIR DI
                // path and the emissive-tri NEE path. Gated by the `< 1e20`
                // sentinel so setFireflyClamp(0) disables it.
                vec3 envContrib = wEnv * lobeSum * NdotL_e * envSample
                                  * shadowVisibility / max(pdfEnv, 1e-8);
                if (pc.fireflyClamp < 1e20) {
                    const float cLumE = lum3(envContrib);
                    if (cLumE > pc.fireflyClamp) envContrib *= pc.fireflyClamp / cLumE;
                }
                contribution = envContrib;
            }
        }
    } else {
        // Fallback: BSDF-sampled env NEE with constant 0.5 MIS (no CDF). The
        // sampler shares its lobe-pick logic with the indirect bounce below
        // so pdfs match and the balance-heuristic weight collapses to 0.5/0.5.
        const BsdfSample s = sampleBsdf(V, N, F0, albedo, alpha, ccAlpha,
                                        metalness, NdotV, pSpec, ccProb,
                                        ccWeight, baseScale, invBaseProb, seed);
        if (s.valid) {
            shadowVisibility = 1.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT |
                        gl_RayFlagsNoOpaqueEXT,
                        0xff, 1, 0, 1,
                        hitPos + N * 1e-3, 0.0, s.dir, 1e30, 1);
            if (shadowVisibility > 0.0) {
                vec3 envSample = sampleEquirect(s.dir);
                const float envLum = dot(envSample, vec3(0.2126, 0.7152, 0.0722));
                if (envLum > pc.fireflyClamp) envSample *= pc.fireflyClamp / envLum;
                // Post-multiply clamp — same reasoning as the env-CDF branch
                // above. `s.weight = BRDF·cos / pdf_lobe / p_lobeSel` can spike
                // on glossy surfaces when the lobe pdf was small. Without this,
                // bright env values reflected off near-mirror surfaces produce
                // visible fireflies.
                vec3 envContrib = 0.5 * s.weight * envSample * shadowVisibility;
                if (pc.fireflyClamp < 1e20) {
                    const float cLumE = lum3(envContrib);
                    if (cLumE > pc.fireflyClamp) envContrib *= pc.fireflyClamp / cLumE;
                }
                contribution = envContrib;
            }
        }
    }
    return contribution;
}

// ── Classic per-light + emissive-tri NEE for opaque surfaces ──────────────
// Fires one shadow ray per light (dir / point / spot / rect) and one for a
// power-CDF-picked emissive triangle, evaluating Cook-Torrance + Lambert
// (+ optional clearcoat / sheen) at each. Mirrors the chit's pre-ReSTIR-DI
// behaviour; the ReSTIR DI path replaces this with RIS-sampled single
// shadow ray.
//
// Required globals (must be declared by the includer):
//   - topAS (binding 0)
//   - shadowVisibility (rayPayloadEXT at location 1)
//   - lights (binding 5)
//   - emissiveTris (binding 14)
//   - fog (binding 17) — fogEnabled() / fogTransmittance() use it
//   - pc with fireflyClamp + emissiveCount + emissiveTotalPower fields
//
// Glass / transmission primaries should skip this — they have their own
// refraction handling. Bounce hits (bsdfOnlyMode) should also skip.
vec3 analyticNeeOpaque(
        vec3  V,         vec3  N,        vec3  hitPos,
        vec3  F0,        vec3  albedo,
        float roughness, float metalness, float alpha,
        float NdotV,     float k,
        float baseScale,
        float sheenScaling, bool hasSheen, vec3 sheenColor, float sheenRoughness,
        float ccWeight,  float ccRough,
        inout uint seed) {
    vec3 lit = vec3(0.0);

    // === Directional lights ===
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
                    0xff, 1, 0, 1,
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
            perLight += sheenColor * D_Charlie(NdotH, sheenRoughness)
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
            perPt += sheenColor * D_Charlie(NdotH, sheenRoughness)
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
            perSp += sheenColor * D_Charlie(NdotH, sheenRoughness)
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
    // sampleEmissiveTriCdf. The next BSDF bounce's MIS partner uses
    // payload.inFlags bit 0 (set by raygen) so emission isn't double-counted.
    if (pc.emissiveCount > 0u && pc.emissiveTotalPower > 0.0) {
        // Binary search the cumulative-power CDF (stored in v1.w of each tri).
        // Iteration count tightened to ceil(log2(emissiveCount)).
        const float xi = urand(seed) * pc.emissiveTotalPower;
        uint lo = 0u;
        uint hi = pc.emissiveCount - 1u;
        const int emIters = findMSB(max(pc.emissiveCount - 1u, 1u)) + 1;
        for (int s = 0; s < emIters; ++s) {
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
            const vec3 ge1 = t.v1.xyz - t.v0.xyz;
            const vec3 ge2 = t.v2.xyz - t.v0.xyz;
            const vec3 lnRaw = cross(ge1, ge2);
            const float lnLen = length(lnRaw);
            // Reject grazing receiver / emitter angles — small pdfOmega and
            // nearly-zero cosLe in the denominator otherwise produce
            // arbitrarily-large NEE samples that leak fireflies through MIS.
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
                        // w_light → 0 (NEE contributes nothing). On rough/diffuse
                        // pdfOmega dominates → w_light → 1.
                        const float pdfBsdfNee = brdfPdf(V, toL, N, roughness, metalness);
                        const float wLight = pdfOmega / max(pdfOmega + pdfBsdfNee, 1e-8);
                        vec3 emContrib = perEm * NdotLe * emCol * wLight / max(pdfOmega, 1e-8);
                        const float emCLum = dot(emContrib, vec3(0.2126, 0.7152, 0.0722));
                        if (emCLum > pc.fireflyClamp) emContrib *= pc.fireflyClamp / emCLum;
                        const vec3 fogAttenEm = fogEnabled() ? fogTransmittance(dist) : vec3(1.0);
                        lit += emContrib * shadowVisibility * fogAttenEm;
                    }
                }
            }
        }
    }

    return lit;
}

#endif // THREEPP_VULKAN_SHADE_COMMON_GLSL
