// Emitter LIGHT TABLE + the sample PLAN shared by every coherent emitter-NEE loop
// (emissiveNEE, emissiveSpecNEE, the two emissiveIrradiance copies).
//
// WHY. The coherent stratified power-CDF pick is noise-free, but a handful of strata
// cannot RESOLVE several small emitters. The sailboat's port/starboard nav lights are
// ~1-3 % of the emitter power each: at a reflected hit (4 strata) at most ONE stratum
// lands on them, and which light it lands on moves whenever the TOTAL power moves
// (the lighthouse lamp's 5 % glow wobble) — the reflected sail flipped red ↔ green a
// few times a second ("a pulsating light that isn't") and, with other power splits,
// simply lost both lights. The primary surface never showed it: ReSTIR DI samples
// emitters per pixel and keeps reservoirs; reflected / GI / probe hits have no
// reservoir and rely on this pick.
//
// FIX — COVERAGE mode. With ≤ kEmissiveCoverMaxLights emissive instances every light
// gets ≥ 1 sample and the remaining strata go ∝ power among the lights that still need
// AREA sampling. A light that is SMALL from the shading point (bounding radius ≤
// kEmProxyMaxRatio × distance) is shaded as ONE point at its centre with a direction-
// dependent projected area: exact for a sphere (irradiance Le·πr²/d²), A/4 (Cauchy's
// mean) for any closed shell, A·cosθ for a one-sided panel — no triangle pick, so
// nothing can flip and no far-side rejection. Bigger / closer lights keep the
// stratified triangle sampling, confined to that light's own CDF range (a lone big
// panel at 16 strata is bit-identical to before). Many-light scenes (Bistro: hundreds
// of emissive instances) keep the global pick untouched. Deterministic and pixel-
// coherent throughout — still zero temporal noise. Cost: a loop of ≤ max(EM, L)
// samples, so a caller asking for 2 (GI / probe) may trace up to 8.
//
// BUFFER LAYOUT (host: buildAndUploadEmissiveTris):
//   [0, N)        EmTri triangles, running power CDF in v1.w      (N = pc.emissiveCount)
//   [N]           header: v0.x = light count L (float-int), v0.y = gateK (the range
//                 gate's exposure term, 0 = gate off); v1.xyzw, v2.xyzw = the TLAS
//                 instanceCustomIndex of lights 0..7 (float-ints, -1 = none), written
//                 when L <= kEmissiveCoverMaxLights (emLightOfInstance); rest 0
//   [N+1, N+1+L)  EmLight records, written only when L ≤ kEmissiveGateMaxLights (the
//                 coverage mode reads them only when L ≤ kEmissiveCoverMaxLights):
//                   v0.xyz centre, v0.w bounding radius
//                   v1.xyz Σ area·n̂ (world; ≈ 0 for a closed shell), v1.w CDF start
//                   v2.x triBegin, v2.y triCount (float-ints), v2.z area, v2.w power
//                   emission.rgb Le, emission.w gate = max(Le)·area
// Needs, from the including shader: EmTri emissiveTris[], pc.emissiveCount,
// pc.emissiveTotalPower. Callers check pc.emissiveCount > 0 before planning.

#define kEmProxyMaxRatio 0.25// bounding radius / distance at or below which a light is a point
// ...and only when the light is TESSELLATED (spheres, cylinders, lamp meshes). The
// proxy's closed-shell term is A/4 (exact for a sphere, ~20 % off for a cylinder, up
// to 50 % off face-on for a box), so a coarse box-like emitter keeps its area strata
// instead: a lone 12-tri emissive cube renders bit-identically to the global pick.
#define kEmProxyMinTris 32u
// A light's share of the spare strata, extra * power / areaPower, is an integer
// for a lone area light (extra * P / P) and for lights of equal power (extra / k),
// and the float quotient is that integer or one ulp under it depending on the
// bit pattern of the power. The host sums a moving emitter's power from its
// world-space triangle areas every frame, so the pattern changes with the pose:
// a plain floor() gave a lone rotating torus knot 7 strata instead of 8 on 17
// of 240 frames (the frames where float32 7 * P / P < 7), and every lighting
// point moved for that frame. A share within kEmShareSnap below an integer
// counts as that integer. The shares still sum to at most `extra`:
// kEmissiveCoverMaxLights * kEmShareSnap < 1.
#define kEmShareSnap 0.015625

// WHO DRAWS A LIGHT'S SPECULAR: the strata here, or the reflection ray.
// The coherent strata stand in for a light with n points. Through a lobe that is
// wide against the gaps between them that is a noise-free quadrature; through a
// lobe that is narrow against them it is n separate highlights where the light's
// mirrored image belongs (an emissive torus knot 1.1 m over a roughness-0.22
// floor: three or four blobs). The reflection ray draws that image, and it hits a
// light that fills its lobe on most samples, so it has none of the spike train
// that made the strata the owner of a SMALL lamp's specular (emissiveSpecNEE).
// Per light, from the shaded point, with y = (the light's solid angle) /
// (pi a2), the lobe's solid angle taken as 1 / D(n):
//   spacing  y / n   the solid angle one stratum stands for, against the lobe
//   hit      y       a GGX lobe sample hits the light with chance about y / (4 + y)
//                    (the half-vector cone of a light of half-angle t is t / 2, and
//                    GGX's cdf in it is tan2 / (a2 + tan2))
// The ray's share of light l is smoothstep(spacing) * smoothstep(hit): the strata
// must be too sparse AND the ray must hit. A stochastic lobe draws a bright light
// it misses on 1 sample in 3 as grain the filter does not take out (rg_ab's room:
// its 3 m panel of Le 18 on the roughness-0.25 gold sphere, y = 7.7, when the
// ramp ended at y = 3), so the ramp runs from 75 % of samples hitting to 89 %;
// below it the strata keep the light. (The floor of VulkanEmitterSpecular_test
// is clean from y = 40 up, 8 % misses; a ramp to y = 96 brought its strata
// blobs back on the near floor.) The strata keep the rest
// (emissiveSpecNEE), and the reflection hit scales the emitter it hit by the
// same number (emLightOfInstance, gReflEmitterRayShare), so the two add to one.
// Coverage mode only (L <= kEmissiveCoverMaxLights); elsewhere the share is 0,
// and so is it where the ray cannot hold the image (deferred_shade.comp,
// rayShareScale): in the STOCHASTIC band (roughness 0.25 and up,
// kReflStochastic) the ray is a 1-spp lobe sample; on a MOVING mesh
// (kInstFlagMoving) the reflection history is of other normals; and on a
// CURVED surface (specAaKernelSq) the mirror direction turns from pixel to
// pixel and one ray through the gloss blur is patches. A 3 m panel mirrored
// in a roughness-0.15 metal knot: turning, a jagged patch; at rest, scattered
// patches; the strata draw a smooth band in both. The hand-off is for the
// flat glossy floor under a light, where the strata drew their blobs.
#define kEmRaySpacing0 0.1
#define kEmRaySpacing1 0.5
#define kEmRayHit0     12.0
#define kEmRayHit1     32.0
#define kEmNoLight     0xFFFFFFFFu

struct EmPlan {
    int  count;                        // samples to take (loop bound)
    bool cover;                        // coverage mode (else the global stratified pick)
    uint L;                            // lights in the table (coverage mode)
    uint proxyBits;                    // bit l: light l is a point proxy from this P
    uint n[kEmissiveCoverMaxLights];   // strata per light (coverage mode)
};

struct EmSample {
    vec3  lp;      // lighting point (world)
    vec3  L;       // unit direction P → lp
    float dist;    // |lp − P|
    float backoff; // shorten the shadow ray toward lp by this (proxy: the light's radius)
    vec3  Le;      // emission radiance
    float gw;      // geometry ÷ pdf: (area·cosLight)/(pickPdf·dist²), or A_proj/dist² for a proxy
    float w;       // plan weight: 1/strata-of-this-light, or 1/EM in global mode
    uint  light;   // index in the light table (coverage mode), else kEmNoLight
};

uint emLightCount() { return uint(emissiveTris[pc.emissiveCount].v0.x + 0.5); }

// RANGE gate. A lamp that is on costs the frame its sampling at EVERY shaded point,
// lit by it or not: a fishing boat's lanterns and twelve cars' tail lamps were 3 ms
// of a 19 ms daylight frame on a site they light nothing of. Light i can give a
// surface at P at most the irradiance max(Le)·A/d² (d from its bounding sphere), and
// the host knows the exposure: under gateK·gate_i/d² = 1 the light adds less than its
// display threshold (buildAndUploadEmissiveTris) to a white surface that faces it.
// Returns the largest such ratio over the lights; below 1 no emitter is worth a
// sample at P and the caller returns black. No table or the gate off: always above 1.
float emGateBound(vec3 P) {
    const uint  L = emLightCount();
    const float k = emissiveTris[pc.emissiveCount].v0.y;
    if (k <= 0.0 || L == 0u || L > uint(kEmissiveGateMaxLights)) return 1e30;
    float m = 0.0;
    for (uint l = 0u; l < L; ++l) {
        const vec4  cr = emissiveTris[pc.emissiveCount + 1u + l].v0;
        const float d  = max(length(cr.xyz - P) - cr.w, 0.0);
        m = max(m, emissiveTris[pc.emissiveCount + 1u + l].emission.w / max(d * d, 1e-6));
    }
    return m * k;
}

EmPlan emPlanBuild(vec3 P, int EM) {
    EmPlan pl;
    pl.count = EM; pl.cover = false; pl.L = 0u; pl.proxyBits = 0u;
    for (int i = 0; i < kEmissiveCoverMaxLights; ++i) pl.n[i] = 0u;
    const uint L = emLightCount();
    if (L == 0u || L > uint(kEmissiveCoverMaxLights)) return pl;// global pick
    pl.cover = true; pl.L = L;
    float pw[kEmissiveCoverMaxLights];
    float areaPow = 0.0;// power of the lights that still need AREA sampling
    for (uint l = 0u; l < L; ++l) {
        const EmTri r = emissiveTris[pc.emissiveCount + 1u + l];
        pw[l] = r.v2.w;
        const float d = length(r.v0.xyz - P);
        const bool proxy = (r.v0.w <= kEmProxyMaxRatio * d) && (uint(r.v2.y + 0.5) >= kEmProxyMinTris);
        if (proxy) pl.proxyBits |= (1u << l); else areaPow += r.v2.w;
        pl.n[l] = 1u;// every light gets at least one
    }
    const int extra = max(EM - int(L), 0);
    if (extra > 0 && areaPow > 0.0) {
        for (uint l = 0u; l < L; ++l) {
            if ((pl.proxyBits & (1u << l)) != 0u) continue;// a point needs no more strata
            pl.n[l] += uint(floor(float(extra) * pw[l] / areaPow + kEmShareSnap));
        }
    }
    int cnt = 0;
    for (uint l = 0u; l < L; ++l) cnt += int(pl.n[l]);
    pl.count = cnt;
    return pl;
}

// Sample s of the plan. false = contributes nothing (degenerate / grazing / emitter
// facing away from P). `twoSided`: accept emitter triangles facing away (the GI
// irradiance convention); the diffuse/spec NEE use front-facing only.
bool emPlanSample(in EmPlan pl, int s, vec3 P, bool twoSided, out EmSample es) {
    uint lo, hi;
    float xi, pickDen;// CDF search range, stratum position, pick-pdf normaliser
    es.w = 1.0;
    es.light = kEmNoLight;
    if (!pl.cover) {
        lo = 0u; hi = pc.emissiveCount - 1u;
        xi = (float(s) + 0.5) / float(pl.count) * pc.emissiveTotalPower;
        pickDen = pc.emissiveTotalPower;
        es.w = 1.0 / float(pl.count);
    } else {
        uint l = 0u; int base = 0;// which light owns stratum s
        for (; l < pl.L; ++l) {
            if (s < base + int(pl.n[l])) break;
            base += int(pl.n[l]);
        }
        if (l >= pl.L) return false;
        es.light = l;
        const EmTri r = emissiveTris[pc.emissiveCount + 1u + l];
        if ((pl.proxyBits & (1u << l)) != 0u) {
            // POINT PROXY: the whole light from its centre, projected area by shape.
            const vec3  toC = r.v0.xyz - P;
            const float d2  = dot(toC, toC);
            const float d   = sqrt(max(d2, 1e-20));
            if (d <= 1e-4) return false;
            es.lp = r.v0.xyz; es.L = toC / d; es.dist = d; es.backoff = r.v0.w; es.Le = r.emission.rgb;
            const float nLen    = length(r.v1.xyz);// |Σ area·n̂| = the one-sided (flat) part
            const float flatCos = nLen > 1e-12 ? dot(r.v1.xyz / nLen, -es.L) : 0.0;
            const float aFlat   = nLen * (twoSided ? abs(flatCos) : max(flatCos, 0.0));
            const float aProj   = aFlat + 0.25 * max(r.v2.z - nLen, 0.0);// + the closed part (A/4)
            if (aProj <= 0.0) return false;
            es.gw = aProj / d2; es.w = 1.0;
            return true;
        }
        const uint triBegin = uint(r.v2.x + 0.5), triCount = uint(r.v2.y + 0.5);
        if (triCount == 0u || r.v2.w <= 0.0) return false;
        lo = triBegin; hi = triBegin + triCount - 1u;
        xi = r.v1.w + (float(s - base) + 0.5) / float(pl.n[l]) * r.v2.w;
        pickDen = r.v2.w;
        es.w = 1.0 / float(pl.n[l]);
    }
    // Coherent stratified power-CDF pick within [lo, hi] (binary search on the running CDF).
    const int iters = findMSB(max(hi - lo, 1u)) + 1;
    for (int it = 0; it < iters; ++it) {
        if (lo >= hi) break;
        const uint mid = (lo + hi) >> 1u;
        if (emissiveTris[mid].v1.w < xi) lo = mid + 1u; else hi = mid;
    }
    const EmTri t = emissiveTris[lo];
    // Coherent R2 barycentric → the SAME lighting point at every pixel (smooth irradiance).
    const float su1 = sqrt(fract(0.5 + float(s) * 0.7548776662));
    const float r2  =      fract(0.5 + float(s) * 0.5698402909);
    es.lp = (1.0 - su1) * t.v0.xyz + (su1 * (1.0 - r2)) * t.v1.xyz + (su1 * r2) * t.v2.xyz;
    const vec3  toL   = es.lp - P;
    const float dist2 = dot(toL, toL);
    es.dist = sqrt(max(dist2, 1e-20));
    if (es.dist <= 1e-4) return false;
    es.L = toL / es.dist; es.backoff = 0.0; es.Le = t.emission.rgb;
    const vec3  lnRaw = cross(t.v1.xyz - t.v0.xyz, t.v2.xyz - t.v0.xyz);
    const float lnLen = length(lnRaw);
    if (lnLen < 1e-20 || t.v0.w <= 1e-20 || t.v2.w <= 0.0) return false;
    // FRONT-FACING only (unless twoSided): the emitter's far side isn't visible from P;
    // a shadow ray to it would pass through the near side → a fake self-occlusion
    // "fence" across lit surfaces. Also rejects grazing → firefly source.
    float cosLight = dot(-es.L, lnRaw / lnLen);
    if (twoSided) cosLight = abs(cosLight);
    if (cosLight <= 0.05) return false;
    const float pickPdf = t.v2.w / pickDen;
    const float p_omega = pickPdf * dist2 / (t.v0.w * cosLight);
    if (p_omega <= 1e-20) return false;
    es.gw = 1.0 / p_omega;
    return true;
}

// The reflection ray's share of light l's specular at P (see kEmRaySpacing0).
// The solid angle is the point proxy's projected area over d^2: a one-sided
// light by its facing, a closed shell by A/4.
float emSpecRayShare(in EmPlan pl, uint l, vec3 P, float roughness) {
#ifdef THREEPP_NO_RAY_QUERY
    return 0.0;// no ray hits anything: the strata are the light's only specular
#else
    if (!pl.cover) return 0.0;
    const EmTri r = emissiveTris[pc.emissiveCount + 1u + l];
    const vec3  toC = r.v0.xyz - P;
    const float d2  = max(dot(toC, toC), 1e-8);
    const float nLen    = length(r.v1.xyz);
    const float flatCos = nLen > 1e-12 ? dot(r.v1.xyz / nLen, -toC * inversesqrt(d2)) : 0.0;
    const float aProj   = nLen * max(flatCos, 0.0) + 0.25 * max(r.v2.z - nLen, 0.0);
    const float omega   = min(aProj / d2, 6.2831853);
    const float a       = roughness * roughness;
    const float y       = omega / (3.14159265 * max(a * a, 1e-8));
    return smoothstep(kEmRaySpacing0, kEmRaySpacing1, y / float(max(pl.n[l], 1u)))
         * smoothstep(kEmRayHit0, kEmRayHit1, y);
#endif
}

// Every light's share in 8 bits, lights 0..3 in .x and 4..7 in .y: the strata
// and the reflection hit read the same quantised number.
uvec2 emSpecRayShares(in EmPlan pl, vec3 P, float roughness, float scale) {
    uvec2 bits = uvec2(0u);
    if (!pl.cover) return bits;
    for (uint l = 0u; l < pl.L; ++l) {
        const uint q = uint(emSpecRayShare(pl, l, P, roughness) * scale * 255.0 + 0.5);
        bits[l >> 2u] |= q << (8u * (l & 3u));
    }
    return bits;
}

float emRayShareOf(uvec2 bits, uint l) {
    return float((bits[l >> 2u] >> (8u * (l & 3u))) & 255u) * (1.0 / 255.0);
}

// The light a TLAS instance is, or kEmNoLight (the header's slots).
uint emLightOfInstance(int instanceId) {
    const uint L = emLightCount();
    if (L == 0u || L > uint(kEmissiveCoverMaxLights)) return kEmNoLight;
    const vec4 lo = emissiveTris[pc.emissiveCount].v1;
    const vec4 hi = emissiveTris[pc.emissiveCount].v2;
    for (uint l = 0u; l < L; ++l) {
        const float e = l < 4u ? lo[l] : hi[l - 4u];
        if (e >= 0.0 && int(e + 0.5) == instanceId) return l;
    }
    return kEmNoLight;
}
