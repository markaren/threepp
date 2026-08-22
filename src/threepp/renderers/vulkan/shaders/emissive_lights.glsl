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
//   [N]           header: v0.x = light count L (float-int); rest 0
//   [N+1, N+1+L)  EmLight records, written only when L ≤ kEmissiveCoverMaxLights:
//                   v0.xyz centre, v0.w bounding radius
//                   v1.xyz Σ area·n̂ (world; ≈ 0 for a closed shell), v1.w CDF start
//                   v2.x triBegin, v2.y triCount (float-ints), v2.z area, v2.w power
//                   emission.rgb Le
// Needs, from the including shader: EmTri emissiveTris[], pc.emissiveCount,
// pc.emissiveTotalPower. Callers check pc.emissiveCount > 0 before planning.

#define kEmProxyMaxRatio 0.25// bounding radius / distance at or below which a light is a point
// ...and only when the light is TESSELLATED (spheres, cylinders, lamp meshes). The
// proxy's closed-shell term is A/4 (exact for a sphere, ~20 % off for a cylinder, up
// to 50 % off face-on for a box), so a coarse box-like emitter keeps its area strata
// instead: a lone 12-tri emissive cube renders bit-identically to the global pick.
#define kEmProxyMinTris 32u

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
};

uint emLightCount() { return uint(emissiveTris[pc.emissiveCount].v0.x + 0.5); }

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
            pl.n[l] += uint(floor(float(extra) * pw[l] / areaPow));
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
