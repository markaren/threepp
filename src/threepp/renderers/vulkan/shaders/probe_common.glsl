// World-space irradiance probe grid — shared sampling logic.
//
// Included by deferred_shade.comp (taps the grid at GI-ray hit points) and
// probe_update.comp (the self-feedback tap that makes the cache multi-bounce).
// Both must declare, BEFORE including this file, with matching layouts:
//
//   probeSh[]  — vec4[] storage buffer, 4 vec4 per probe (SH-L1 irradiance):
//                  [0] = L00.rgb,  .a = validity (0 = probe inside geometry)
//                  [1] = L1x.rgb,  .a = update count (EMA history length)
//                  [2] = L1y.rgb
//                  [3] = L1z.rgb
//   probeDepth[] — uint[] storage buffer, kProbeDepthTexels per probe: an 8×8
//                octahedral map of packHalf2x16(mean, mean²) ray-hit distance,
//                NORMALIZED by probeMaxDist() (half-safe at any scene scale).
//                0u = texel never updated (fresh grid) → treated as "all
//                clear" so bootstrap can't reject every probe.
//   probeGrid  — the ProbeGridUbo uniform block:
//                  vec3 origin; float enabled; vec3 spacing; ivec3 dims;
//                origin = the FIRST probe's world position (cell centers:
//                aabbMin + 0.5·spacing), probes at origin + coord·spacing.
//
// ENERGY ACCOUNTING (why nothing double-counts):
//   A probe stores the INCIDENT irradiance from (a) surfaces shaded with
//   direct analytic light + emissive NEE + their own probe-fed indirect —
//   i.e. bounce 1..∞ — and (b) the environment on ray MISSES. It EXCLUDES
//   the emission of emitter surfaces its rays land on and any analytic
//   light arriving at the probe point directly, because every consumer of
//   probeIrradiance(P, N) evaluates those two DIRECT terms at P itself
//   (giRadiance's light loops + emissiveIrradiance; probe_update's copies).
//   So probeIrradiance is exactly "the indirect + sky irradiance at P" and
//   adding  albedo/π · probeIrradiance(P, N)  to a hit's direct shading
//   composes the full outgoing radiance without counting anything twice.

// Irradiance from one probe's SH-L1 in direction n (unit surface normal).
// Ramamoorthi & Hanrahan cosine convolution: Â0·Y00 = π·0.282095 = 0.886227,
// Â1·Y1c = (2π/3)·0.488603 = 1.023327. Clamped ≥ 0 (L1 can ring negative).
vec3 probeShIrradiance(uint pIdx, vec3 n) {
    const vec3 c0 = probeSh[pIdx * 4u + 0u].rgb;
    const vec3 cx = probeSh[pIdx * 4u + 1u].rgb;
    const vec3 cy = probeSh[pIdx * 4u + 2u].rgb;
    const vec3 cz = probeSh[pIdx * 4u + 3u].rgb;
    return max(c0 * 0.8862269 + (cx * n.x + cy * n.y + cz * n.z) * 1.0233267,
               vec3(0.0));
}

// ── Probe depth (visibility) — DDGI Chebyshev test (Majercik et al. 2019) ───
// Each probe carries an 8×8 octahedral map of (mean, mean²) ray-hit distance.
// At sample time the point-to-probe distance is tested against the moments in
// that direction: a point statistically BEHIND what the probe saw (r > mean)
// is weighted down by Chebyshev's inequality — this is what stops irradiance
// from leaking through walls that sit between the point and the probe.

const int   kProbeDepthRes       = 8;   // must match ProbeGI::kDepthRes
const int   kProbeDepthTexels    = kProbeDepthRes * kProbeDepthRes;
const float kProbeDepthSharpness = 50.0;// power-cosine blend kernel (paper value)

// Depth rays saturate here; probes never test points beyond one cell diagonal
// (+ the lookup bias), so 1.5× keeps every real query inside the linear range.
float probeMaxDist() { return 1.5 * length(probeGrid.spacing); }

vec2 octEncode(vec3 v) {// unit dir → [-1,1]² octahedral
    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    if (v.z < 0.0) {
        const vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return v.xy;
}

vec3 octDecode(vec2 e) {// [-1,1]² octahedral → unit dir
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        const vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Texel wrap for the manual bilinear: an octahedral map continues across its
// square edge MIRRORED (the rule DDGI bakes into its 1-texel border gutter —
// we have no gutter, so apply it analytically). Bilinear only ever steps one
// texel out, so single-step handling suffices; both-out = corner → diagonal.
ivec2 octWrapTexel(ivec2 c) {
    const int R = kProbeDepthRes;
    const bool ox = c.x < 0 || c.x >= R;
    const bool oy = c.y < 0 || c.y >= R;
    if (ox && oy) return ivec2((c.x + R) % R, (c.y + R) % R);
    if (ox) return ivec2(clamp(c.x, 0, R - 1), R - 1 - c.y);
    if (oy) return ivec2(R - 1 - c.x, clamp(c.y, 0, R - 1));
    return c;
}

// (mean, mean²) of probe pIdx's ray-hit distance around dirFromProbe, still
// NORMALIZED by probeMaxDist(). Manual 4-tap bilinear with octahedral wrap.
vec2 probeDepthMoments(uint pIdx, vec3 dirFromProbe) {
    const vec2  p    = (octEncode(dirFromProbe) * 0.5 + 0.5) * float(kProbeDepthRes) - 0.5;
    const ivec2 base = ivec2(floor(p));
    const vec2  f    = p - vec2(base);
    vec2 m = vec2(0.0);
    for (int i = 0; i < 4; ++i) {
        const ivec2 o    = ivec2(i & 1, i >> 1);
        const ivec2 t    = octWrapTexel(base + o);
        const uint  bits = probeDepth[pIdx * uint(kProbeDepthTexels) + uint(t.y * kProbeDepthRes + t.x)];
        const vec2  d    = (bits == 0u) ? vec2(1.0) : unpackHalf2x16(bits);// 0u = no data → all clear
        const vec2  w2   = mix(1.0 - f, f, vec2(o));
        m += (w2.x * w2.y) * d;
    }
    return m;
}

// Chebyshev visibility weight of probe pIdx for the (biased) point Pb.
// 1 = the probe saw past the point in this direction; → 0 when everything it
// saw stopped well short (a wall in between). Cubed like the paper to crush
// the estimator's long leaky tail.
float probeChebyshev(uint pIdx, vec3 pPos, vec3 Pb) {
    const vec3  fromProbe = Pb - pPos;
    const float rLen      = length(fromProbe);
    if (rLen <= 1e-4) return 1.0;
    const float maxDist = probeMaxDist();
    const vec2  mm = probeDepthMoments(pIdx, fromProbe / rLen) * vec2(maxDist, maxDist * maxDist);
    // The stored distances saturate at maxDist — clamp the test distance just
    // under it so an unoccluded probe (mean ≈ maxDist) never self-shadows.
    const float r = min(rLen, 0.95 * maxDist);
    if (r <= mm.x) return 1.0;
    const float v = abs(mm.y - mm.x * mm.x);
    const float d = r - mm.x;
    const float cheb = v / (v + d * d);
    return cheb * cheb * cheb;
}

// Trilinear-interpolated indirect irradiance at surface point P with normal N.
// Leak handling:
//   • normal-offset bias: the lookup point moves 0.25·min(spacing) off the
//     surface so a wall pixel doesn't straddle the cell boundary into probes
//     behind the wall,
//   • smooth backface weight: probes BEHIND the surface's tangent plane
//     (dot(toProbe, N) < 0) fade out quadratically — they see the wrong side,
//   • Chebyshev visibility (probeChebyshev): probes whose depth map says the
//     point lies BEHIND geometry they can see fade to ~0 — the DDGI depth
//     test, and the primary wall-leak stop. Combined with the backface term
//     through the RTXGI floor + crush curve so barely-visible probes fade
//     smoothly instead of popping at the threshold,
//   • validity: probes whose update rays mostly hit interior backfaces are
//     inside geometry; their stored (black) SH is excluded.
// All-invalid neighbourhoods return vec3(0) — the caller's direct terms stand.
//
// `conf` (out) = valid-weight fraction: the share of the GEOMETRICALLY relevant
// trilinear weight (position × backface × visibility) carried by VALID probes.
// 1 = the full neighbourhood measured something; → 0 where the surrounding
// probes sit inside geometry (thin shells, recessed cavities, models much
// smaller than a grid cell) or none of them can SEE the point (deep crevices)
// and the interpolation would be black not because the point is dark but
// because nothing measured it. Callers that have a cheaper fallback fill (env /
// ambient) should blend by conf instead of trusting a starved neighbourhood.
vec3 probeIrradianceConf(vec3 P, vec3 N, out float conf) {
    conf = 0.0;
    if (probeGrid.enabled < 0.5) return vec3(0.0);
    const float minSp = min(probeGrid.spacing.x,
                            min(probeGrid.spacing.y, probeGrid.spacing.z));
    const vec3 Pb = P + N * (0.25 * minSp);
    vec3 g = (Pb - probeGrid.origin) / probeGrid.spacing;
    g = clamp(g, vec3(0.0), vec3(probeGrid.dims) - 1.0);
    const ivec3 base = min(ivec3(g), probeGrid.dims - 2);
    const vec3  t    = clamp(g - vec3(base), 0.0, 1.0);

    vec3  sum  = vec3(0.0);
    float wSum = 0.0;
    float wGeo = 0.0;// geometric weight ignoring validity — the conf denominator
    for (int c = 0; c < 8; ++c) {
        const ivec3 o    = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        const ivec3 pi   = base + o;
        const vec3  w3   = mix(1.0 - t, t, vec3(o));
        const uint  pIdx = uint(pi.x + probeGrid.dims.x * (pi.y + probeGrid.dims.y * pi.z));
        const vec3  pPos = probeGrid.origin + vec3(pi) * probeGrid.spacing;
        vec3 toP = pPos - P;
        const float dLen = length(toP);
        toP = (dLen > 1e-4) ? toP / dLen : N;
        const float ndp  = clamp(dot(toP, N) * 0.5 + 0.5, 0.0, 1.0);
        // Shading weight = backface × Chebyshev visibility, floored (the
        // least-occluded probe should win over pure black) and crushed below
        // 0.2 (RTXGI curve: w²/0.2² — continuous fade instead of a hard cut).
        float ws = (ndp * ndp) * probeChebyshev(pIdx, pPos, Pb);
        ws = max(ws, 1e-4);
        if (ws < 0.2) ws *= ws * ws * (1.0 / (0.2 * 0.2));
        const float wg   = w3.x * w3.y * w3.z * ws;// position × shading weight
        wGeo += wg;
        const float w    = wg * probeSh[pIdx * 4u + 0u].a;  // × validity
        if (w <= 1e-5) continue;
        sum  += w * probeShIrradiance(pIdx, N);
        wSum += w;
    }
    // conf gates on BOTH guards: a neighbourhood too occluded to return real
    // irradiance (wSum ≤ 1e-3 → vec3(0) below) must read as unmeasured, not
    // as a trusted black.
    conf = (wSum > 1e-3 && wGeo > 1e-4) ? clamp(wSum / wGeo, 0.0, 1.0) : 0.0;
    return (wSum > 1e-3) ? sum / wSum : vec3(0.0);
}

vec3 probeIrradiance(vec3 P, vec3 N) {
    float conf;
    return probeIrradianceConf(P, N, conf);
}
