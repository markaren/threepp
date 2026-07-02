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

// Trilinear-interpolated indirect irradiance at surface point P with normal N.
// Leak handling (no depth probes yet — DDGI-lite):
//   • normal-offset bias: the lookup point moves 0.25·min(spacing) off the
//     surface so a wall pixel doesn't straddle the cell boundary into probes
//     behind the wall,
//   • smooth backface weight: probes BEHIND the surface's tangent plane
//     (dot(toProbe, N) < 0) fade out quadratically — they see the wrong side,
//   • validity: probes whose update rays mostly hit interior backfaces are
//     inside geometry; their stored (black) SH is excluded.
// All-invalid neighbourhoods return vec3(0) — the caller's direct terms stand.
//
// `conf` (out) = valid-weight fraction: the share of the GEOMETRICALLY relevant
// trilinear weight (position × backface) carried by VALID probes. 1 = the full
// neighbourhood measured something; → 0 where the surrounding probes sit inside
// geometry (thin shells, recessed cavities, models much smaller than a grid
// cell) and the interpolation would be black not because the point is dark but
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
        const float wg   = w3.x * w3.y * w3.z * (ndp * ndp);// position × backface
        wGeo += wg;
        const float w    = wg * probeSh[pIdx * 4u + 0u].a;  // × validity
        if (w <= 1e-5) continue;
        sum  += w * probeShIrradiance(pIdx, N);
        wSum += w;
    }
    conf = (wGeo > 1e-4) ? clamp(wSum / wGeo, 0.0, 1.0) : 0.0;
    return (wSum > 1e-3) ? sum / wSum : vec3(0.0);
}

vec3 probeIrradiance(vec3 P, vec3 N) {
    float conf;
    return probeIrradianceConf(P, N, conf);
}
