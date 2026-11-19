// THE SHORE'S DEPTH, and what it does to the waves drawn over it.
//
// One buffer an ocean (WakeFieldPipeline::State::shoreDepth), next to the
// shore's maps: for each of the shore's two windows the water's depth at every
// texel, laid when that window's map is taken (shore_jfa.comp, mode 2), with
// the window's placement beside it; and the drawn sea's statistics, each
// cascade's significant height and mean wavenumber, measured from the
// cascades themselves every step (shore_source.comp, mode 1). The buffer
// carries its own placement, so whoever reads it (water_displace.comp per
// vertex, the deferred shade per pixel, the shore's own source) reads a depth
// and the window it was laid in together, whenever in the frame it reads.
//
// What the depth does: the waves drawn over shallow water are the deep sea's
// cascades, each scaled at a parcel's rest point (as the wave climate scales
// them, DisplacedMesh::WaveClimate):
//
//   SHOALING. A wave whose length is over about twice the depth feels the bed:
//   its group slows and, its energy flux kept, its height changes by
//   Ks = [tanh(kd) (1 + 2kd / sinh(2kd))]^(-1/2) (linear theory; Green's law,
//   Ks ~ d^(-1/4), in the shallow limit): a dip of about 9 % near kd = 1.3,
//   then a rise. Taken at each cascade's mean wavenumber and UNDERSTATED: half
//   of it, and never more than kShoalMax.
//
//   THE DEPTH LIMIT. No wave stands taller than about 0.78 d (McCowan 1894;
//   Miche 1944 for steep slopes): it breaks. In a sea of many waves the
//   tallest of about a thousand is 1.86 Hs (Rayleigh), so the sea's
//   significant height over a depth d is at most kShoreGamma d = 0.42 d, which
//   puts the tallest drawn wave at 0.78 d and no trough near the bed. The
//   sea's variance over the depth is shared out from the SHORTEST cascade up:
//   the short wind waves a flat keeps, the long ones are what break on it.
//
//   THE BEND (shore_refract.comp). The two long cascades are read through a
//   warp worked out from the depth by the ray method over the far window,
//   W(x) = x + w dP(x) (w the sea's heading, dP the departure of the waves'
//   phase from the open sea's plane wave, in metres): their crests turn
//   toward the shore, wrap round points and shoals, shorten and slow over the
//   bed (each wave keeps its frequency; the local wavenumber is the
//   dispersion relation's); where the rays spread they are a little lower
//   and where they gather a little higher (refraction's Kr = sqrt(b0 / b),
//   much UNDERSTATED), and in the lee of land, by the open water upwind (a
//   wind sea's fetch-limited growth, understated), lower: one factor a
//   cascade, which multiplies the cascade's height before the depth limit
//   takes its share. In the open sea W = x and the factor is 1, exactly. Every reader of the cascades reads
//   the warped cascade alike: the position, the displacement turned with the
//   crests (to the local heading J^T w), the slopes by the chain rule (J^T),
//   the crest foam at W, the surf's incoming crest, sampleHeight. The fine
//   cascade (metres and less, the local wind's) is never warped.
//
// Reflection is left out.
//
// The windows overlap and move (WakeFieldPipeline::setShoreWindows): each
// window's texels lie on the same world lattice wherever it stands, so a
// still bed gives the same depth, texel for texel, after a move; the near
// window's gains go over to the far window's, and the far window's to 1, over
// the outer sixteenth of each side.

#ifndef THREEPP_OCEAN_SHORE_DEPTH_GLSL
#define THREEPP_OCEAN_SHORE_DEPTH_GLSL

// A window's placement (16 bytes), as shore_jfa.comp laid its depth.
struct OceanShoreWindow {
    float originX;// world min corner (the ocean's own frame)
    float originZ;
    float texel;  // m
    float live;   // 1 = its depth is laid
};

// 80 bytes, then the depths, then the bend's table (OceanShoreBendTable).
// Mirrored by WakeFieldPipeline::kShoreDepthHeader.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer OceanShoreDepth {
    OceanShoreWindow win[2];
    vec4  hs;     // each cascade's drawn significant height (m, 4 sigma), .w the sea's
    vec4  km;     // each cascade's rms wavenumber (rad/m)
    uint  res;    // texels a side of a window
    uint  bendRes;// cells a side of the bend's table, 0 before the first is published
    uint  _pad1;
    uint  _pad2;
    float depth[];// window k's texel (i, j) at (k * res + j) * res + i: m under the rest level, 0 on land
};

// The bend's table (shore_refract.comp publishes it, a window at a time,
// each with the placement it was solved on, and the heading): 48 bytes, then
// 4 floats a cell, the phase departures dP of cascades 0 and 1 (m, along the
// heading) and their factors. Mirrored by WakeFieldPipeline::kShoreBendHeader.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer OceanShoreBendTable {
    OceanShoreWindow win[2];// 0 near, 1 far
    vec2  heading;          // the sea's heading it was solved for (unit)
    uint  res;
    uint  chop;             // 1 once the near window's chop table (below) is laid
    float v[];// window k's cell (i, j) at ((k * res + j) * res + i) * 4: dP0, dP1, A0, A1
};
OceanShoreBendTable oceanShoreBendTable(uint64_t addr, uint depthRes) {
    return OceanShoreBendTable(addr + 80ul + uint64_t(depthRes) * uint64_t(depthRes) * 8ul);
}
// THE CHOP's bend (cascade 2), over the near window only (the chop is seen
// near the camera, and its few metres of wavelength feel the bed only in the
// last metre or two): after both windows' cells, a uint a cell of the near
// window, packHalf2x16(dP2, A2) (half floats: a centimetre at 30 m of
// departure, a hundredth of the chop's wavelength). It goes to the open sea
// over the near window's outer quarter, as the near window's bend does.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer OceanShoreChopTable {
    uint v[];// the near window's cell (i, j) at j * res + i
};
OceanShoreChopTable oceanShoreChopTable(uint64_t addr, uint depthRes, uint n) {
    return OceanShoreChopTable(addr + 80ul + uint64_t(depthRes) * uint64_t(depthRes) * 8ul + 48ul +
                               uint64_t(n) * uint64_t(n) * 32ul);
}
// The near window's bend goes over to the far one's over this share of its
// side from each edge (a quarter: the two solves agree in slope and differ by
// their resolution's few metres of phase after a wide flat, and spread over a
// quarter of 256 m that is a few per cent of a wavelength); the far one's to
// the open sea over a sixteenth.
const float kShoreBendNearBlend = 0.25;

const float kShoreGamma = 0.42;// the sea's significant height over a depth d is at most this times d
const float kShoalShare = 0.5; // the share of linear shoaling drawn
const float kShoalMax   = 1.15;// ... and its most

// Window k's depth at a world xz (bilinear between texel centres) and its
// weight: 1 well inside, falling to 0 over the outer sixteenth of the side, 0
// outside or before the window's depth is laid.
float oceanShoreDepthIn(OceanShoreDepth b, uint k, uint res, vec2 xz, out float wgt) {
    wgt = 0.0;
    const OceanShoreWindow w = b.win[k];
    if (w.live < 0.5 || res < 2u) return 0.0;
    const float n  = float(res);
    const vec2  uv = (xz - vec2(w.originX, w.originZ)) / (w.texel * n);
    const vec2  e  = min(uv, 1.0 - uv);
    wgt = smoothstep(0.0, 1.0 / 16.0, min(e.x, e.y));
    if (wgt <= 0.0) return 0.0;
    const vec2  f  = clamp(uv * n - 0.5, vec2(0.0), vec2(n - 1.0));
    const uvec2 i0 = uvec2(min(floor(f), vec2(n - 2.0)));
    const vec2  t  = f - vec2(i0);
    const uint  a  = (k * res + i0.y) * res + i0.x;
    return mix(mix(b.depth[a], b.depth[a + 1u], t.x), mix(b.depth[a + res], b.depth[a + res + 1u], t.x), t.y);
}

// The cascades' gains over a depth d (m), the sea's statistics hs and km,
// and each cascade's refraction factor amp (1, 1, 1 in the open sea, which
// gives the gains without it exactly): the wave that arrives is amp times the
// deep sea's, then shoals and is held under the limit.
vec3 oceanShoreGainAt(float d, vec4 hs, vec4 km, vec3 amp) {
    if (d >= 60.0) return amp;
    d = max(d, 0.0);
    vec3  g;
    float budget = kShoreGamma * kShoreGamma * d * d;// the sea's Hs^2 the depth holds
    for (int c = 2; c >= 0; --c) {
        float s = 1.0;
        if (km[c] > 0.0) {
            const float kd = clamp(km[c] * d, 1e-4, 10.0);
            const float ks = inversesqrt(tanh(kd) * (1.0 + 2.0 * kd / sinh(2.0 * kd)));
            s = min(mix(1.0, ks, kShoalShare), kShoalMax);
        }
        const float e    = hs[c] * hs[c] * s * s * (amp[c] * amp[c]);
        const float take = min(e, budget);
        budget -= take;
        g[c] = e > 1e-12 ? amp[c] * s * sqrt(take / e) : amp[c] * s;
    }
    return g;
}

// Window k's bend at a world xz (bilinear between cell centres) and its
// weight, smoothstep over `blend` of the side from each edge.
void oceanShoreBendIn(OceanShoreBendTable t, uint k, uint n, vec2 xz, float blend, out vec4 v, out float wgt) {
    v   = vec4(0.0, 0.0, 1.0, 1.0);
    wgt = 0.0;
    const OceanShoreWindow w = t.win[k];
    if (w.live < 0.5) return;
    const float fn = float(n);
    const vec2  uv = (xz - vec2(w.originX, w.originZ)) / (w.texel * fn);
    const vec2  e  = min(uv, 1.0 - uv);
    wgt = smoothstep(0.0, blend, min(e.x, e.y));
    if (wgt <= 0.0) return;
    const vec2  f  = clamp(uv * fn - 0.5, vec2(0.0), vec2(fn - 1.0));
    const uvec2 i0 = uvec2(min(floor(f), vec2(fn - 2.0)));
    const vec2  tt = f - vec2(i0);
    const uint  a  = ((k * n + i0.y) * n + i0.x) * 4u;
    const uint  r  = n * 4u;
    for (uint c = 0u; c < 4u; ++c) {
        const float v00 = t.v[a + c], v10 = t.v[a + 4u + c];
        const float v01 = t.v[a + r + c], v11 = t.v[a + r + 4u + c];
        // (equal corners give that value exactly: the open sea stays W = x)
        v[c] = (v00 == v10 && v01 == v11 && v00 == v01) ? v00 : mix(mix(v00, v10, tt.x), mix(v01, v11, tt.x), tt.y);
    }
}

// The bend at a world xz: the long cascades' phase departures dP (m) and
// their factors A, the near window's over the far one's and the far one's over
// the open sea's (0 and 1, exactly where neither has anything); the heading,
// and the cell (m) of the table that holds it.
// The chop's (dP2, A2) at a world xz, the near window's weight wN taken
// (its bilinear read of the half floats; (0, 1) exactly where it has none).
vec2 oceanShoreChopIn(OceanShoreBendTable t, uint64_t addr, uint depthRes, uint n, vec2 xz, float wN) {
    if (wN <= 0.0 || t.chop == 0u) return vec2(0.0, 1.0);
    OceanShoreChopTable ct = oceanShoreChopTable(addr, depthRes, n);
    const OceanShoreWindow w = t.win[0];
    const float fn = float(n);
    const vec2  f  = clamp((xz - vec2(w.originX, w.originZ)) / w.texel - 0.5, vec2(0.0), vec2(fn - 1.0));
    const uvec2 i0 = uvec2(min(floor(f), vec2(fn - 2.0)));
    const vec2  tt = f - vec2(i0);
    const uint  a  = i0.y * n + i0.x;
    const uint  u00 = ct.v[a], u10 = ct.v[a + 1u], u01 = ct.v[a + n], u11 = ct.v[a + n + 1u];
    // (equal corners give that value exactly: the open sea stays W = x)
    const vec2 v = (u00 == u10 && u01 == u11 && u00 == u01)
                 ? unpackHalf2x16(u00)
                 : mix(mix(unpackHalf2x16(u00), unpackHalf2x16(u10), tt.x),
                       mix(unpackHalf2x16(u01), unpackHalf2x16(u11), tt.x), tt.y);
    return vec2(wN * v.x, 1.0 + wN * (v.y - 1.0));
}

void oceanShoreBendAt(uint64_t addr, vec2 xz, out vec3 dP, out vec3 A, out vec2 heading, out float cellM) {
    dP      = vec3(0.0);
    A       = vec3(1.0);
    heading = vec2(1.0, 0.0);
    cellM   = 1.0;
    if (addr == 0ul) return;
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint n = b.bendRes;
    if (n < 2u) return;
    OceanShoreBendTable t = oceanShoreBendTable(addr, b.res);
    vec4  vN, vF;
    float wN, wF;
    oceanShoreBendIn(t, 0u, n, xz, kShoreBendNearBlend, vN, wN);
    oceanShoreBendIn(t, 1u, n, xz, 1.0 / 16.0, vF, wF);
    heading = t.heading;
    dP.xy = wF * vF.xy;
    A.xy  = 1.0 + wF * (vF.zw - 1.0);
    dP.xy = dP.xy + wN * (vN.xy - dP.xy);
    A.xy  = A.xy + wN * (vN.zw - A.xy);
    // The chop: the near window's alone.
    const vec2 c2 = oceanShoreChopIn(t, addr, b.res, n, xz, wN);
    dP.z = c2.x;
    A.z  = c2.y;
    const float cF = t.win[1].live > 0.5 ? t.win[1].texel : t.win[0].texel;
    const float cN = t.win[0].live > 0.5 ? t.win[0].texel : cF;
    cellM = max(mix(cF, cN, wN), 0.05);
}

struct OceanShoreBend {
    vec4 d;    // departures W - x: cascade 0's .xy, cascade 1's .zw (m)
    vec2 d2;   // ... and the chop's
    mat2 j0;   // dW0 / dx (columns: d / dx, d / dz)
    mat2 j1;
    mat2 j2;
    vec4 turn; // cascade 0's (cos, sin) .xy, cascade 1's .zw
    vec2 turn2;// ... and the chop's
    vec3 amp;  // the three cascades' factors
    float n2;  // the chop's compression |grad P2| (1 in the open sea)
    bool on;   // the long cascades are bent here
    bool on2;  // the chop is
};
vec2 oceanShoreTurnOf(vec2 w, vec2 g) {
    const vec2  n = w + g;// grad P = w + grad dP
    const float l = length(n);
    if (l < 1e-6) return vec2(1.0, 0.0);
    const vec2 u = n / l;
    return vec2(dot(u, w), w.x * u.y - w.y * u.x);
}
OceanShoreBend oceanShoreBend(uint64_t addr, vec2 xz) {
    OceanShoreBend s;
    s.d     = vec4(0.0);
    s.d2    = vec2(0.0);
    s.j0    = mat2(1.0);
    s.j1    = mat2(1.0);
    s.j2    = mat2(1.0);
    s.turn  = vec4(1.0, 0.0, 1.0, 0.0);
    s.turn2 = vec2(1.0, 0.0);
    s.n2    = 1.0;
    vec3  dP, aT, pXp, pXm, pZp, pZm;
    vec2  w, wT;
    float h, hT;
    oceanShoreBendAt(addr, xz, dP, s.amp, w, h);
    oceanShoreBendAt(addr, xz + vec2(h, 0.0), pXp, aT, wT, hT);
    oceanShoreBendAt(addr, xz - vec2(h, 0.0), pXm, aT, wT, hT);
    oceanShoreBendAt(addr, xz + vec2(0.0, h), pZp, aT, wT, hT);
    oceanShoreBendAt(addr, xz - vec2(0.0, h), pZm, aT, wT, hT);
    s.on  = any(notEqual(vec4(dP.xy, pXp.xy), vec4(0.0))) || any(notEqual(vec4(pXm.xy, pZp.xy), vec4(0.0))) ||
            any(notEqual(pZm.xy, vec2(0.0)));
    s.on2 = any(notEqual(vec4(dP.z, pXp.z, pXm.z, pZp.z), vec4(0.0))) || pZm.z != 0.0;
    const vec3 gx = (pXp - pXm) / (2.0 * h);// d dP_c / dx
    const vec3 gz = (pZp - pZm) / (2.0 * h);
    if (s.on) {
        s.d    = vec4(w * dP.x, w * dP.y);
        s.j0   = mat2(1.0) + mat2(w * gx.x, w * gz.x);
        s.j1   = mat2(1.0) + mat2(w * gx.y, w * gz.y);
        s.turn = vec4(oceanShoreTurnOf(w, vec2(gx.x, gz.x)), oceanShoreTurnOf(w, vec2(gx.y, gz.y)));
    }
    if (s.on2) {
        s.d2    = w * dP.z;
        s.j2    = mat2(1.0) + mat2(w * gx.z, w * gz.z);
        s.turn2 = oceanShoreTurnOf(w, vec2(gx.z, gz.z));
        s.n2    = length(w + vec2(gx.z, gz.z));
    }
    return s;
}
// The near window's weight at a world xz (oceanShoreBendIn's, without its read).
float oceanShoreNearWeight(OceanShoreWindow w, uint n, vec2 xz) {
    if (w.live < 0.5) return 0.0;
    const vec2 uv = (xz - vec2(w.originX, w.originZ)) / (w.texel * float(n));
    const vec2 e  = min(uv, 1.0 - uv);
    return smoothstep(0.0, kShoreBendNearBlend, min(e.x, e.y));
}

// The chop's bend alone (oceanShoreBend's d2, j2, turn2, n2, on2 and amp.z;
// the long cascades' left as the open sea): what the shade's per-pixel fine
// cascade needs, from the near window's half-float table only.
OceanShoreBend oceanShoreChopBend(uint64_t addr, vec2 xz) {
    OceanShoreBend s;
    s.d     = vec4(0.0);
    s.d2    = vec2(0.0);
    s.j0    = mat2(1.0);
    s.j1    = mat2(1.0);
    s.j2    = mat2(1.0);
    s.turn  = vec4(1.0, 0.0, 1.0, 0.0);
    s.turn2 = vec2(1.0, 0.0);
    s.amp   = vec3(1.0);
    s.n2    = 1.0;
    s.on    = false;
    s.on2   = false;
    if (addr == 0ul) return s;
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint n = b.bendRes;
    if (n < 2u) return s;
    OceanShoreBendTable t = oceanShoreBendTable(addr, b.res);
    if (t.chop == 0u || t.win[0].live < 0.5) return s;
    const OceanShoreWindow nw = t.win[0];
    const float h = max(nw.texel, 0.05);
    const vec2 c  = oceanShoreChopIn(t, addr, b.res, n, xz, oceanShoreNearWeight(nw, n, xz));
    const float xp = oceanShoreChopIn(t, addr, b.res, n, xz + vec2(h, 0.0), oceanShoreNearWeight(nw, n, xz + vec2(h, 0.0))).x;
    const float xm = oceanShoreChopIn(t, addr, b.res, n, xz - vec2(h, 0.0), oceanShoreNearWeight(nw, n, xz - vec2(h, 0.0))).x;
    const float zp = oceanShoreChopIn(t, addr, b.res, n, xz + vec2(0.0, h), oceanShoreNearWeight(nw, n, xz + vec2(0.0, h))).x;
    const float zm = oceanShoreChopIn(t, addr, b.res, n, xz - vec2(0.0, h), oceanShoreNearWeight(nw, n, xz - vec2(0.0, h))).x;
    s.amp.z = c.y;
    s.on2   = c.x != 0.0 || xp != 0.0 || xm != 0.0 || zp != 0.0 || zm != 0.0;
    if (!s.on2) return s;
    const vec2 w = t.heading;
    const vec2 g = vec2(xp - xm, zp - zm) / (2.0 * h);
    s.d2    = w * c.x;
    s.j2    = mat2(1.0) + mat2(w * g.x, w * g.y);
    s.turn2 = oceanShoreTurnOf(w, g);
    s.n2    = length(w + g);
    return s;
}

// A cascade's vector turned with its crests.
vec2 oceanShoreTurn(vec2 cs, vec2 v) {
    return vec2(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y);
}

// The three cascades' gains at a world xz (1, 1, 1 where no window has a
// depth, and with no buffer: addr 0), with the cascades' refraction factors
// amp (oceanShoreBend's, or oceanShoreBendAt's).
vec3 oceanShoreGainWith(uint64_t addr, vec2 xz, vec3 amp) {
    if (addr == 0ul) return vec3(1.0);
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint res = b.res;
    const vec3 a3 = amp;
    if (res == 0u) return a3;
    float wN, wF;
    const float dN = oceanShoreDepthIn(b, 0u, res, xz, wN);
    const float dF = oceanShoreDepthIn(b, 1u, res, xz, wF);
    if (wN <= 0.0 && wF <= 0.0) return a3;
    const vec4 hs = b.hs;
    const vec4 km = b.km;
    const vec3 gF = wF > 0.0 ? mix(a3, oceanShoreGainAt(dF, hs, km, a3), wF) : a3;
    return wN > 0.0 ? mix(gF, oceanShoreGainAt(dN, hs, km, a3), wN) : gF;
}
vec3 oceanShoreGain(uint64_t addr, vec2 xz) {
    vec3  dP, A;
    vec2  w;
    float h;
    oceanShoreBendAt(addr, xz, dP, A, w, h);
    return oceanShoreGainWith(addr, xz, A);
}

#endif// THREEPP_OCEAN_SHORE_DEPTH_GLSL
