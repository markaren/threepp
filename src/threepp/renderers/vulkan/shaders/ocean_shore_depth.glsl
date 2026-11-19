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
//   dispersion relation's), and where the rays spread or no ray from the open
//   sea arrives (a lee) they are lower: refraction's Kr = sqrt(b0 / b),
//   UNDERSTATED as the shoaling is (its square root), multiplies the
//   cascade's height before the depth limit takes its share. In the open sea
//   W = x and the factor is 1, exactly. Every reader of the cascades reads
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

// The bend's table (shore_refract.comp publishes it, over the far window,
// with the placement and the heading it was solved on): 32 bytes, then 4
// floats a cell, the phase departures dP of cascades 0 and 1 (m, along the
// heading) and their refraction factors. Mirrored by
// WakeFieldPipeline::kShoreBendHeader.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer OceanShoreBendTable {
    OceanShoreWindow win;
    vec2  heading;// the sea's heading it was solved for (unit)
    uint  res;
    uint  _pad0;
    float v[];// cell (i, j) at (j * res + i) * 4: dP0, dP1, A0, A1
};
OceanShoreBendTable oceanShoreBendTable(uint64_t addr, uint depthRes) {
    return OceanShoreBendTable(addr + 80ul + uint64_t(depthRes) * uint64_t(depthRes) * 8ul);
}

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

// The bend at a world xz (bilinear between cell centres): the long
// cascades' phase departures dP (m) and their factors A, toward the open sea
// (0 and 1, exactly) over the outer sixteenth of the window and outside it;
// the heading, and the table's cell (m).
void oceanShoreBendAt(uint64_t addr, vec2 xz, out vec2 dP, out vec2 A, out vec2 heading, out float cellM) {
    dP      = vec2(0.0);
    A       = vec2(1.0);
    heading = vec2(1.0, 0.0);
    cellM   = 1.0;
    if (addr == 0ul) return;
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint n = b.bendRes;
    if (n < 2u) return;
    OceanShoreBendTable t = oceanShoreBendTable(addr, b.res);
    const OceanShoreWindow w = t.win;
    if (w.live < 0.5) return;
    heading = t.heading;
    cellM   = max(w.texel, 0.05);
    const float fn = float(n);
    const vec2  uv = (xz - vec2(w.originX, w.originZ)) / (w.texel * fn);
    const vec2  e  = min(uv, 1.0 - uv);
    const float wgt = smoothstep(0.0, 1.0 / 16.0, min(e.x, e.y));
    if (wgt <= 0.0) return;
    const vec2  f  = clamp(uv * fn - 0.5, vec2(0.0), vec2(fn - 1.0));
    const uvec2 i0 = uvec2(min(floor(f), vec2(fn - 2.0)));
    const vec2  tt = f - vec2(i0);
    const uint  a  = (i0.y * n + i0.x) * 4u;
    const uint  r  = n * 4u;
    vec4 v;
    for (uint c = 0u; c < 4u; ++c) {
        const float v00 = t.v[a + c], v10 = t.v[a + 4u + c];
        const float v01 = t.v[a + r + c], v11 = t.v[a + r + 4u + c];
        // (equal corners give that value exactly: the open sea stays W = x)
        v[c] = (v00 == v10 && v01 == v11 && v00 == v01) ? v00 : mix(mix(v00, v10, tt.x), mix(v01, v11, tt.x), tt.y);
    }
    dP = wgt * v.xy;
    A  = 1.0 + wgt * (v.zw - 1.0);
}

// The bend at a rest point as a reader of the cascades takes it: each long
// cascade is read at W_c(x) = x + w dP_c, near x at W_c(x) + J_c (y - x) (the
// warp's Jacobian I + w grad dP_c, the departure differenced a cell either
// way); a vector of the cascade's (its horizontal displacement) comes back to
// the world turned from the sea's heading w to the waves' own here, J_c^T w =
// grad P_c (turn: its cos, sin); amp the refraction factors. on false: the
// open sea, where W = x exactly and nothing need be read.
struct OceanShoreBend {
    vec4 d;   // departures W - x: cascade 0's .xy, cascade 1's .zw (m)
    mat2 j0;  // dW0 / dx (columns: d / dx, d / dz)
    mat2 j1;
    vec4 turn;// cascade 0's (cos, sin) .xy, cascade 1's .zw
    vec2 amp;
    bool on;
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
    s.d    = vec4(0.0);
    s.j0   = mat2(1.0);
    s.j1   = mat2(1.0);
    s.turn = vec4(1.0, 0.0, 1.0, 0.0);
    vec2  dP, w, aT, wT;
    float h, hT;
    oceanShoreBendAt(addr, xz, dP, s.amp, w, h);
    vec2 pXp, pXm, pZp, pZm;
    oceanShoreBendAt(addr, xz + vec2(h, 0.0), pXp, aT, wT, hT);
    oceanShoreBendAt(addr, xz - vec2(h, 0.0), pXm, aT, wT, hT);
    oceanShoreBendAt(addr, xz + vec2(0.0, h), pZp, aT, wT, hT);
    oceanShoreBendAt(addr, xz - vec2(0.0, h), pZm, aT, wT, hT);
    s.on = any(notEqual(vec4(dP, pXp), vec4(0.0))) || any(notEqual(vec4(pXm, pZp), vec4(0.0))) ||
           any(notEqual(pZm, vec2(0.0)));
    if (!s.on) return s;
    const vec2 gx = (pXp - pXm) / (2.0 * h);// d dP_c / dx, c = 0, 1
    const vec2 gz = (pZp - pZm) / (2.0 * h);
    s.d    = vec4(w * dP.x, w * dP.y);
    s.j0   = mat2(1.0) + mat2(w * gx.x, w * gz.x);
    s.j1   = mat2(1.0) + mat2(w * gx.y, w * gz.y);
    s.turn = vec4(oceanShoreTurnOf(w, vec2(gx.x, gz.x)), oceanShoreTurnOf(w, vec2(gx.y, gz.y)));
    return s;
}
// A cascade's vector turned with its crests.
vec2 oceanShoreTurn(vec2 cs, vec2 v) {
    return vec2(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y);
}

// The three cascades' gains at a world xz (1, 1, 1 where no window has a
// depth, and with no buffer: addr 0), with the long cascades' refraction
// factors amp (oceanShoreBend's, or oceanShoreBendAt's).
vec3 oceanShoreGainWith(uint64_t addr, vec2 xz, vec2 amp) {
    if (addr == 0ul) return vec3(1.0);
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint res = b.res;
    const vec3 a3 = vec3(amp, 1.0);
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
    vec2  dP, A, w;
    float h;
    oceanShoreBendAt(addr, xz, dP, A, w, h);
    return oceanShoreGainWith(addr, xz, A);
}

#endif// THREEPP_OCEAN_SHORE_DEPTH_GLSL
