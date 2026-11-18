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
// Wavelength shortening over the bed (the dispersion relation's tanh(kd))
// cannot vary from place to place in an FFT sea, whose every wave has one
// length everywhere: it is left out. So are refraction (the crests keep
// running with the wind) and reflection.
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

// 80 bytes, then the depths. Mirrored by WakeFieldPipeline::kShoreDepthHeader.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer OceanShoreDepth {
    OceanShoreWindow win[2];
    vec4  hs;  // each cascade's drawn significant height (m, 4 sigma), .w the sea's
    vec4  km;  // each cascade's rms wavenumber (rad/m)
    uint  res; // texels a side of a window
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
    float depth[];// window k's texel (i, j) at (k * res + j) * res + i: m under the rest level, 0 on land
};

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

// The cascades' gains over a depth d (m), the sea's statistics hs and km.
vec3 oceanShoreGainAt(float d, vec4 hs, vec4 km) {
    if (d >= 60.0) return vec3(1.0);
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
        const float e    = hs[c] * hs[c] * s * s;
        const float take = min(e, budget);
        budget -= take;
        g[c] = e > 1e-12 ? s * sqrt(take / e) : s;
    }
    return g;
}

// The three cascades' gains at a world xz (1, 1, 1 where no window has a
// depth, and with no buffer: addr 0).
vec3 oceanShoreGain(uint64_t addr, vec2 xz) {
    if (addr == 0ul) return vec3(1.0);
    OceanShoreDepth b = OceanShoreDepth(addr);
    const uint res = b.res;
    if (res == 0u) return vec3(1.0);
    float wN, wF;
    const float dN = oceanShoreDepthIn(b, 0u, res, xz, wN);
    const float dF = oceanShoreDepthIn(b, 1u, res, xz, wF);
    if (wN <= 0.0 && wF <= 0.0) return vec3(1.0);
    const vec4 hs = b.hs;
    const vec4 km = b.km;
    const vec3 gF = wF > 0.0 ? mix(vec3(1.0), oceanShoreGainAt(dF, hs, km), wF) : vec3(1.0);
    return wN > 0.0 ? mix(gF, oceanShoreGainAt(dN, hs, km), wN) : gF;
}

#endif// THREEPP_OCEAN_SHORE_DEPTH_GLSL
