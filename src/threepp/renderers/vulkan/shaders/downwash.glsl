// The rotor downwash's profile on the GPU (threepp::Downwash). Three readers:
//   gbuffer.frag       dwWaterSlope: the rings on a `ripples` material
//   grass_wind.comp    dwGrassOffset: blades lean out and are pressed down
//   particle_emit.comp dwWarp: the hollow a jet blows in a particle cloud
// The includer enables GL_EXT_buffer_reference and GL_EXT_scalar_block_layout.
// The profile's CPU twin is src/threepp/scenes/Downwash.cpp: KEEP IN SYNC.
//
// The constants are the pine forest's stills (pines/world/downwash.py): the
// four rotors' wakes come down as one jet, strike the surface and turn into a
// wall jet whose speed peaks at the PEAK radius and dies over the DECAY
// radius past it, both growing with the rotor's height over the surface; the
// whole effect fades from full under 2 m of height to none at 9 m.

#ifndef THREEPP_DOWNWASH_GLSL
#define THREEPP_DOWNWASH_GLSL

#include "downwash_shared.h"

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DownwashBuf {
    DownwashHeaderGpu h;
    DownwashSourceGpu s[kDownwashMaxSources];
};

const float kDwTau       = 6.28318530718;
const float kDwRefRadius = 0.5;// m, the reference disc (Downwash::kReferenceRadius)

float dwSmooth(float a, float b, float v) {
    const float t = clamp((v - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float dwHeightFade(float h) { return 1.0 - dwSmooth(2.0, 9.0, h); }
float dwPeakRadius(float R, float h) { return 0.6 * (R / kDwRefRadius) + 0.30 * max(h, 0.0); }
float dwDecayRadius(float R, float h) { return 1.2 * (R / kDwRefRadius) + 0.55 * max(h, 0.0); }

// ── Water ────────────────────────────────────────────────────────────────────
// The slope (dh/dx, dh/dz) of the rings at world point p on a surface at
// p.y, and the roughness the struck disc adds. Analytic: a height
// h = A(r) sin(phase(r, t)) differentiated along r (the envelope's own
// derivative is left out; it is slow next to the rings). `footprint` is the
// pixel's size on the surface: a ring narrower than a few pixels fades out
// rather than alias.
vec2 dwWaterSlope(DownwashBuf d, vec3 p, float footprint, out float roughAdd) {
    vec2 g = vec2(0.0);
    roughAdd = 0.0;
    const uint  n = min(d.h.count, uint(kDownwashMaxSources));
    const float t = d.h.time;
    for (uint i = 0u; i < n; ++i) {
        const DownwashSourceGpu s = d.s[i];
        const vec2  dxz = p.xz - s.center.xz;
        const float r   = length(dxz);
        const vec2  u   = r > 1e-5 ? dxz / r : vec2(0.0);
        if (s.kind == kDownwashKindJet) {
            const float h = max(s.center.y - p.y, 0.0);
            const float k = s.share * dwHeightFade(h);
            if (k <= 0.0) continue;
            const float rp = dwPeakRadius(s.radius, h);
            const float rd = dwDecayRadius(s.radius, h);
            if (r > rp + 2.2 * rd * 7.0) continue;
            // The wall jet's rings: the wavelength grows outward (dispersion),
            // the amplitude rises to the jet's ring and dies away past it, a
            // lobed swell from the rotors' wakes. Travelling outward in time.
            const float lam   = d.h.ringLam0 + d.h.ringLamGrowth * r;
            const float phase = kDwTau * (r / lam - d.h.ringHz * t);
            const float env   = dwSmooth(0.25 * rp, rp, r) * exp(-max(r - rp, 0.0) / (2.2 * rd));
            const float lobe  = s.lobes > 0.5 ? 1.0 + 0.22 * cos(s.lobes * atan(dxz.y, dxz.x) + s.lobePhase) : 1.0;
            const float amp   = d.h.rippleAmp * k * env * lobe * dwSmooth(2.0 * footprint, 4.0 * footprint, lam);
            // d(r / lam)/dr = lam0 / lam^2
            g += (amp * cos(phase) * kDwTau * d.h.ringLam0 / (lam * lam)) * u;
            // Inside the jet's ring the struck water is rougher, not chopped.
            const float q = r / (1.1 * rp);
            roughAdd = max(roughAdd, d.h.roughBoost * min(k, 1.0) * exp(-q * q));
        } else if (s.kind == kDownwashKindRings) {
            // Rings leaving the centre one after another from t0, each a short
            // crest train under a Gaussian, fading as it spreads.
            const float age0 = t - s.t0;
            if (age0 <= 0.0) continue;
            const float wl = max(s.wavelength, 1e-3);
            const float wd = 0.625 * wl;
            const int   cnt = min(int(s.count), 32);
            float dhdr = 0.0;
            for (int m = 0; m < cnt; ++m) {
                const float age = age0 - float(m) * s.interval;
                if (age <= 0.0) break;
                const float rk = s.radius + s.speed * age;
                const float x  = r - rk;
                const float xg = x / wd;
                if (xg * xg > 9.0) continue;
                const float a  = s.share * exp(-0.35 * max(rk - s.radius, 0.0)) * min(1.0, age / 0.4);
                const float ga = a * exp(-xg * xg);
                const float ph = kDwTau * x / wl;
                dhdr += ga * (kDwTau / wl * cos(ph) - 2.0 * x / (wd * wd) * sin(ph));
            }
            g += (dhdr * dwSmooth(2.0 * footprint, 4.0 * footprint, wl)) * u;
        }
    }
    return g;
}

// ── Grass ────────────────────────────────────────────────────────────────────
// The world offset of a blade vertex at world rest point p, height fraction hf
// (0 base, 1 tip), on a field whose blades stand H world metres; t is the
// field's clock. Each jet leans the blade out from its axis, most at the wall
// jet's peak radius, and presses it down under the axis, weighted by hf^2
// like the wind (base planted). Bounded: the horizontal offset never exceeds
// H, and a vertex never drops below 0.1 of its own height (the cull bound,
// VulkanRenderer::Impl::grassSwayWorldAabb, relies on both).
vec3 dwGrassOffset(DownwashBuf d, vec3 p, float hf, float H, float t) {
    vec3 off = vec3(0.0);
    if (hf <= 0.0 || H <= 0.0) return off;
    const float ground = p.y - hf * H;
    const float w      = hf * hf;
    const uint  n      = min(d.h.count, uint(kDownwashMaxSources));
    for (uint i = 0u; i < n; ++i) {
        const DownwashSourceGpu s = d.s[i];
        if (s.kind != kDownwashKindJet) continue;
        const float h = max(s.center.y - ground, 0.0);
        const float k = s.share * dwHeightFade(h);
        if (k <= 0.0) continue;
        const float rp = dwPeakRadius(s.radius, h);
        const float rd = dwDecayRadius(s.radius, h);
        const vec2  dxz = p.xz - s.center.xz;
        const float r   = length(dxz);
        if (r > rp + 7.0 * rd) continue;
        const float wall  = dwSmooth(0.0, rp, r) * exp(-max(r - rp, 0.0) / rd);
        const float q     = r / (0.8 * rp);
        const float press = exp(-q * q);
        // A smooth spatial spread instead of a per-plant hash (a blade's own
        // vertices must agree), and the jet's beat in time.
        const float jit  = 1.0 + 0.3 * sin(p.x * 3.1 + p.z * 2.3);
        const float beat = 1.0 + d.h.flutter * sin(t * 11.0 + p.x * 1.7 + p.z * 2.9);
        const float a    = min(d.h.leanMax * wall * k * jit * beat, 1.4);
        const vec2  u    = r > 1e-4 ? dxz / r : vec2(0.0);
        off.xz += u * (H * sin(a) * w);
        off.y  -= H * ((1.0 - cos(a)) + d.h.pressMax * press * min(k, 1.0)) * w;
    }
    off.y = max(off.y, -0.9 * hf * H);
    const float L = length(off.xz);
    if (L > H) off.xz *= H / L;
    return off;
}

// ── Particles ────────────────────────────────────────────────────────────────
// The hollow: a point at radius r from a jet's axis moves out to
// sqrt(r^2 + c(r)^2), c(r) = c0 exp(-(r / 2Rh)^2). Near the axis that empties
// a disc of radius c0; the map keeps the points' order (c0 < 2 Rh) so the
// cloud never folds, its tail dies a few Rh out, and the rim where the moved
// points pile up is lifted a little. Only below the rotor plane. `prev` warps
// with last frame's centres (the motion vectors' sample). `jit` in [0, 1] is
// the particle's own, stable share of the hollow (0.45 + 0.55 jit of c0), so
// the rim thins out instead of cutting a hard circle.
vec3 dwWarp(DownwashBuf d, vec3 p, bool prev, float jit) {
    const float own = 0.45 + 0.55 * clamp(jit, 0.0, 1.0);
    const uint n = min(d.h.count, uint(kDownwashMaxSources));
    for (uint i = 0u; i < n; ++i) {
        const DownwashSourceGpu s = d.s[i];
        if (s.kind != kDownwashKindJet) continue;
        const vec3  c      = prev ? s.prevCenter : s.center;
        const float ground = s.surfaceY > -1e30 ? s.surfaceY : p.y;// -FLT_MAX = unset
        const float h      = max(c.y - ground, 0.0);
        const float k      = s.share * dwHeightFade(h) * (1.0 - dwSmooth(c.y - 0.3, c.y + 0.6, p.y));
        if (k <= 0.0) continue;
        const float rh = dwPeakRadius(s.radius, h) + 0.8;
        const float c0 = min(d.h.hollow * k, 1.8 * rh) * own;
        const vec2  dxz = p.xz - c.xz;
        const float r   = length(dxz);
        if (r > 4.0 * rh) continue;
        const float e  = r / (2.0 * rh);
        const float ce = c0 * exp(-e * e);
        const float r2 = sqrt(r * r + ce * ce);
        const vec2  u  = r > 1e-5 ? dxz / r : vec2(1.0, 0.0);
        const float qq = (r2 - rh) / rh;
        p.xz = c.xz + u * r2;
        p.y += d.h.rimLift * c0 * exp(-qq * qq);
    }
    return p;
}

#endif// THREEPP_DOWNWASH_GLSL
