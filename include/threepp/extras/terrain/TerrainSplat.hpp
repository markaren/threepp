// Data-driven terrain splat rules (CPU bake, renderer-agnostic).
//
// Promotes the per-demo "slope/altitude/snow" texturing logic (mountains.cpp,
// TerrainGenerator::bakeSplatColors, vulkan_fjord's fjordAlbedo) into one
// reusable, tunable helper. A SplatRules holds an ordered list of SplatLayers;
// each layer is a base colour gated by SLOPE, HEIGHT and CURVATURE windows with
// smoothstep feathers and per-layer world-anchored noise breakup of the
// thresholds. evaluate(x, z, h, slope) blends the active layers and then adds
// low-frequency MACRO VARIATION (luminance + warm/cool tint) so distant terrain
// isn't one flat tone once the shader's detail layer has faded.
//
// Curvature is the Laplacian of the height field, taken with a FIXED world-space
// epsilon (~2-4 m) — the same rationale as TileTerrain's analytic normals: tiles
// baked at different LOD depths must agree, so the metric can't depend on tile
// resolution. Sign convention:
//     curv < 0  → CONVEX  (ridge crests, spurs)  → bare rock
//     curv > 0  → CONCAVE (gullies, hollows)     → scree / talus collects
// SplatRules holds a copy of the height std::function and computes curvature
// itself (4 extra height taps per texel), so the TerrainProvider::albedo
// signature is unchanged — existing providers compile untouched.
//
// UNIT CONVENTION: every height quantity (evaluate's `h`, each layer's height
// window, and the height std::function used for curvature) is in the SAME unit.
// World metres is the natural choice (what TileTerrain's provider passes); a
// caller working in a normalised [0,1] field can use that instead as long as it
// is consistent across all three. `slope` is always 0 (flat) .. 1 (vertical).
//
// Everything is a PURE function of (x, z) and thread-safe — SplatRules can be
// handed to TileTerrain's async bake via albedoFunction(). Header-only, extras.

#ifndef THREEPP_EXTRAS_TERRAIN_TERRAINSPLAT_HPP
#define THREEPP_EXTRAS_TERRAIN_TERRAINSPLAT_HPP

#include "threepp/extras/terrain/TerrainTiles.hpp"// TerrainProvider, HeightGrid

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace threepp::terrain {

    using Rgb = std::array<float, 3>;

    // One splat band. A layer contributes weight
    //     w = slopeWindow(slope) · heightWindow(h) · curvatureResponse(curv) + weightFloor
    // and the final colour is the weight-normalised sum over all layers.
    struct SplatLayer {
        Rgb color{0.5f, 0.5f, 0.5f};

        // Slope window (0 flat .. 1 vertical). A band: full inside [lo,hi],
        // smoothstep-feathered by ±feather at each edge. lo=0 → "flat and up";
        // hi=1 → "steep and up".
        float slopeLo = 0.f, slopeHi = 1.f, slopeFeather = 0.06f;

        // Height window, same unit as evaluate()'s `h`. Defaults span everything.
        float heightLo = -1e30f, heightHi = 1e30f, heightFeather = 1.f;

        // Curvature response (curv normalised to ~[-1,1], convex<0, concave>0):
        //   convexBias  boosts the layer on convex ridge crests (bare rock),
        //   concaveBias boosts it in concave hollows (scree / talus fans).
        float convexBias = 0.f, concaveBias = 0.f;

        // World-anchored value-noise breakup of the thresholds — wiggles the
        // height/slope test per texel so bands never read as a clean iso-contour.
        float noiseAmpHeight = 0.f;// added to the height test (± this, in h-units)
        float noiseAmpSlope = 0.f; // added to the slope test (± this)
        float noiseFreq = 0.05f;   // 1 / metres

        // Constant weight floor — give the "fallback" band (bare rock) a small
        // value so pixels that match no window still resolve to it, never grey.
        float weightFloor = 0.f;
    };

    struct SplatRules {
        std::vector<SplatLayer> layers;

        // Height field for CURVATURE only (world height, same unit as `h`).
        // Optional: curvature terms are inert (curv=0) when unset.
        std::function<float(float x, float z)> height;
        float curvEps = 3.f;    // metres — FIXED so LOD depths agree
        float curvScale = 60.f; // squash strength: curvN = tanh(laplacian·scale)

        // Baked ambient occlusion: darken concave folds (gullies) a touch — real
        // ground has occlusion where surfaces cup. 0 = off.
        float aoStrength = 0.f;// multiplies concave curvature
        float aoMax = 0.30f;   // max darkening fraction

        // ── Macro variation (WS2): low-frequency world-anchored tone drift ────
        // Two octaves of value noise at ~80-300 m wavelength drive a luminance
        // swing and a subtle warm/cool tint lerp. CPU-baked into the splat, so
        // it costs the shader nothing and works on GL/WGPU too.
        bool macroEnabled = true;
        float macroWavelength1 = 240.f;// metres (large blotches)
        float macroWavelength2 = 95.f; // metres (finer drift)
        float macroLumaAmp = 0.12f;    // ± fraction of luminance
        Rgb macroTintWarm{1.05f, 1.00f, 0.94f};
        Rgb macroTintCool{0.95f, 1.00f, 1.05f};
        float macroTintAmp = 0.5f;// 0..1 blend toward the tint

        // Fine high-frequency "de-plastic" grain so flat band fills don't read
        // as painted plastic (separate from the macro drift). 0 = off.
        float fineAmp = 0.08f;
        float fineFreq = 0.14f;// 1 / metres

        // Slope/height/curvature → blended albedo (linear or sRGB — the caller's
        // colours define the space; the blend is colourimetrically neutral).
        [[nodiscard]] Rgb evaluate(float x, float z, float h, float slope) const {
            const float curvN = curvatureN(x, z);

            Rgb acc{0.f, 0.f, 0.f};
            float wsum = 0.f;
            for (const auto& L : layers) {
                // One decorrelated noise sample drives both threshold wiggles.
                const float nH = (vnoise(x * L.noiseFreq, z * L.noiseFreq) - 0.5f) * 2.f;
                const float nS = (vnoise(x * L.noiseFreq + 19.3f, z * L.noiseFreq - 7.1f) - 0.5f) * 2.f;
                const float st = slope + nS * L.noiseAmpSlope;
                const float ht = h + nH * L.noiseAmpHeight;

                float w = band(st, L.slopeLo, L.slopeHi, L.slopeFeather) *
                          band(ht, L.heightLo, L.heightHi, L.heightFeather);
                const float cr = 1.f + L.convexBias * std::max(0.f, -curvN) +
                                 L.concaveBias * std::max(0.f, curvN);
                w = w * cr + L.weightFloor;
                w = std::max(w, 0.f);

                acc[0] += L.color[0] * w;
                acc[1] += L.color[1] * w;
                acc[2] += L.color[2] * w;
                wsum += w;
            }
            if (wsum < 1e-5f) return {0.5f, 0.5f, 0.5f};
            const float inv = 1.f / wsum;
            acc[0] *= inv;
            acc[1] *= inv;
            acc[2] *= inv;

            // Baked AO in concave folds.
            if (aoStrength > 0.f) {
                const float ao = 1.f - std::clamp(std::max(0.f, curvN) * aoStrength, 0.f, aoMax);
                acc[0] *= ao;
                acc[1] *= ao;
                acc[2] *= ao;
            }

            // Fine grain (high-freq luma jitter around 1).
            if (fineAmp > 0.f) {
                const float g = 1.f + fineAmp * (vnoise(x * fineFreq + 3.1f, z * fineFreq + 6.7f) - 0.5f) * 2.f;
                acc[0] *= g;
                acc[1] *= g;
                acc[2] *= g;
            }

            // Macro variation: luminance drift + warm/cool tint.
            if (macroEnabled) {
                const float m1 = vnoise(x / macroWavelength1, z / macroWavelength1);
                const float m2 = vnoise(x / macroWavelength2, z / macroWavelength2);
                const float macro = std::clamp(m1 * 0.65f + m2 * 0.35f, 0.f, 1.f);
                const float luma = 1.f + macroLumaAmp * (macro * 2.f - 1.f);
                // Decorrelated field drives the tint so hue and brightness don't
                // move in lockstep.
                const float t = std::clamp(vnoise(x / macroWavelength1 + 41.f, z / macroWavelength1 - 17.f), 0.f, 1.f);
                for (int c = 0; c < 3; ++c) {
                    const float tint = macroTintWarm[c] + (macroTintCool[c] - macroTintWarm[c]) * t;
                    acc[c] *= luma * (1.f + (tint - 1.f) * macroTintAmp);
                }
            }

            acc[0] = std::clamp(acc[0], 0.f, 1.f);
            acc[1] = std::clamp(acc[1], 0.f, 1.f);
            acc[2] = std::clamp(acc[2], 0.f, 1.f);
            return acc;
        }

        // Ready-to-assign TerrainProvider::albedo. The returned functor copies
        // the rules by value (pure/thread-safe for TileTerrain's async bake).
        [[nodiscard]] std::function<void(float, float, float, float, float*)> albedoFunction() const {
            SplatRules copy = *this;
            return [copy](float x, float z, float h, float slope, float* rgb) {
                const Rgb c = copy.evaluate(x, z, h, slope);
                rgb[0] = c[0];
                rgb[1] = c[1];
                rgb[2] = c[2];
            };
        }

    private:
        // Normalised curvature at (x,z): tanh(Laplacian · scale). Fixed eps so
        // adjacent LODs agree. <0 convex (ridge), >0 concave (hollow).
        [[nodiscard]] float curvatureN(float x, float z) const {
            if (!height) return 0.f;
            const float e = curvEps;
            const float hc = height(x, z);
            const float lap = height(x + e, z) + height(x - e, z) +
                              height(x, z + e) + height(x, z - e) - 4.f * hc;
            return std::tanh((lap / (e * e)) * curvScale);
        }

        static float smoothstep(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }
        static float band(float x, float lo, float hi, float f) {
            f = std::max(f, 1e-4f);
            return smoothstep(lo - f, lo + f, x) * (1.f - smoothstep(hi - f, hi + f, x));
        }

        // Hash-based 2D value noise in [0,1], world-anchored & pure. Inputs are
        // already scaled (freq applied by the caller).
        static float hash21(float px, float py) {
            // integer-lattice hash → [0,1)
            const float s = std::sin(px * 127.1f + py * 311.7f) * 43758.5453f;
            return s - std::floor(s);
        }
        static float vnoise(float x, float y) {
            const float xi = std::floor(x), yi = std::floor(y);
            const float xf = x - xi, yf = y - yi;
            const float a = hash21(xi, yi), b = hash21(xi + 1.f, yi);
            const float c = hash21(xi, yi + 1.f), d = hash21(xi + 1.f, yi + 1.f);
            const float ux = xf * xf * (3.f - 2.f * xf);
            const float uy = yf * yf * (3.f - 2.f * yf);
            return (a + (b - a) * ux) * (1.f - uy) + (c + (d - c) * ux) * uy;
        }
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINSPLAT_HPP
