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

#include "threepp/math/MathUtils.hpp"
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

        // Multiplier on the windowed weight (floor excluded). The blend is a
        // weight-normalised sum, so a layer whose window is fully open still
        // only ties with every other open layer; a seabed layer under the sea
        // surface would mix 50/50 with the grass the slope window admits there.
        // Scaling lets a layer OWN its band. 1 = the old arithmetic exactly.
        float weightScale = 1.f;

        // STRUCTURE band this layer feeds in the terrain shader's per-band
        // texture sets (MaterialWithTerrainMaps): 0 grass, 1 rock, 2 scree,
        // 3 snow (the makeTerrainBandSet convention). -1 = no structure — the
        // layer's coverage shows only the macro splat colour. Several layers
        // may share a band (grass + heath + wetland are all "grass" structure).
        int structureBand = -1;
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
        //
        // AO uses its OWN concavity measure (aoConcavity below), separate from
        // the layers' curvatureN: the layer biases want a saturating "is this a
        // ridge or a hollow" signal, but a saturating response cannot tell a
        // broad gully from a 2 m erosion channel — with it, every droplet
        // drainage line and noise pit darkens identically and the terrain
        // stipples into dot fields and parallel streaks. The AO measure instead
        //   (a) requires concavity at BOTH the local scale (curvEps) and a
        //       larger scale (aoEps): a flat pixel beside a cliff (concave only
        //       through the remote wall inside the footprint), a Laplacian
        //       ringing shoulder at a slope break, and a noise pit all fail one
        //       of the two and are gated off;
        //   (b) takes the WEAKER of the two magnitudes at a LOW gain
        //       (aoCurvScale, far from tanh saturation) through an aoLo..aoHi
        //       deadband — narrow channels register weakly at aoEps and fall in
        //       the deadband, broad gullies pass and darken.
        float aoStrength = 0.f; // multiplies the gated concavity
        float aoMax = 0.30f;    // max darkening fraction
        float aoEps = 0.f;      // metres; 0 → 3 × curvEps
        float aoCurvScale = 8.f;// LOW gain (cf. curvScale) — must not saturate
        float aoLo = 0.25f;     // deadband: no darkening below this response
        float aoHi = 0.80f;     // full response above this

        // ── Macro variation (WS2): low-frequency world-anchored tone drift ────
        // Two octaves of value noise at ~80-300 m wavelength drive a luminance
        // swing and a subtle warm/cool tint lerp. CPU-baked into the splat, so
        // it costs the shader nothing and works on GL too.
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
                const float w = layerWeight(L, x, z, h, slope, curvN);

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

            // Baked AO in concave folds — dual-scale gated, deadbanded (see the
            // aoStrength field comment for why this is NOT curvN).
            if (aoStrength > 0.f) {
                const float cc = aoConcavity(x, z);
                const float ao = 1.f - std::clamp(cc * aoStrength, 0.f, aoMax);
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

        // STRUCTURE-band coverage at a sample: the same per-layer weights the
        // colour blend uses, binned by SplatLayer::structureBand into w4[0..3]
        // and normalised against the TOTAL layer weight — a pixel owned by
        // band-less layers keeps Σw4 < 1 and shows only the macro colour.
        // Written for the terrain shader's weight-map bake (LINEAR RGBA8).
        void evaluateWeights(float x, float z, float h, float slope, float* w4) const {
            const float curvN = curvatureN(x, z);
            w4[0] = w4[1] = w4[2] = w4[3] = 0.f;
            float wsum = 0.f;
            for (const auto& L : layers) {
                const float w = layerWeight(L, x, z, h, slope, curvN);
                wsum += w;
                if (L.structureBand >= 0 && L.structureBand < 4) w4[L.structureBand] += w;
            }
            if (wsum < 1e-5f) return;
            const float inv = 1.f / wsum;
            for (int b = 0; b < 4; ++b) w4[b] = std::clamp(w4[b] * inv, 0.f, 1.f);
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

        // Ready-to-assign TerrainProvider::weights (same by-value copy contract).
        [[nodiscard]] std::function<void(float, float, float, float, float*)> weightsFunction() const {
            SplatRules copy = *this;
            return [copy](float x, float z, float h, float slope, float* w4) {
                copy.evaluateWeights(x, z, h, slope, w4);
            };
        }

    private:
        // One layer's weight at a sample (shared by evaluate/evaluateWeights so
        // colour and structure coverage can never disagree). One decorrelated
        // noise sample drives both threshold wiggles.
        [[nodiscard]] float layerWeight(const SplatLayer& L, float x, float z,
                                        float h, float slope, float curvN) const {
            const float nH = (vnoise(x * L.noiseFreq, z * L.noiseFreq) - 0.5f) * 2.f;
            const float nS = (vnoise(x * L.noiseFreq + 19.3f, z * L.noiseFreq - 7.1f) - 0.5f) * 2.f;
            const float st = slope + nS * L.noiseAmpSlope;
            const float ht = h + nH * L.noiseAmpHeight;

            float w = band(st, L.slopeLo, L.slopeHi, L.slopeFeather) *
                      band(ht, L.heightLo, L.heightHi, L.heightFeather);
            const float cr = 1.f + L.convexBias * std::max(0.f, -curvN) +
                             L.concaveBias * std::max(0.f, curvN);
            return std::max(w * cr * L.weightScale + L.weightFloor, 0.f);
        }
        // Normalised curvature at (x,z): tanh(Laplacian · scale). Fixed eps so
        // adjacent LODs agree. <0 convex (ridge), >0 concave (hollow).
        //
        // MULTI-SCALE SIGN AGREEMENT: a single-scale Laplacian is a band-pass —
        // it responds to EVERYTHING near its epsilon wavelength. Height detail
        // at that scale (erosion droplet pits, provider noise octaves) makes
        // every bump alternate concave/convex and stipples the AO into a dot
        // field; and any sharp slope break (groove, terrace edge, cliff top)
        // RINGS — a concave centre flanked by convex shoulders at ±e (and vice
        // versa), so every drainage channel renders as parallel companion
        // lines. Real landform concavity (a gully) is concave across scales;
        // the artifacts are not. So: take the Laplacian at e AND 2e, keep the
        // smaller magnitude when the signs agree, zero otherwise. Pits and
        // ringing shoulders flip sign (or collapse) at 2e and are gated off;
        // wide gullies pass at both scales and keep their full response.
        [[nodiscard]] float lapAt(float x, float z, float hc, float e) const {
            return (height(x + e, z) + height(x - e, z) +
                    height(x, z + e) + height(x, z - e) - 4.f * hc) / (e * e);
        }

        [[nodiscard]] float curvatureN(float x, float z) const {
            if (!height) return 0.f;
            const float hc = height(x, z);
            const float l1 = lapAt(x, z, hc, curvEps);
            const float l2 = lapAt(x, z, hc, curvEps * 2.f);
            float m = 0.f;
            if (l1 > 0.f && l2 > 0.f) m = std::min(l1, l2);
            else if (l1 < 0.f && l2 < 0.f) m = std::max(l1, l2);
            return std::tanh(m * curvScale);
        }

        // Gated AO concavity in [0,1] — see the aoStrength field comment.
        // Requires concavity at BOTH curvEps (local — gates off flat-beside-
        // cliff pixels and ringing shoulders) and aoEps (broad — gates off
        // pits and narrow channels), then deadbands the weaker magnitude at
        // low gain so only genuinely broad concavity darkens.
        [[nodiscard]] float aoConcavity(float x, float z) const {
            if (!height) return 0.f;
            const float hc = height(x, z);
            const float lLocal = lapAt(x, z, hc, curvEps);
            if (lLocal <= 0.f) return 0.f;
            const float eBig = aoEps > 0.f ? aoEps : curvEps * 3.f;
            const float lBroad = lapAt(x, z, hc, eBig);
            if (lBroad <= 0.f) return 0.f;
            const float r = std::tanh(std::min(lLocal, lBroad) * aoCurvScale);
            return math::smoothstep(aoLo, aoHi, r);
        }

        static float band(float x, float lo, float hi, float f) {
            f = std::max(f, 1e-4f);
            return math::smoothstep(lo - f, lo + f, x) * (1.f - math::smoothstep(hi - f, hi + f, x));
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
