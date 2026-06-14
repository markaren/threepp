// Procedural terrain heightfield generator (CPU).
//
// Produces a displaced, horizontal PlaneGeometry from a multifractal noise
// stack (fBm / ridged-multifractal / hybrid + domain warp). Header-only and
// dependency-free beyond threepp core so it can be reused from examples and,
// later, as the CPU reference / fallback for the GPU compute path and the
// material-splat bake (see the mountain-configurator plan, M2/M3).
//
// Coordinate convention: the generated geometry lies in the XZ plane (rotated
// flat, +Y up), centred on the origin, spanning [-worldSize/2, +worldSize/2]
// on each axis. heightAt() returns metres of elevation (+Y).
//
// All noise is seeded: reseed(seed) shuffles a fresh permutation table so the
// same seed always reproduces the same terrain.

#ifndef THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP
#define THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace threepp::terrain {

    enum class NoiseType { fBm = 0,
                           Ridged = 1,
                           Hybrid = 2 };

    enum class Falloff { None = 0,
                         Radial = 1 };// radial fade to base: grounds the massif on a plain
                                      // (high falloffStart = gentle round base; low = tight cone)

    // Configurator-facing knobs. Mapped 1:1 onto ImGui controls in the example.
    struct TerrainParams {
        unsigned int seed = 1337;

        float worldSize = 1200.f;// metres — full mesh extent on X and Z
        int resolution = 256;    // grid segments per side (verts = resolution+1 squared)

        NoiseType noiseType = NoiseType::Ridged;
        float featureScale = 420.f;// metres per base-noise period (macro wavelength)
        int octaves = 7;
        float lacunarity = 2.0f;// frequency multiplier per octave
        float gain = 0.5f;      // amplitude persistence per octave
        float amplitude = 350.f;// metres — peak elevation
        float warp = 0.4f;      // 0..1 domain-warp strength (breaks grid-aligned ridges)

        float ridgeSharpness = 0.7f;// 0..1 — crest contrast for Ridged/Hybrid
        float heightExponent = 1.3f;// >1 deepens valleys, <1 lifts plateaus
        int terraces = 0;           // 0 = off; >1 = strata steps (mesa look)

        Falloff falloff = Falloff::None;
        float falloffStart = 0.55f;// radial: full height inside this normalised radius
    };

    class TerrainGenerator {

    public:
        explicit TerrainGenerator(unsigned int seed = 1337) { reseed(seed); }

        void reseed(unsigned int seed) {
            std::array<int, 256> p{};
            std::iota(p.begin(), p.end(), 0);
            std::mt19937 rng(seed ? seed : 1u);
            for (int i = 255; i > 0; --i) {
                std::uniform_int_distribution<int> d(0, i);
                std::swap(p[i], p[d(rng)]);
            }
            for (int i = 0; i < 512; ++i) perm_[i] = p[i & 255];
            seed_ = seed;
        }

        [[nodiscard]] unsigned int seed() const { return seed_; }

        // Elevation in metres at world (wx, wz).
        [[nodiscard]] float heightAt(float wx, float wz, const TerrainParams& tp) const {
            float nx = wx / tp.featureScale;
            float nz = wz / tp.featureScale;

            // Domain warp (single level): bend the sample position by a low-octave
            // fBm so ridges meander instead of aligning to the noise grid.
            if (tp.warp > 0.f) {
                const float w = tp.warp * 1.5f;
                const float qx = fbm(nx, nz, 4, 2.f, 0.5f);
                const float qz = fbm(nx + 5.2f, nz + 1.3f, 4, 2.f, 0.5f);
                nx += w * qx;
                nz += w * qz;
            }

            float h;// expected ~[0,1]
            switch (tp.noiseType) {
                case NoiseType::Ridged:
                    h = ridged(nx, nz, tp.octaves, tp.lacunarity, tp.gain, tp.ridgeSharpness);
                    break;
                case NoiseType::Hybrid:
                    h = hybrid(nx, nz, tp.octaves, tp.lacunarity, tp.gain, tp.ridgeSharpness);
                    break;
                case NoiseType::fBm:
                default:
                    h = fbm(nx, nz, tp.octaves, tp.lacunarity, tp.gain) * 0.5f + 0.5f;
                    break;
            }
            h = std::clamp(h, 0.f, 1.f);

            if (tp.terraces > 1) {
                const float steps = static_cast<float>(tp.terraces);
                h = std::round(h * steps) / steps;
            }
            if (tp.heightExponent != 1.f) h = std::pow(h, tp.heightExponent);

            float baseSink = 0.f;
            if (tp.falloff == Falloff::Radial) {
                // Euclidean radial fade: full height inside `falloffStart`,
                // eased to 0 by the patch radius so the finite terrain settles
                // onto a surrounding ground plane (a round massif base, or a
                // cone at low falloffStart) instead of a hard floating edge.
                const float half = tp.worldSize * 0.5f;
                const float r = std::sqrt(wx * wx + wz * wz) / std::max(half, 1e-3f);
                const float fall = smoothstep(1.0f, tp.falloffStart, r);
                h *= fall;
                // Sink the faded rim well below the base level so the square
                // patch border dives under the surrounding ground plane (no
                // visible 1 m lip / square outline). The massif emerges from
                // the plain where its slope crosses y=0.
                baseSink = 0.12f * tp.amplitude * (1.f - fall);
            }

            return h * tp.amplitude - baseSink;
        }

        // Build a fresh horizontal, displaced PlaneGeometry.
        [[nodiscard]] std::shared_ptr<BufferGeometry> createGeometry(const TerrainParams& tp) const {
            auto geo = PlaneGeometry::create(tp.worldSize, tp.worldSize,
                                             static_cast<unsigned int>(std::max(tp.resolution, 1)),
                                             static_cast<unsigned int>(std::max(tp.resolution, 1)));
            geo->rotateX(-kHalfPi);// lie flat, +Y up
            applyTo(*geo, tp);
            return geo;
        }

        // Re-displace an existing geometry in place (same extent + resolution).
        // Bumps the position/normal attribute versions so the renderer rebuilds
        // the BLAS (the plain-dynamic-geometry path — no special mesh type).
        void applyTo(BufferGeometry& geo, const TerrainParams& tp) const {
            auto* pos = geo.getAttribute<float>("position");
            if (!pos) return;
            auto& a = pos->array();
            const int vcount = pos->count();

            std::vector<int> idx(static_cast<size_t>(vcount));
            std::iota(idx.begin(), idx.end(), 0);
            // Disjoint per-vertex writes — safe to parallelise.
            std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
                const float x = a[i * 3 + 0];
                const float z = a[i * 3 + 2];
                a[i * 3 + 1] = heightAt(x, z, tp);
            });
            pos->needsUpdate();

            geo.computeVertexNormals();
            if (auto* nrm = geo.getAttribute<float>("normal")) nrm->needsUpdate();
            geo.computeBoundingBox();
            geo.computeBoundingSphere();
        }

    private:
        static constexpr float kHalfPi = 1.57079632679489661923f;

        static float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
        static float lerp(float a, float b, float t) { return a + t * (b - a); }
        static float smoothstep(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }
        static float grad2(int h, float x, float y) {
            switch (h & 7) {
                case 0: return x + y;
                case 1: return -x + y;
                case 2: return x - y;
                case 3: return -x - y;
                case 4: return x;
                case 5: return -x;
                case 6: return y;
                default: return -y;
            }
        }

        // 2D Perlin gradient noise, output ~[-1, 1].
        float noise2(float x, float y) const {
            const int X = static_cast<int>(std::floor(x)) & 255;
            const int Y = static_cast<int>(std::floor(y)) & 255;
            x -= std::floor(x);
            y -= std::floor(y);
            const float u = fade(x), v = fade(y);
            const int aa = perm_[perm_[X] + Y];
            const int ab = perm_[perm_[X] + Y + 1];
            const int ba = perm_[perm_[X + 1] + Y];
            const int bb = perm_[perm_[X + 1] + Y + 1];
            const float x1 = lerp(grad2(aa, x, y), grad2(ba, x - 1.f, y), u);
            const float x2 = lerp(grad2(ab, x, y - 1.f), grad2(bb, x - 1.f, y - 1.f), u);
            return lerp(x1, x2, v);
        }

        // Fractional Brownian motion — amplitude-normalised to ~[-1, 1].
        float fbm(float x, float y, int oct, float lac, float gain) const {
            float f = 1.f, a = 1.f, sum = 0.f, norm = 0.f;
            for (int i = 0; i < oct; ++i) {
                sum += a * noise2(x * f, y * f);
                norm += a;
                f *= lac;
                a *= gain;
            }
            return norm > 0.f ? sum / norm : 0.f;
        }

        // Ridged multifractal (Musgrave): ridges form where noise crosses zero;
        // each octave's detail is gated by the previous octave's signal so
        // detail concentrates on crests. Output ~[0, 1].
        float ridged(float x, float y, int oct, float lac, float gain, float sharp) const {
            const float offset = 1.0f;
            const float exp = 1.f + sharp * 2.f;// 1..3 crest sharpening
            float freq = 1.f, amp = 1.f, sum = 0.f, norm = 0.f, prev = 1.f;
            for (int i = 0; i < oct; ++i) {
                const float n = noise2(x * freq, y * freq);
                float signal = offset - std::abs(n);
                signal = std::pow(std::clamp(signal, 0.f, 1.f), exp);
                signal *= std::clamp(prev * 2.f, 0.f, 1.f);// gate by previous octave
                sum += amp * signal;
                norm += amp;
                prev = signal;
                freq *= lac;
                amp *= gain;
            }
            return norm > 0.f ? std::clamp(sum / norm, 0.f, 1.f) : 0.f;
        }

        // Hybrid multifractal — smooth valleys, rough peaks. Blends fBm body with
        // ridged crests, biasing toward ridges at altitude. Output ~[0, 1].
        float hybrid(float x, float y, int oct, float lac, float gain, float sharp) const {
            const float fb = fbm(x, y, oct, lac, gain) * 0.5f + 0.5f;
            const float rg = ridged(x, y, oct, lac, gain, sharp);
            const float t = std::clamp(fb, 0.f, 1.f);
            return std::clamp(lerp(fb, rg, 0.5f * t + 0.25f * sharp), 0.f, 1.f);
        }

        std::array<int, 512> perm_{};
        unsigned int seed_ = 0;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP
