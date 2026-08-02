// The DSP primitives every procedural-audio recipe in this repo is built from.
//
// Extracted from the PhysX vehicle demo's sound rig (examples/projects/Vehicle/
// VehicleSounds.hpp), which had grown the same four helpers the Shooter, FPS
// and Ocean demos had each copy-pasted for themselves. The recipes stay where
// they are — they are demo material — but the filters, the normaliser and the
// loop-seam crossfade are general, and the editor's generated sounds needed
// exactly them.
//
// Header-only, like threepp/audio/WavFile.hpp next door, and free of the audio
// module: this synthesises float samples, it does not play them. Pair it with
// writeWav() and hand the file to Audio.
//
// DETERMINISM IS THE CONTRACT. Nothing here reads a clock or a global; feed the
// same seeded PRNG and the same parameters and the same samples come out, which
// is what lets a generated sound be cached by its parameters alone.

#ifndef THREEPP_SYNTHUTIL_HPP
#define THREEPP_SYNTHUTIL_HPP

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace threepp::audio::synth {

    struct OnePole {
        float y = 0.f;
        float operator()(float x, float a) {
            y += a * (x - y);
            return y;
        }
    };

    inline float lpAlpha(float cutoffHz, int sr) {
        return 1.f - std::exp(-2.f * math::PI * cutoffHz / static_cast<float>(sr));
    }

    // Two-pole resonator (narrow bandpass). Noise through this is the
    // physically-honest way to get a squeal/ring: the band wanders in
    // phase and amplitude like a real resonance, where a raw sin() reads
    // as a flute. r in [0.98, 0.999] sets the Q.
    struct Resonator {
        float a1 = 0.f, a2 = 0.f, g = 1.f, y1 = 0.f, y2 = 0.f;
        void set(float freqHz, float r, int sr) {
            a1 = 2.f * r * std::cos(2.f * math::PI * freqHz / static_cast<float>(sr));
            a2 = -r * r;
            g = 1.f - r;// rough gain normalisation
        }
        float operator()(float x) {
            const float y = g * x + a1 * y1 + a2 * y2;
            y2 = y1;
            y1 = y;
            return y;
        }
    };

    inline std::vector<float> normalized(std::vector<float> s, float peak) {
        float m = 0.f;
        for (float x : s) m = std::max(m, std::abs(x));
        if (m > 1e-6f)
            for (float& x : s) x *= peak / m;
        return s;
    }

    // Fold the `extra`-sample overhang back onto the head (linear crossfade)
    // so noise/filter state passes the loop seam without a click.
    inline std::vector<float> loopable(const std::vector<float>& s, int n, int extra) {
        std::vector<float> out(s.begin(), s.begin() + n);
        for (int i = 0; i < extra; ++i) {
            const float w = static_cast<float>(i) / static_cast<float>(extra);
            out[i] = s[n + i] * (1.f - w) + s[i] * w;
        }
        return out;
    }

}// namespace threepp::audio::synth

#endif//THREEPP_SYNTHUTIL_HPP
