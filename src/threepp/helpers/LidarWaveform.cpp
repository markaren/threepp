
#include "threepp/helpers/LidarWaveform.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {
    constexpr float kFwhmToSigma = 0.42466452f;// 1 / (2 √(2 ln 2))
    // 3 sigma covers >99.7% of the Gaussian — enough fidelity for the
    // peak shape without paying for the long tails.
    constexpr float kSupportSigmas = 3.f;
}// namespace

void threepp::synthesizeLidarWaveform(
        const std::vector<LidarReturn>& returns,
        std::size_t numBeams,
        std::size_t slotsPerBeam,
        const LidarWaveformParams& params,
        std::vector<float>& waveformOut) {

    const std::size_t bins = std::max<std::size_t>(1, params.bins);
    const float       maxR = std::max(1e-3f, params.maxRange);
    const float       binW = maxR / static_cast<float>(bins);
    const float       sigma = std::max(binW * 0.5f, params.pulseFWHM * kFwhmToSigma);
    const float       supportR = sigma * kSupportSigmas;
    const float       norm = 1.f / (sigma * 2.5066283f);// 1 / (σ √(2π))
    const float       invTwoSigmaSq = 1.f / (2.f * sigma * sigma);

    waveformOut.assign(numBeams * bins, params.noiseFloor);
    if (slotsPerBeam == 0) return;

    const std::size_t slots = slotsPerBeam;
    for (std::size_t beam = 0; beam < numBeams; ++beam) {
        const std::size_t begin = beam * slots;
        const std::size_t end   = std::min(begin + slots, returns.size());
        float* out = &waveformOut[beam * bins];
        for (std::size_t k = begin; k < end; ++k) {
            const auto& r = returns[k];
            if (r.returnNo <= 0) continue;// miss / sub-threshold
            if (r.distance <= 0.f || r.distance > maxR) continue;

            // Centre bin + support window. Splat a Gaussian peak whose
            // area equals the discrete return's intensity (so a 0.5
            // intensity return contributes 0.5 in integrated waveform
            // area, regardless of pulse width).
            const float centreBin = r.distance / binW;
            const int lo = std::max(0,
                                    static_cast<int>(std::floor(centreBin - supportR / binW)));
            const int hi = std::min(static_cast<int>(bins) - 1,
                                    static_cast<int>(std::ceil(centreBin + supportR / binW)));
            for (int b = lo; b <= hi; ++b) {
                const float binDist = (static_cast<float>(b) + 0.5f) * binW;
                const float d       = binDist - r.distance;
                const float g       = norm * std::exp(-d * d * invTwoSigmaSq);
                out[b] += r.intensity * g * binW;
            }
        }
    }
}
