// Time-of-flight pulse waveform synthesis from discrete LIDAR returns.
//
// Real LIDAR detectors don't output discrete `LidarReturn`s — they sample
// a continuous analog signal of received intensity vs time-of-flight,
// and the discrete returns are detected (peak-find / matched filter) from
// that waveform after analog filtering. This helper inverts the process:
// given the discrete returns the GPU path-tracer produced, synthesize the
// per-beam waveform you'd see on a vector oscilloscope at the detector.
//
// Each return contributes a Gaussian splat at its slant range, with
// FWHM = `pulseFWHM` metres. Multi-returns from the same beam (multiple
// surfaces along the path, fog scatter events) all sum into the same
// per-beam waveform — which is exactly how the optical front-end
// integrates the signal. Sub-beam samples (from `LidarParams::samplesPerBeam`
// > 1) also sum, naturally averaging out Monte Carlo variance.
//
// Use this when you need realistic LIDAR sensor output (e.g. for
// downstream peak-detection code or signal-processing studies). For
// plain point-cloud consumption, the discrete `LidarReturn` vector is
// usually what you want.

#ifndef THREEPP_LIDARWAVEFORM_HPP
#define THREEPP_LIDARWAVEFORM_HPP

#include "LidarTypes.hpp"

#include <cstddef>
#include <vector>

namespace threepp {

    struct LidarWaveformParams {
        // Total range covered by the waveform [0, maxRange] in metres.
        // Each beam's waveform is a 1-D array of `bins` floats laid out
        // uniformly across this range.
        float maxRange = 100.f;

        // Number of distance bins per beam. 256 gives ~0.4 m resolution
        // over 100 m, comparable to a 5 ns analog sampling rate on a
        // real sensor.
        std::size_t bins = 256;

        // Gaussian pulse full-width-half-max in metres. Real time-of-flight
        // LIDARs use 5–10 ns pulses; at the speed of light that's a
        // range FWHM of ~0.75–1.5 m. The 1 σ of the Gaussian =
        // FWHM / (2 √(2 ln 2)) ≈ FWHM · 0.4247.
        float pulseFWHM = 1.0f;

        // Optional noise floor added uniformly to every bin (sensor's
        // dark-current floor). 0 = noiseless.
        float noiseFloor = 0.f;
    };

    /**
     * Splat the discrete returns produced by `VulkanRenderer::scanLidar`
     * into per-beam Gaussian waveforms. `returns` is the fixed-stride
     * vector the scanner produces; `slotsPerBeam` = samplesPerBeam ·
     * maxReturns (use 1 if you called scanLidar with both at 1).
     *
     * `waveformOut` is resized to `numBeams * params.bins`. Beam `b`'s
     * waveform lives at `&waveformOut[b * params.bins]`, indexed by
     * distance bin in ascending order (bin 0 = closest range).
     *
     * Cost: O(numReturns · 6σ_bins) per beam. For typical settings
     * (~150 returns × ~10 bins of Gaussian support) this is a sub-ms
     * CPU loop even at 130k-beam OS0-128 density.
     */
    void synthesizeLidarWaveform(
            const std::vector<LidarReturn>& returns,
            std::size_t numBeams,
            std::size_t slotsPerBeam,
            const LidarWaveformParams& params,
            std::vector<float>& waveformOut);

}// namespace threepp

#endif//THREEPP_LIDARWAVEFORM_HPP
