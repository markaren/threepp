// The signal-level half shared by ImagingSonar and MarineRadar.
//
// Both sensors are ray-sampled coherent echo synthesizers. The renderer's
// ray tracer answers one question per ray: what does this direction strike
// first, at what range, with what normal and which instance? Everything that
// makes the picture look like a sonar or a radar rather than a depth map is
// signal processing on top of those answers, and it lives here, renderer-free:
//
//   - an ARRAY PATTERN: the voltage response of a tapered line aperture
//     versus u = sin(theta) - sin(theta_steer). Its main lobe is the
//     beamwidth and its sidelobes are the arcs a bright target smears across
//     neighbouring beams, the artefact every real imaging sonar and every
//     marine radar shows;
//   - a RANGE RESPONSE: the matched-filter output of the transmitted pulse,
//     a triangle for a CW pulse and a Hann-weighted sinc for a chirp, which
//     is what spreads a point target over range bins;
//   - a stable RANDOM PHASE per scatterer, so that echoes summed coherently
//     interfere into speckle with the statistics of the real thing (a
//     Rayleigh amplitude over a rough surface) and the same scene, pose and
//     seed give the same speckle bit for bit;
//   - the dB and noise helpers both sensor equations share.
//
// Nothing here knows about the GPU, so CI pins it without one.
#ifndef THREEPP_ECHOSYNTHESIS_HPP
#define THREEPP_ECHOSYNTHESIS_HPP

#include "threepp/math/Vector3.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace threepp::echo {

    constexpr double kPi = 3.14159265358979323846;

    [[nodiscard]] inline double dbToPower(double db) { return std::pow(10.0, 0.1 * db); }
    [[nodiscard]] inline double powerToDb(double p, double floorDb = -400.0) {
        return p > 0.0 ? 10.0 * std::log10(p) : floorDb;
    }

    // SplitMix64's finalizer: a 64-bit key to a well-mixed 64-bit word.
    [[nodiscard]] inline std::uint64_t mix64(std::uint64_t z) {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Uniform in [0, 1) from a key.
    [[nodiscard]] inline double unitFromKey(std::uint64_t key) {
        return static_cast<double>(mix64(key) >> 11) * (1.0 / 9007199254740992.0);
    }
    [[nodiscard]] inline std::uint64_t combineKeys(std::uint64_t a, std::uint64_t b) {
        return mix64(a ^ (b + 0x632BE59BD9B4E019ULL + (a << 6) + (a >> 2)));
    }

    /**
     * The phase of the sub-ray-scale scatterer field at world point `p`: the
     * point is snapped to cells of `cell` metres and the cell is hashed with
     * `seed`. A static surface keeps its phases, so a still sensor sees still
     * speckle; a moving one sees its hits slide across cells and the speckle
     * decorrelate, as a real one does. Returns radians in [0, 2 pi).
     */
    [[nodiscard]] double scattererPhase(const Vector3& p, double cell, std::uint64_t seed);

    // One complex Gaussian draw of unit mean power (each component variance
    // 1/2), Box-Muller from two keyed uniforms, so a cell's noise is a pure
    // function of (seed, cell) and not of the order cells are visited in.
    [[nodiscard]] std::complex<double> complexGaussian(std::uint64_t key);

    enum class Taper {
        Uniform,//!< -13.3 dB first sidelobe, the narrowest main lobe
        Hann,   //!< -31.5 dB, a main lobe 1.6x wider
        Hamming,//!< -42.7 dB, a main lobe 1.5x wider
        Taylor25//!< Taylor n-bar = 4, -25 dB design sidelobes: what arrays are built with
    };

    /**
     * Voltage pattern of a tapered line array with half-wavelength element
     * spacing, normalised to 1 on the steering direction, tabulated against
     * u = sin(theta) - sin(theta_steer).
     *
     * Built from the -3 dB beamwidth it must have at broadside, not from an
     * aperture: the aperture (in wavelengths) is solved for, so a sensor
     * datasheet's "1 deg horizontal beamwidth" is exactly what comes out.
     * Steering off broadside widens the beam by 1/cos(theta) for free,
     * because the pattern is a function of u.
     *
     * The table spans |u| <= span(); beyond it the pattern reads 0. That is a
     * truncation of the far sidelobes, chosen by the caller as the kernel
     * half-width the synthesizers sum over.
     */
    class ArrayPattern {
    public:
        ArrayPattern() = default;
        ArrayPattern(double beamwidthDeg, Taper taper, double spanU, unsigned tableSize = 8192);

        // Signed voltage: the sidelobes alternate in sign, which matters when
        // echoes from two angles add coherently in one beam.
        [[nodiscard]] double operator()(double u) const {
            const double a = std::abs(u);
            if (a >= span_ || table_.empty()) return 0.0;
            const double x = a * invStep_;
            const auto i = static_cast<std::size_t>(x);
            const double f = x - static_cast<double>(i);
            return table_[i] + f * (table_[i + 1] - table_[i]);
        }

        [[nodiscard]] double span() const { return span_; }
        // Aperture length in wavelengths, and its element count.
        [[nodiscard]] double apertureWavelengths() const { return aperture_; }
        [[nodiscard]] unsigned elements() const { return elements_; }
        // Measured on the built table: the -3 dB half-width in u and the
        // highest sidelobe in dB (both what the tests pin).
        [[nodiscard]] double halfPowerU() const { return halfPowerU_; }
        [[nodiscard]] double peakSidelobeDb() const { return peakSidelobeDb_; }

    private:
        std::vector<double> table_;
        double span_ = 0.0, invStep_ = 0.0, aperture_ = 0.0;
        double halfPowerU_ = 0.0, peakSidelobeDb_ = 0.0;
        unsigned elements_ = 0;
    };

    // The aperture weights for `taper` at n points across [-1/2, 1/2].
    [[nodiscard]] std::vector<double> taperWeights(Taper taper, unsigned n);

    enum class PulseKind {
        CW,   //!< rectangular pulse, matched filter = triangle over +-resolution
        Chirp,//!< LFM pulse, Hann-weighted on receive: sinc-like, -31 dB range sidelobes
    };

    /**
     * Matched-filter response of the transmitted pulse against the range
     * offset in metres, peak 1. `resolution` is the -3 dB range resolution
     * the datasheet quotes (c tau / 2 for CW, about 1.44 c / 2B for a
     * Hann-weighted chirp).
     */
    class RangeResponse {
    public:
        RangeResponse() = default;
        RangeResponse(double resolution, PulseKind kind);

        [[nodiscard]] double operator()(double dr) const;
        // Beyond this offset the response is taken as zero.
        [[nodiscard]] double halfSpan() const { return halfSpan_; }
        [[nodiscard]] double resolution() const { return resolution_; }
        [[nodiscard]] PulseKind kind() const { return kind_; }

    private:
        double resolution_ = 0.0, halfSpan_ = 0.0, scale_ = 0.0;
        PulseKind kind_ = PulseKind::CW;
    };

    // Sample a uniform-in-index stratum with a keyed jitter: (i + u) / n.
    [[nodiscard]] inline double stratum(unsigned i, unsigned n, std::uint64_t key, bool jitter) {
        const double u = jitter ? unitFromKey(key) : 0.5;
        return (static_cast<double>(i) + u) / static_cast<double>(n);
    }

    // Hardware threads the synthesizers fan out over (1 when unknown).
    [[nodiscard]] unsigned workerCount(unsigned cap = 8);

}// namespace threepp::echo

#endif//THREEPP_ECHOSYNTHESIS_HPP
