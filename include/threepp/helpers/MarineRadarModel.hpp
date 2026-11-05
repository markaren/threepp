// A rotating marine navigation radar: the renderer-free half of MarineRadar.
//
// The antenna turns; each SLOT of its revolution (one spoke's width of
// azimuth) is ray-sampled through the vertical fan by the renderer's tracer,
// and every hit is weighed by the radar equation
//
//     Pr = Pt G^2 lambda^2 sigma F^4 / ((4 pi)^3 R^4 L)
//
//   sigma  the radar cross-section of the patch of surface the ray stands for:
//          per-material backscatter (Lambert + a Kirchhoff facet glint, or an
//          angle-independent corner-reflector cluster), and on the SEA the NRL
//          sea-clutter model (Gregers-Hansen & Mital, 2012) at the mean grazing
//          angle, redistributed over the rendered waves by their local tilt;
//   F      the two-ray propagation factor over the sea: the direct and the
//          sea-reflected paths interfere (Fresnel reflection of sea water,
//          reduced by the Rayleigh roughness of the waves), which is why a
//          low buoy fades in and out with range and why low targets vanish
//          before the horizon;
//   L      system losses, and rain's specific attenuation on the way.
//
// Each slot's returns are summed coherently (two-way phase 2kR plus a stable
// scatterer phase) into a complex range profile through the pulse's
// matched-filter response; a SPOKE is then the antenna's two-way azimuth
// pattern convolved across the slots around it, plus thermal noise kTBF. Rain
// cells add volume returns (Marshall-Palmer Z = 200 R^1.6) and attenuate what
// lies behind them. The receiver applies the controls an operator has: gain,
// sea clutter (STC) and rain clutter (FTC, a range differentiator in dB).
//
// What it is not: flat earth (no radar horizon past the scene), no ducting,
// no multiple bounces between targets (a corner reflector is a material, not
// traced), no second-time-around echoes. The NRL coefficients for HH were
// checked against a published worked value; the VV set is as recalled and
// unverified, marked where it is used.
#ifndef THREEPP_MARINERADARMODEL_HPP
#define THREEPP_MARINERADARMODEL_HPP

#include "threepp/helpers/EchoSynthesis.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace threepp {

    enum class RadarPolarization { HH, VV };

    /**
     * NRL sea-clutter reflectivity sigma0 in dB (m^2/m^2): grazing angle in
     * degrees, frequency in GHz, Douglas sea state 0-6. Valid 0.1-60 deg,
     * 0.5-35 GHz; about 2-3 dB from the data it was fitted to.
     */
    [[nodiscard]] double nrlSeaClutterDb(double grazingDeg, double frequencyGHz, double seaState,
                                         RadarPolarization pol);

    // Significant wave height (m) at the middle of a Douglas sea state.
    [[nodiscard]] double douglasWaveHeight(double seaState);

    /**
     * How a surface scatters the radar back, per unit area:
     *   sigma0 = 10^(mu/10) cos^2 theta + |R|^2 exp(-tan^2 theta / s^2) / (s^2 cos^4 theta)
     *          + 10^(iso/10)                         (a reflector: angle-independent)
     * or, with `sea` set, the NRL model at the hit's grazing angle.
     */
    struct RadarMaterial {
        float lambertDb = -18.f;
        float reflectionLossDb = -6.f;
        float rmsSlope = 0.12f;
        float isotropicDb = -400.f;// <= -300 = off
        bool sea = false;

        [[nodiscard]] double sigma0(double cosTheta) const;

        static RadarMaterial Sea() {
            RadarMaterial m;
            m.sea = true;
            return m;
        }
        static RadarMaterial Metal() { return {-12.f, -0.2f, 0.05f, -400.f, false}; }
        static RadarMaterial Fibreglass() { return {-20.f, -9.f, 0.08f, -400.f, false}; }
        static RadarMaterial Rock() { return {-10.f, -10.f, 0.35f, -400.f, false}; }
        static RadarMaterial Vegetation() { return {-8.f, -20.f, 0.f, -400.f, false}; }
        static RadarMaterial Concrete() { return {-13.f, -7.f, 0.10f, -400.f, false}; }
        // An octahedral reflector cluster: about 10 m^2 from any bearing over
        // a ~0.25 m^2 cross-section.
        static RadarMaterial RadarReflector() { return {-10.f, -3.f, 0.1f, 16.f, false}; }
    };

    struct RadarMaterials {
        RadarMaterial defaultMaterial = RadarMaterial::Rock();
        std::unordered_map<std::int32_t, RadarMaterial> byInstance;

        void set(std::int32_t instanceId, const RadarMaterial& m) { byInstance[instanceId] = m; }
        [[nodiscard]] const RadarMaterial& of(std::int32_t instanceId) const {
            const auto it = byInstance.find(instanceId);
            return it == byInstance.end() ? defaultMaterial : it->second;
        }
    };

    /**
     * A rain shower: a column of rain from the sea up to `top`, with a
     * Gaussian rain-rate profile across (sigma = `radius`) peaking at
     * `rainRate` mm/h.
     */
    struct RainCell {
        Vector3 centre;// x, z centre; y unused
        float radius = 300.f;
        float top = 1500.f;   // world y
        float rainRate = 8.f; // mm/h at the centre
    };

    /**
     * Everything that defines one marine radar. Frame: the antenna turns about
     * the node's local +Y, clockwise seen from above; relative bearing 0 is
     * local -Z (mount it so that is the bow). GEOMETRY fields (beams, spokes,
     * the sampling) are baked when a synthesizer is built; the rest is read
     * per slot.
     */
    struct MarineRadarModel {
        // ── Transmitter / antenna ────────────────────────────────────────
        double frequency = 9.41e9;   // Hz, X band
        float peakPowerW = 4000.f;
        RadarPolarization polarization = RadarPolarization::HH;
        float horizontalBeamwidth = 1.8f;// deg, -3 dB one-way (geometry)
        echo::Taper horizontalTaper = echo::Taper::Taylor25;
        float verticalBeamwidth = 22.f;  // deg (geometry)
        float antennaEfficiency = 0.6f;
        float rpm = 24.f;
        unsigned spokes = 2048;// per revolution (geometry)

        // ── Range and receiver ───────────────────────────────────────────
        float minRange = 20.f;   // the main bang / blind zone, m
        // Where the trace starts, m: the antenna's own radome. Nothing nearer is
        // seen and everything past it occludes, the blind zone included: echoes
        // inside minRange are dropped, not looked through, so a sea surface
        // inside the main bang still hides the seabed under it.
        float housingRange = 1.f;
        float maxRange = 1852.f; // 1 nmi
        unsigned rangeBins = 1024;
        float pulseLength = 80e-9f;// s
        float noiseFigureDb = 5.f;
        float systemLossDb = 6.f;

        // ── Ray sampling (geometry) ──────────────────────────────────────
        unsigned azimuthSamples = 2;  // rays across one slot's width
        unsigned elevationSamples = 88;
        // Share of the elevation samples spent just below the horizon
        // (uniform in range on the sea plane), where the sea at long range
        // lives, and just above it (the same band mirrored: up to twice the
        // antenna's height at each range, out to twice the maximum range),
        // where a mast, a reflector or a wheelhouse taller than a low antenna
        // stands at range. The rest is uniform over the vertical fan.
        float horizonShare = 0.55f;
        float aboveHorizonShare = 0.25f;
        float patternSpanBeamwidths = 4.f;// azimuth kernel half-width
        std::uint64_t seed = 0x7AD4A5EAULL;

        // ── Environment ──────────────────────────────────────────────────
        float seaLevel = 0.f;
        float seaState = 3.f;  // Douglas
        bool multipath = true; // two-ray propagation factor over the sea
        std::vector<RainCell> rain;

        // ── Display controls ─────────────────────────────────────────────
        float gainDb = 0.f;         // lowers the threshold
        float thresholdDb = 9.f;    // SNR shown as black at zero gain
        float dynamicRangeDb = 40.f;
        float seaClutterDb = 0.f;   // STC depth at min range
        float stcRange = 900.f;     // where STC has faded to zero, m
        float rainClutter = 0.f;    // FTC 0..1

        [[nodiscard]] double wavelength() const { return 299792458.0 / frequency; }
        [[nodiscard]] float binWidth() const {
            return rangeBins ? (maxRange - minRange) / static_cast<float>(rangeBins) : 0.f;
        }
        [[nodiscard]] double revolutionPeriod() const { return rpm > 0.f ? 60.0 / rpm : 1e9; }
        [[nodiscard]] double antennaGain() const;// linear, from the beamwidths and efficiency

        // ── Presets: class values for the categories, not a fitted unit ─
        // A 4 ft open array on a 4 kW magnetron: 1.8 x 22 deg, 24 rpm.
        static MarineRadarModel OpenArray4ft();
        // A 6 ft open array, 12 kW: 1.2 x 22 deg.
        static MarineRadarModel OpenArray6ft();
        // A 24 in radome, 4 kW: 3.9 x 25 deg; what a small USV carries.
        static MarineRadarModel Radome24in();
    };

    /** One spoke as the receiver hands it to the display. */
    struct RadarSpoke {
        std::int64_t index = 0;    // absolute slot index (grows across revolutions)
        double time = 0.0;         // sim time the antenna pointed here
        float bearingDeg = 0.f;    // relative to the antenna's zero (the bow), clockwise
        float worldBearingDeg = 0.f;// from world -Z ("north") clockwise toward +X
        std::vector<float> snrDb;  // per range bin, before the display controls
        std::vector<float> display;// [0, 1] after gain / STC / FTC
        std::vector<std::int32_t> label;// strongest contributor's instance id, -1 none, -3 rain
    };

    /**
     * The last revolution, spoke by spoke, indexed by relative slot; plus a
     * world-bearing lookup for north-up displays.
     */
    struct RadarPolarImage {
        unsigned spokes = 0;
        unsigned bins = 0;
        float minRange = 0.f;
        float maxRange = 0.f;
        std::vector<float> display;    // spokes x bins
        std::vector<float> snrDb;      // spokes x bins
        std::vector<std::int32_t> label;// spokes x bins
        std::vector<double> time;      // per spoke; < 0 = never written
        std::vector<float> worldBearingDeg;
        std::vector<std::int32_t> byWorldBearing;// world-bearing slot -> spoke, -1 empty

        void reset(unsigned spokes, unsigned bins, float minRange, float maxRange);
        void store(const RadarSpoke& s);
    };

    /**
     * Ray tables, patterns and the per-slot / per-spoke folds. Holds one bit
     * of state: the running mean of the sea's tilt modulation per range band,
     * which keeps the clutter's mean on the NRL curve while the rendered waves
     * shape it. Deterministic for a given sequence of calls.
     */
    class MarineRadarSynthesizer {
    public:
        explicit MarineRadarSynthesizer(const MarineRadarModel& model);

        [[nodiscard]] unsigned raysPerSlot() const { return azimuthSamples_ * elevationSamples_; }
        [[nodiscard]] unsigned kernelHalfSlots() const { return kernelHalf_; }
        [[nodiscard]] const echo::ArrayPattern& horizontalPattern() const { return hPattern_; }

        /**
         * Directions and solid angles of slot `slot` (relative to the antenna's
         * zero), for an antenna `antennaHeight` m above the sea, in the
         * antenna's LEVEL frame: turned with it, not pitched or rolled. The
         * horizon samples aim at the sea, which is level whatever the boat
         * does: laid out in a frame tilted by a degree, rays meant for the
         * sea at 1 nmi would hit it at 150 m on one side and pass over every
         * low target on the other. Appends raysPerSlot() entries to each vector.
         */
        void slotRays(std::int64_t slot, double antennaHeight, std::vector<Vector3>& dirs,
                      std::vector<float>& solidAngle) const;

        /**
         * Fold one traced slot into its complex range profile (`column`,
         * rangeBins long, overwritten) and its per-bin strongest-contributor
         * label / amplitude. `returns`, `dirs` (the level frame of slotRays,
         * which `worldFromLevel` turns into the world) and `solidAngle` are
         * the slot's raysPerSlot() rays. `antennaFromLevel` is the antenna's
         * pitch and roll, where its vertical pattern is read; nullptr = level.
         * Returns the profile's extent: one past its last non-zero bin.
         */
        unsigned foldSlot(const MarineRadarModel& model, const RadarMaterials& materials, std::int64_t slot,
                      const LidarReturn* returns, const Vector3* dirs, const float* solidAngle,
                      const Vector3& origin, const Matrix3& worldFromLevel, std::uint64_t revolutionKey,
                      std::complex<float>* column, std::int32_t* label, float* labelAmp,
                      const Matrix3* antennaFromLevel = nullptr);

        /**
         * Form spoke `slot` from the slot ring. `ring(s)` returns the complex
         * profile of slot s (any s within kernelHalfSlots() of `slot`);
         * `labels(s)` / `amps(s)` its labels. Fills snr, display and label.
         */
        template<class RingFn, class LabelFn, class AmpFn>
        void formSpoke(const MarineRadarModel& model, std::int64_t slot, RingFn&& ring, LabelFn&& labels,
                       AmpFn&& amps, std::uint64_t revolutionKey, RadarSpoke& out) const {
            const unsigned all = std::max(1u, model.rangeBins);
            formSpoke(model, slot, ring, labels, amps, [all](std::int64_t) { return all; }, revolutionKey, out);
        }
        // The same with `extent(s)`: the bins of slot s past which its profile is
        // zero (what foldSlot returns), so a spoke skips the empty tail.
        template<class RingFn, class LabelFn, class AmpFn, class ExtentFn>
        void formSpoke(const MarineRadarModel& model, std::int64_t slot, RingFn&& ring, LabelFn&& labels,
                       AmpFn&& amps, ExtentFn&& extent, std::uint64_t revolutionKey, RadarSpoke& out) const;

    private:
        unsigned spokes_ = 0, azimuthSamples_ = 0, elevationSamples_ = 0, horizonSamples_ = 0, aboveSamples_ = 0;
        unsigned kernelHalf_ = 0;
        unsigned mainLobeHalf_ = 0;// slots inside the -3 dB beam: where labels are taken
        double elevationSpan_ = 0.0;
        // The horizon samples' range interval and the jitter seed (baked).
        float minRange_ = 20.f, maxRange_ = 1852.f;
        std::uint64_t seed_ = 0;
        echo::ArrayPattern hPattern_;
        echo::ArrayPattern vPattern_;
        std::vector<float> kernel_;   // two-way azimuth voltage per slot offset, [-K, K]
        std::vector<double> tiltMean_;// running mean of the sea tilt modulation per range band

        void finishSpoke(const MarineRadarModel& model, std::vector<std::complex<double>>& sum,
                         std::uint64_t revolutionKey, RadarSpoke& out) const;
    };

    /**
     * A PPI display: the polar image rasterised into an RGBA square, with the
     * sweep, afterglow, optional target trails, range rings, the heading line
     * and own ship at the centre. Head-up (bow at the top) or north-up.
     */
    class RadarPpiDisplay {
    public:
        struct Style {
            float afterglowSeconds = 1.8f;// echo glow behind the sweep
            float trailSeconds = 0.f;     // > 0: fading trails of what moved
            float ringStep = 0.f;         // m between rings; 0 = a quarter of the range
            bool showLabels = false;      // colour echoes by their ground-truth label
        };

        explicit RadarPpiDisplay(unsigned size = 512);

        Style style;

        // `headingDeg` is the bow's world bearing; `sweepDeg` the antenna's
        // relative bearing now; `rangeShown` the radius of the scope in m.
        void draw(const RadarPolarImage& image, double now, float headingDeg, float sweepDeg, bool northUp,
                  float rangeShown);

        [[nodiscard]] unsigned size() const { return size_; }
        // size x size x 4, top row first.
        [[nodiscard]] const std::vector<std::uint8_t>& rgba() const { return rgba_; }

    private:
        unsigned size_;
        std::vector<std::uint8_t> rgba_;
        std::vector<float> trail_;
        std::vector<float> pixRadius_; // px from the centre
        std::vector<float> pixBearing_;// deg clockwise from up
        std::vector<float> glow_;      // per spoke, this draw
        std::vector<std::int32_t> pixSlot_;    // relative slot under each pixel (head up)
        std::vector<std::int32_t> pixBin_;     // range bin under each pixel, -1 off the image
        std::vector<std::uint8_t> pixOverlay_; // 0 none, 1 ring, 2 scope edge, 3 outside
        std::vector<std::int32_t> worldSpoke_; // north up: spoke per world-bearing slot
        std::array<std::array<std::uint8_t, 3>, 256> lut_{};
        float keyRange_ = -1.f, keyMin_ = 0.f, keyMax_ = 0.f, keyRing_ = -1.f;
        unsigned keyBins_ = 0, keySpokes_ = 0;
        bool keyNorthUp_ = false;
        double lastDraw_ = -1.0;
    };

    // ─────────────────────────────────────────────────────────────────────
    template<class RingFn, class LabelFn, class AmpFn, class ExtentFn>
    void MarineRadarSynthesizer::formSpoke(const MarineRadarModel& model, std::int64_t slot, RingFn&& ring,
                                           LabelFn&& labels, AmpFn&& amps, ExtentFn&& extent,
                                           std::uint64_t revolutionKey, RadarSpoke& out) const {
        const unsigned bins = std::max(1u, model.rangeBins);
        std::vector<std::complex<double>> sum(bins);
        out.label.assign(bins, -1);
        std::vector<float> best(bins, 0.f);
        const int K = static_cast<int>(kernelHalf_);
        const int L = static_cast<int>(mainLobeHalf_);
        for (int k = -K; k <= K; ++k) {
            const double w = kernel_[static_cast<std::size_t>(k + K)];
            if (w == 0.0) continue;
            const std::complex<float>* col = ring(slot + k);
            const unsigned n = std::min<unsigned>(bins, static_cast<unsigned>(extent(slot + k)));
            for (unsigned j = 0; j < n; ++j) sum[j] += std::complex<double>(col[j]) * w;
            if (k < -L || k > L) continue;// labels come from inside the beam
            const std::int32_t* lab = labels(slot + k);
            const float* amp = amps(slot + k);
            for (unsigned j = 0; j < n; ++j) {
                const float a = static_cast<float>(std::abs(w)) * amp[j];
                if (a > best[j]) {
                    best[j] = a;
                    out.label[j] = lab[j];
                }
            }
        }
        out.index = slot;
        finishSpoke(model, sum, revolutionKey, out);
    }

}// namespace threepp

#endif//THREEPP_MARINERADARMODEL_HPP
