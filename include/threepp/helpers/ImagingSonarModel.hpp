// Forward-looking imaging sonar as a sonar engineer models one: the
// renderer-free half of ImagingSonar.
//
// The renderer's ray tracer samples the scene densely through the sonar's
// vertical fan and its horizontal swath; every hit becomes a scatterer, and
// the image is formed the way the hardware forms it:
//
//     echo level = SL - 2 TL + target strength       (the sonar equation)
//
//   SL   the transducer's source level, dB re 1 uPa at 1 m;
//   TL   one-way transmission loss: spherical spreading 20 log10 R plus
//        absorption alpha R, with alpha from Francois & Garrison (1982) at the
//        sonar's frequency, the water's temperature, salinity, depth and pH,
//        and the speed of sound from Mackenzie (1981);
//   TS   each scatterer's backscatter: Lambert's law (mu + 10 log10 cos^2) for
//        the diffuse part of a surface and a Kirchhoff facet glint for its
//        specular part, per acoustic material, times the patch of surface the
//        ray stands for.
//
// The echoes are then summed COHERENTLY, each with its two-way phase 2kR and
// a stable random scatterer phase, through the receive array's beam pattern
// (a tapered line array whose -3 dB width is the datasheet's horizontal
// beamwidth, sidelobes and all) and the pulse's matched-filter range
// response. That is where an imaging sonar's look comes from: speckle over
// rough surfaces, bright targets smeared into sidelobe arcs across the fan,
// conical beams that bend straight seabed lines into curves.
//
// On top: volume reverberation from the water column (Sv), ambient and
// receiver noise after the array gain, and the sea surface as a reflector:
// an image-source multipath that puts ghosts of what is near the surface at
// the longer path's range, coherent in calm water and mostly scattered away
// as the surface roughens (the Rayleigh roughness parameter k sigma sin psi).
//
// What it is not, said plainly: scatterers are the surfaces the direct rays
// reach, so the surface-bounce legs assume the image path is unobstructed
// (no second trace), the bistatic legs reuse the monostatic scattering law
// at the image incidence, and there is no refraction (a straight-ray,
// iso-velocity water column). Echo levels are physical units; the scattering
// strengths in the material presets are textbook class values, not a
// calibration of any one seabed.
#ifndef THREEPP_IMAGINGSONARMODEL_HPP
#define THREEPP_IMAGINGSONARMODEL_HPP

#include "threepp/helpers/EchoSynthesis.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace threepp {

    /**
     * The water the sound travels through. Iso-velocity: one sound speed and
     * one absorption for the whole path.
     */
    struct WaterProperties {
        float temperature = 8.f;// deg C
        float salinity = 34.f;  // PSU
        float depth = 10.f;     // m, the depth the pressure terms are evaluated at
        float pH = 8.f;

        // Mackenzie (1981), m/s. Valid 0-30 deg C, 25-40 PSU, 0-8000 m.
        [[nodiscard]] double soundSpeed() const;
        // Francois & Garrison (1982): boric acid + MgSO4 relaxations + pure
        // water viscosity, dB/km at `frequencyHz`. Valid ~0.2 kHz - 1 MHz.
        [[nodiscard]] double absorptionDbPerKm(double frequencyHz) const;
    };

    /**
     * How a surface scatters sound back to the sonar, per unit area:
     *
     *   sigma0(theta) = 10^(mu/10) cos^2 theta                              (Lambert)
     *                 + |R|^2 exp(-tan^2 theta / s^2) / (4 pi s^2 cos^4 theta)   (glint)
     *
     * per unit area per steradian, the acoustic convention (a radar's sigma0
     * is 4 pi times it). theta is the incidence angle from the surface normal,
     * mu the Lambert parameter (dB), |R|^2 the facet's normal-incidence power
     * reflection and s^2 its mean-square slope. The glint is what makes a
     * steel hull or a flat rock face flare when it faces the sonar; over a
     * smooth sphere of radius a it integrates to TS = 10 log10(|R|^2 a^2 / 4),
     * the calibration sphere's (pinned in EchoSensors_test).
     */
    struct AcousticMaterial {
        float lambertDb = -27.f;
        float reflectionLossDb = -10.f;// 10 log10 |R|^2; 0 = a perfect reflector
        float rmsSlope = 0.15f;        // sqrt(mean-square slope); <= 0 turns the glint off

        [[nodiscard]] double sigma0(double cosTheta) const;

        // Class values of the right order for high-frequency work (sand's
        // mu = -27 dB is Mackenzie's 1961 Lambert constant; the rest are
        // picked inside the usual ranges). Not a calibration of any seabed.
        static AcousticMaterial Mud() { return {-40.f, -16.f, 0.10f}; }
        static AcousticMaterial Sand() { return {-27.f, -10.f, 0.15f}; }
        static AcousticMaterial Gravel() { return {-22.f, -8.f, 0.25f}; }
        static AcousticMaterial Rock() { return {-18.f, -5.f, 0.30f}; }
        static AcousticMaterial Concrete() { return {-20.f, -5.f, 0.12f}; }
        static AcousticMaterial Steel() { return {-14.f, -0.5f, 0.06f}; }
        // A fish with a swim bladder: bright for its size (salmon TS of about
        // -30 dB over ~0.1 m^2 of flank).
        static AcousticMaterial Fish() { return {-20.f, -6.f, 0.35f}; }
        // Net panels: a fine diffuse scatterer, no glint.
        static AcousticMaterial Net() { return {-22.f, -30.f, 0.f}; }
        // The sea surface seen from below: a pressure-release boundary
        // (|R| = 1), rough.
        static AcousticMaterial SeaSurface() { return {-30.f, 0.f, 0.12f}; }
    };

    /**
     * Acoustic material per target, keyed on the STABLE instance id the
     * return carries (VulkanRenderer::setObjectInstanceId). Optical material
     * is ignored on purpose: black paint on steel is bright to sound.
     */
    struct AcousticMaterials {
        AcousticMaterial defaultMaterial = AcousticMaterial::Sand();
        std::unordered_map<std::int32_t, AcousticMaterial> byInstance;

        void set(std::int32_t instanceId, const AcousticMaterial& m) { byInstance[instanceId] = m; }
        [[nodiscard]] const AcousticMaterial& of(std::int32_t instanceId) const {
            const auto it = byInstance.find(instanceId);
            return it == byInstance.end() ? defaultMaterial : it->second;
        }
    };

    /**
     * Everything that defines one forward-looking multibeam imaging sonar.
     *
     * Frame: the LIDAR / camera convention. Forward is the sensor's local -Z,
     * local +X is the operator's right, local +Y up. The receive array lies
     * along local X, so a beam is a CONE of constant u = (direction . +X) and
     * its bearing is asin(u) — the reason a flat seabed's range lines curve on
     * a real sonar's fan. Tilt the node to tilt the fan.
     *
     * Two kinds of field: GEOMETRY (swath, beams, beamwidths, taper, the ray
     * sampling, seed) is baked into the ray table and the beam patterns when a
     * synthesizer is built; everything else is read on every ping.
     */
    struct ImagingSonarModel {
        // ── Transducer ───────────────────────────────────────────────────
        double frequency = 750e3;   // Hz
        float sourceLevelDb = 206.f;// dB re 1 uPa at 1 m

        // ── Geometry (baked: ImagingSonar re-bakes it at the next ping) ──
        float horizontalFov = 130.f;      // deg, swath of beam centres
        unsigned beams = 512;
        float horizontalBeamwidth = 1.0f; // deg, -3 dB receive beam at broadside
        echo::Taper horizontalTaper = echo::Taper::Taylor25;
        float verticalBeamwidth = 20.f;   // deg, -3 dB, one-way (tx and rx alike)
        // Ray sampling: columns across the swath per beam, rays per column
        // through the vertical fan (out to its first nulls).
        unsigned azimuthOversample = 2;
        unsigned verticalSamples = 48;
        // Half-width of the beam-pattern kernel, in beamwidths. Five keeps the
        // main lobe and the first several sidelobes (Taylor: ~-25..-33 dB).
        float patternSpanBeamwidths = 5.f;
        std::uint64_t seed = 0x51D3F00DULL;

        // ── Range ────────────────────────────────────────────────────────
        float minRange = 0.3f;// blind zone, m
        // Where the trace starts, m: the transducer's face. Everything past it
        // occludes, the blind zone included (see MarineRadarModel::housingRange).
        float housingRange = 0.05f;
        float maxRange = 40.f;
        unsigned rangeBins = 1024;
        float rangeResolution = 0.025f;// m: c tau / 2 (CW) or c / 2B (chirp)
        echo::PulseKind pulse = echo::PulseKind::Chirp;

        // ── Medium and noise ─────────────────────────────────────────────
        WaterProperties water;
        float volumeBackscatterDb = -78.f;// Sv, dB re 1 m^2/m^3: clear coastal water
        // Ambient + self-noise spectrum level, dB re 1 uPa^2/Hz. At several
        // hundred kHz thermal noise sets it: -15 + 20 log10(f_kHz) ~ 43 dB.
        float noiseSpectrumLevelDb = 45.f;

        // ── Sea surface multipath (image source) ─────────────────────────
        bool surfaceMultipath = true;
        float surfaceY = 0.f;           // world y of the mean sea surface
        float surfaceRmsHeight = 0.05f; // m, of the roughness the coherent bounce sees
        // Of the energy the rough surface scatters out of the specular
        // direction, how much still reaches the image path (incoherently).
        float incoherentSurfaceLossDb = 12.f;

        // ── Display: TVG, gain and log compression ───────────────────────
        float tvgSpreading = 30.f;// dB per decade of range (30 log R suits a seabed)
        float gainDb = 0.f;       // brightens: lowers the black level
        float dynamicRangeDb = 45.f;

        [[nodiscard]] double wavelength() const { return water.soundSpeed() / frequency; }
        [[nodiscard]] float binWidth() const {
            return rangeBins ? (maxRange - minRange) / static_cast<float>(rangeBins) : 0.f;
        }
        [[nodiscard]] float beamBearingDeg(unsigned beam) const {
            return -0.5f * horizontalFov + horizontalFov * (static_cast<float>(beam) + 0.5f) /
                                                   static_cast<float>(beams);
        }

        // ── Presets: geometry and band from the datasheets; source level,
        //    taper and noise are class values (manufacturers do not publish
        //    them), marked as such here ───────────────────────────────────
        // Blueprint Oculus M750d, 750 kHz mode: 130 x 20 deg, 512 beams, 1 deg.
        static ImagingSonarModel OculusM750d();
        // Oculus M750d, 1.2 MHz mode: 70 x 12 deg, 512 beams, 0.6 deg, 40 m.
        static ImagingSonarModel OculusM750dHigh();
        // Tritech Gemini 720is: 120 x 20 deg, 512 beams, 720 kHz, 0.5 deg.
        static ImagingSonarModel Gemini720is();
        // Teledyne BlueView M900-130: 130 x 20 deg, 768 beams, 900 kHz, 1 deg.
        static ImagingSonarModel BlueViewM900();
    };

    // What produced the strongest echo in a cell.
    enum class SonarEchoKind : std::uint8_t {
        None = 0,
        Direct = 1,   //!< out and back along the ray
        Multipath = 2,//!< one or both legs via the sea surface
        Volume = 3,   //!< water-column reverberation
    };

    /**
     * One ping: echo level per (beam, range bin), beam-major, and the ground
     * truth behind it. Range bins span [minRange, maxRange].
     */
    struct SonarFrame {
        unsigned beams = 0;
        unsigned bins = 0;
        float minRange = 0.f;
        float maxRange = 0.f;
        double time = 0.0;
        std::vector<float> bearingsDeg;     // per beam
        std::vector<float> levelDb;         // echo level, dB re 1 uPa
        std::vector<float> display;         // [0, 1] after TVG, gain, log compression
        std::vector<std::int32_t> label;    // instance id of the strongest echo; -1 none, -2 volume
        std::vector<std::uint8_t> labelKind;// SonarEchoKind of the strongest echo
        // The noise floor in the band after the array gain, dB re 1 uPa.
        float noiseLevelDb = 0.f;

        [[nodiscard]] std::size_t index(unsigned beam, unsigned bin) const {
            return static_cast<std::size_t>(beam) * bins + bin;
        }
        [[nodiscard]] float rangeOfBin(unsigned bin) const {
            return minRange + (static_cast<float>(bin) + 0.5f) / static_cast<float>(bins) * (maxRange - minRange);
        }
    };

    /**
     * Builds the ray table and the beam patterns for one sonar geometry and
     * folds traced returns into SonarFrames. Stateless across pings: the same
     * returns, pose and ping key give the same frame bit for bit, whatever
     * the thread count (each beam is summed by one thread, in a fixed order).
     */
    class ImagingSonarSynthesizer {
    public:
        explicit ImagingSonarSynthesizer(const ImagingSonarModel& model);

        // Sensor-frame unit directions, column-major: ray = column * rows + row;
        // column 0 is the left edge of the swath, row 0 the bottom of the fan.
        [[nodiscard]] const std::vector<Vector3>& rayDirections() const { return dirs_; }
        [[nodiscard]] unsigned rayCount() const { return static_cast<unsigned>(dirs_.size()); }
        [[nodiscard]] unsigned columns() const { return columns_; }
        [[nodiscard]] unsigned rows() const { return rows_; }
        [[nodiscard]] const echo::ArrayPattern& horizontalPattern() const { return hPattern_; }
        [[nodiscard]] const echo::ArrayPattern& verticalPattern() const { return vPattern_; }

        /**
         * Fold one ping. `returns` holds one return per ray in rayDirections()
         * order, traced from `origin` with the directions rotated by
         * `worldFromSensor`. `pingKey` seeds the noise and reverberation.
         * `threads` 0 = hardware concurrency (capped at 8).
         */
        void synthesize(const ImagingSonarModel& model, const AcousticMaterials& materials,
                        const std::vector<LidarReturn>& returns, const Vector3& origin,
                        const Matrix3& worldFromSensor, std::uint64_t pingKey, SonarFrame& out,
                        unsigned threads = 0) const;

    private:
        unsigned columns_ = 0, rows_ = 0;
        double elevationSpan_ = 0.0;// half-span of the ray fan, radians
        double columnWidth_ = 0.0;  // radians
        std::vector<Vector3> dirs_;
        echo::ArrayPattern hPattern_;
        echo::ArrayPattern vPattern_;
    };

    /**
     * The fan as an operator sees it: apex at the bottom, range up, bearing
     * across, the display values through a copper palette, with range rings
     * and the swath edges. RGBA, top row first.
     */
    class SonarFanDisplay {
    public:
        SonarFanDisplay(unsigned width = 640, unsigned height = 400);

        bool showLabels = false;// colour echoes by their ground-truth label
        float ringStep = 0.f;   // m between rings; 0 = a round step for the range

        // `fovDeg` is the swath the frame's bearings span.
        void draw(const SonarFrame& frame, float fovDeg);

        [[nodiscard]] unsigned width() const { return w_; }
        [[nodiscard]] unsigned height() const { return h_; }
        [[nodiscard]] const std::vector<std::uint8_t>& rgba() const { return rgba_; }

    private:
        unsigned w_, h_;
        std::vector<std::int32_t> cellOf_;// per pixel: beam * bins + bin, -1 outside
        std::vector<std::int32_t> pixBeam_;// the lower of the two beams a pixel falls between
        std::vector<float> pixFrac_;
        std::vector<std::int32_t> pixBin0_;
        std::vector<std::uint8_t> pixBins_;
        std::vector<std::uint8_t> overlay_;
        std::vector<std::uint8_t> rgba_;
        unsigned keyBeams_ = 0, keyBins_ = 0;
        float keyFov_ = 0.f, keyMin_ = 0.f, keyMax_ = 0.f, keyRing_ = -1.f;
    };

}// namespace threepp

#endif//THREEPP_IMAGINGSONARMODEL_HPP
