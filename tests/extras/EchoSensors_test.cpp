// The renderer-free physics of ImagingSonar and MarineRadar, pinned without a
// GPU: the beam patterns' widths and sidelobes, the water's sound speed and
// absorption, the NRL sea clutter against its published worked value, where a
// point target lands in the image, the surface-multipath ghost, two-ray
// lobing over the sea, the radar equation's range law, that a frame is a
// pure function of its inputs whatever the thread count, and, end to end, the
// sonar and radar equations for the sphere real sensors are calibrated with.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/helpers/EchoSynthesis.hpp"
#include "threepp/helpers/ImagingSonarModel.hpp"
#include "threepp/helpers/MarineRadarModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    constexpr double kDeg = 3.14159265358979323846 / 180.0;

    // A small, quiet sonar: few rays, no noise floor to speak of.
    ImagingSonarModel quietSonar() {
        ImagingSonarModel m;
        m.horizontalFov = 60.f;
        m.beams = 64;
        m.horizontalBeamwidth = 2.f;
        m.verticalBeamwidth = 20.f;
        m.azimuthOversample = 2;
        m.verticalSamples = 16;
        m.maxRange = 20.f;
        m.minRange = 0.5f;
        m.rangeBins = 256;
        m.rangeResolution = 0.05f;
        m.noiseSpectrumLevelDb = -150.f;
        m.volumeBackscatterDb = -300.f;
        m.surfaceMultipath = false;
        return m;
    }

    LidarReturn hitAt(const Vector3& origin, const Vector3& dir, float range, int id) {
        LidarReturn r{};
        r.position = Vector3(dir).multiplyScalar(range).add(origin);
        r.normal = Vector3(dir).multiplyScalar(-1.f);
        r.distance = range;
        r.intensity = 1.f;
        r.hitInstanceId = id;
        r.returnNo = 1;
        return r;
    }

    // Every ray whose azimuth is within `halfDeg` of `azDeg` strikes a
    // surface at `range`; every other ray misses.
    std::vector<LidarReturn> targetColumn(const ImagingSonarSynthesizer& syn, const Vector3& origin,
                                          float azDeg, float halfDeg, float range, int id) {
        std::vector<LidarReturn> out(syn.rayCount());
        const auto& dirs = syn.rayDirections();
        for (std::size_t i = 0; i < dirs.size(); ++i) {
            const double az = std::atan2(dirs[i].x, -dirs[i].z) / kDeg;
            if (std::abs(az - azDeg) <= halfDeg) out[i] = hitAt(origin, dirs[i], range, id);
        }
        return out;
    }

    std::size_t argmax(const std::vector<float>& v) {
        return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
    }

}// namespace

TEST_CASE("ArrayPattern: the beamwidth it was asked for, and the taper's sidelobes") {
    const echo::ArrayPattern uni(1.0, echo::Taper::Uniform, 0.2);
    CHECK_THAT(uni.halfPowerU(), WithinAbs(std::sin(0.5 * kDeg), 0.03 * std::sin(0.5 * kDeg)));
    CHECK_THAT(uni.peakSidelobeDb(), WithinAbs(-13.3, 0.5));

    const echo::ArrayPattern taylor(1.0, echo::Taper::Taylor25, 0.2);
    CHECK_THAT(taylor.halfPowerU(), WithinAbs(std::sin(0.5 * kDeg), 0.03 * std::sin(0.5 * kDeg)));
    CHECK(taylor.peakSidelobeDb() < -24.0);
    CHECK(taylor.peakSidelobeDb() > -27.0);

    const echo::ArrayPattern hann(1.0, echo::Taper::Hann, 0.2);
    CHECK_THAT(hann.peakSidelobeDb(), WithinAbs(-31.5, 1.0));
    // Same beamwidth, heavier taper: a longer array.
    CHECK(hann.apertureWavelengths() > uni.apertureWavelengths() * 1.4);

    CHECK_THAT(uni(0.0), WithinAbs(1.0, 1e-9));
    CHECK(uni(0.25) == 0.0);// past the span
}

TEST_CASE("RangeResponse: CW triangle, Hann-weighted chirp") {
    const echo::RangeResponse cw(0.1, echo::PulseKind::CW);
    CHECK_THAT(cw(0.0), WithinAbs(1.0, 1e-12));
    CHECK_THAT(cw(0.05), WithinAbs(0.5, 1e-12));
    CHECK(cw(0.1) == 0.0);

    const echo::RangeResponse chirp(0.1, echo::PulseKind::Chirp);
    CHECK_THAT(chirp(0.0), WithinAbs(1.0, 1e-12));
    CHECK(std::abs(chirp(0.2)) < 1e-3);// first null at 2 c/2B
    double side = 0.0;
    for (double x = 0.21; x < 0.4; x += 0.001) side = std::max(side, std::abs(chirp(x)));
    CHECK(20.0 * std::log10(side) < -30.0);
}

TEST_CASE("Water: Mackenzie sound speed and Francois-Garrison absorption") {
    WaterProperties w;
    w.temperature = 10.f;
    w.salinity = 35.f;
    w.depth = 0.f;
    CHECK_THAT(w.soundSpeed(), WithinAbs(1489.8, 0.5));

    w.depth = 10.f;
    const double a750 = w.absorptionDbPerKm(750e3);
    CHECK(a750 > 190.0);
    CHECK(a750 < 250.0);
    const double a100 = w.absorptionDbPerKm(100e3);
    CHECK(a100 > 28.0);
    CHECK(a100 < 40.0);
    CHECK(w.absorptionDbPerKm(1.2e6) > a750);

    // Mackenzie's own check value (J. Acoust. Soc. Am. 70, 807, 1981).
    WaterProperties check;
    check.temperature = 25.f;
    check.salinity = 35.f;
    check.depth = 1000.f;
    CHECK_THAT(check.soundSpeed(), WithinAbs(1550.744, 0.01));

    // Francois-Garrison against the independent fit of Ainslie & McColm
    // (J. Acoust. Soc. Am. 103, 1671, 1998), which tracks it within ~10 %.
    auto ainslieMcColm = [](double fkHz, double T, double S, double zKm, double pH) {
        const double f2 = fkHz * fkHz;
        const double f1 = 0.78 * std::sqrt(S / 35.0) * std::exp(T / 26.0);
        const double fm = 42.0 * std::exp(T / 17.0);
        return 0.106 * f1 * f2 / (f1 * f1 + f2) * std::exp((pH - 8.0) / 0.56) +
               0.52 * (1.0 + T / 43.0) * (S / 35.0) * fm * f2 / (fm * fm + f2) * std::exp(-zKm / 6.0) +
               0.00049 * f2 * std::exp(-(T / 27.0 + zKm / 17.0));
    };
    for (const double fkHz : {10.0, 100.0, 300.0, 750.0}) {
        const double fg = w.absorptionDbPerKm(fkHz * 1e3);
        const double am = ainslieMcColm(fkHz, w.temperature, w.salinity, w.depth * 1e-3, w.pH);
        INFO(fkHz << " kHz: Francois-Garrison " << fg << " dB/km, Ainslie-McColm " << am << " dB/km");
        CHECK(std::abs(fg / am - 1.0) < 0.10);
    }
}

TEST_CASE("NRL sea clutter: the published worked value and its trends") {
    // MathWorks' seareflectivity example: NRL, H, sea state 2, 10 deg, 30 GHz.
    CHECK_THAT(nrlSeaClutterDb(10.0, 30.0, 2.0, RadarPolarization::HH), WithinAbs(-36.66, 0.02));

    const double x1 = nrlSeaClutterDb(1.0, 9.41, 3.0, RadarPolarization::HH);
    CHECK(x1 < -38.0);
    CHECK(x1 > -50.0);
    CHECK(nrlSeaClutterDb(3.0, 9.41, 3.0, RadarPolarization::HH) > x1);
    CHECK(nrlSeaClutterDb(1.0, 9.41, 5.0, RadarPolarization::HH) > x1);
}

TEST_CASE("ImagingSonar: a point target lands at its bearing and range, labelled") {
    const auto model = quietSonar();
    const ImagingSonarSynthesizer syn(model);
    const Vector3 origin(0, 0, 0);
    const auto returns = targetColumn(syn, origin, 15.f, 0.3f, 10.f, 5);

    SonarFrame f;
    syn.synthesize(model, AcousticMaterials{}, returns, origin, Matrix3{}, 1, f, 1);
    const std::size_t peak = argmax(f.levelDb);
    const unsigned beam = static_cast<unsigned>(peak / f.bins);
    const unsigned bin = static_cast<unsigned>(peak % f.bins);
    const float spacing = model.horizontalFov / model.beams;
    CHECK(std::abs(f.bearingsDeg[beam] - 15.f) <= 1.5f * spacing);
    CHECK(std::abs(f.rangeOfBin(bin) - 10.f) <= 2.f * model.binWidth());
    CHECK(f.label[peak] == 5);
    CHECK(f.labelKind[peak] == static_cast<std::uint8_t>(SonarEchoKind::Direct));

    // The sidelobes put the target into beams well off its bearing, far down.
    const unsigned far = beam > 20 ? beam - 12 : beam + 12;
    const float off = f.levelDb[f.index(far, bin)];
    CHECK(off < f.levelDb[peak] - 20.f);
}

TEST_CASE("ImagingSonar: the frame is a pure function of its inputs") {
    auto model = quietSonar();
    model.noiseSpectrumLevelDb = 45.f;
    model.volumeBackscatterDb = -70.f;
    const ImagingSonarSynthesizer syn(model);
    const Vector3 origin(0, 0, 0);
    const auto returns = targetColumn(syn, origin, -8.f, 4.f, 7.5f, 3);

    SonarFrame a, b, c;
    syn.synthesize(model, AcousticMaterials{}, returns, origin, Matrix3{}, 42, a, 1);
    syn.synthesize(model, AcousticMaterials{}, returns, origin, Matrix3{}, 42, b, 4);
    syn.synthesize(model, AcousticMaterials{}, returns, origin, Matrix3{}, 43, c, 4);
    CHECK(a.levelDb == b.levelDb);
    CHECK(a.label == b.label);
    CHECK(a.levelDb != c.levelDb);// another ping, other noise
}

TEST_CASE("ImagingSonar: the sea surface puts a ghost at the image path's range") {
    auto model = quietSonar();
    model.surfaceY = 0.f;
    model.surfaceRmsHeight = 1e-5f;// glassy: a coherent mirror
    const ImagingSonarSynthesizer syn(model);
    const Vector3 origin(0, -2.f, 0);
    const auto returns = targetColumn(syn, origin, 0.f, 0.5f, 10.f, 7);

    auto ghostPeak = [&](bool on) {
        auto m = model;
        m.surfaceMultipath = on;
        SonarFrame f;
        syn.synthesize(m, AcousticMaterials{}, returns, origin, Matrix3{}, 1, f, 2);
        // The double bounce of a level hit 10 m out from 2 m deep: sqrt(10^2 + 4^2).
        const float rGhost = std::sqrt(100.f + 16.f);
        float best = -1e9f;
        std::uint8_t kind = 0;
        for (unsigned b = 0; b < f.beams; ++b) {
            for (unsigned j = 0; j < f.bins; ++j) {
                if (std::abs(f.rangeOfBin(j) - rGhost) > 0.15f) continue;
                if (f.levelDb[f.index(b, j)] > best) {
                    best = f.levelDb[f.index(b, j)];
                    kind = f.labelKind[f.index(b, j)];
                }
            }
        }
        return std::make_pair(best, kind);
    };
    const auto on = ghostPeak(true);
    const auto off = ghostPeak(false);
    CHECK(on.first > off.first + 30.f);
    CHECK(on.second == static_cast<std::uint8_t>(SonarEchoKind::Multipath));
}

namespace {

    // One ray of a radar slot aimed at `p` from `origin`, every other ray a miss.
    struct OneRay {
        std::vector<LidarReturn> returns;
        std::vector<Vector3> dirs;
        std::vector<float> solid;
    };

    OneRay oneRayAt(const MarineRadarSynthesizer& syn, const Vector3& origin, const Vector3& p) {
        OneRay o;
        const unsigned n = syn.raysPerSlot();
        o.returns.resize(n);
        o.dirs.assign(n, Vector3(0, 0, -1));
        o.solid.assign(n, 0.f);
        Vector3 d = p;
        d.sub(origin);
        const float R = d.length();
        d.normalize();
        o.dirs[0] = d;
        o.solid[0] = 1e-6f;
        o.returns[0] = hitAt(origin, d, R, 9);
        return o;
    }

    double columnPower(MarineRadarSynthesizer& syn, const MarineRadarModel& m, const RadarMaterials& mats,
                       const Vector3& origin, const Vector3& p, std::int32_t* labelOut = nullptr) {
        const auto o = oneRayAt(syn, origin, p);
        std::vector<std::complex<float>> col(m.rangeBins);
        std::vector<std::int32_t> lab(m.rangeBins);
        std::vector<float> amp(m.rangeBins);
        syn.foldSlot(m, mats, 0, o.returns.data(), o.dirs.data(), o.solid.data(), origin, Matrix3{}, 1, col.data(),
                     lab.data(), amp.data());
        double s = 0.0;
        for (const auto& c : col) s += std::norm(std::complex<double>(c));
        if (labelOut) *labelOut = lab[static_cast<std::size_t>(std::max_element(amp.begin(), amp.end()) - amp.begin())];
        return s;
    }

    // Ground range at which the two-ray path difference equals `delta`.
    double groundRangeFor(double h, double ht, double delta) {
        double lo = 1.0, hi = 1e6;
        for (int i = 0; i < 200; ++i) {
            const double mid = 0.5 * (lo + hi);
            const double d = std::sqrt(mid * mid + (h + ht) * (h + ht)) - std::sqrt(mid * mid + (h - ht) * (h - ht));
            (d > delta ? lo : hi) = mid;
        }
        return 0.5 * (lo + hi);
    }

}// namespace

TEST_CASE("MarineRadar: one ray of a wall follows R^-2, its label rides along") {
    MarineRadarModel m;
    m.multipath = false;
    m.rangeBins = 1000;
    m.minRange = 20.f;
    m.maxRange = 1020.f;// 1 m bins: both ranges below sit on a bin centre
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    mats.set(9, RadarMaterial::Metal());
    const Vector3 origin(0, 5, 0);
    std::int32_t label = -1;
    const double p1 = columnPower(syn, m, mats, origin, Vector3(0, 5, -300.5f), &label);
    const double p2 = columnPower(syn, m, mats, origin, Vector3(0, 5, -600.5f));
    CHECK(label == 9);
    // A ray stands for a fixed solid angle: its patch grows as R^2, so the
    // return falls as R^-2 (a wall), not R^-4 (a point).
    CHECK_THAT(10.0 * std::log10(p1 / p2), WithinAbs(20.0 * std::log10(600.5 / 300.5), 0.05));
}

TEST_CASE("MarineRadar: two-ray lobing over a glassy sea") {
    MarineRadarModel m;
    m.seaState = 0.f;// no roughness: the full Fresnel reflection
    m.rangeBins = 1024;
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    mats.set(9, RadarMaterial::Metal());
    const double h = 3.0, ht = 2.0;
    const Vector3 origin(0, static_cast<float>(h), 0);
    const double lambda = m.wavelength();

    auto gainDb = [&](double delta) {
        const double Rg = groundRangeFor(h, ht, delta);
        const Vector3 p(0, static_cast<float>(ht), static_cast<float>(-Rg));
        auto on = m;
        auto off = m;
        off.multipath = false;
        MarineRadarSynthesizer a(on), b(off);
        return 10.0 * std::log10(columnPower(a, on, mats, origin, p) / columnPower(b, off, mats, origin, p));
    };
    // Half a wavelength of path difference + the sea's ~180 deg phase flip:
    // the two rays add, F ~ 2, +12 dB two-way. A whole wavelength: they cancel.
    const double peak = gainDb(0.5 * lambda);
    CHECK(peak > 10.5);
    CHECK(peak < 12.1);
    CHECK(gainDb(lambda) < -15.0);
}

TEST_CASE("MarineRadar: a slot's solid angles cover its fan") {
    MarineRadarModel m;
    MarineRadarSynthesizer syn(m);
    std::vector<Vector3> dirs;
    std::vector<float> solid;
    syn.slotRays(17, 3.0, dirs, solid);
    REQUIRE(dirs.size() == syn.raysPerSlot());
    double sum = 0.0;
    double maxEl = 0.0;
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        sum += solid[i];
        maxEl = std::max(maxEl, std::abs(std::asin(static_cast<double>(dirs[i].y))));
    }
    const double slot = 2.0 * 3.14159265358979323846 / m.spokes;
    // The rays reach to the vertical fan's first nulls, symmetric about level.
    const double expected = slot * 2.0 * std::sin(maxEl);
    CHECK_THAT(sum, WithinAbs(expected, 0.08 * expected));
}

TEST_CASE("MarineRadar: a spoke forms from the ring and the PPI draws it") {
    MarineRadarModel m;
    m.rangeBins = 256;
    MarineRadarSynthesizer syn(m);
    const unsigned K = syn.kernelHalfSlots();
    REQUIRE(K > 0);
    std::vector<std::complex<float>> zero(m.rangeBins), hot(m.rangeBins);
    std::vector<std::int32_t> lab(m.rangeBins, -1), labHot(m.rangeBins, -1);
    std::vector<float> amp(m.rangeBins, 0.f), ampHot(m.rangeBins, 0.f);
    hot[100] = {1e-3f, 0.f};// far above kTBF
    labHot[100] = 4;
    ampHot[100] = 1e-3f;
    RadarSpoke sp;
    syn.formSpoke(
            m, 50,
            [&](std::int64_t s) { return s == 50 ? hot.data() : zero.data(); },
            [&](std::int64_t s) { return s == 50 ? labHot.data() : lab.data(); },
            [&](std::int64_t s) { return s == 50 ? ampHot.data() : amp.data(); }, 7, sp);
    CHECK(argmax(sp.snrDb) == 100);
    CHECK(sp.label[100] == 4);
    CHECK(sp.display[100] > 0.9f);

    RadarPolarImage img;
    img.reset(m.spokes, m.rangeBins, m.minRange, m.maxRange);
    sp.bearingDeg = 0.f;
    sp.worldBearingDeg = 0.f;
    img.store(sp);
    RadarPpiDisplay ppi(128);
    ppi.draw(img, 0.0, 0.f, 0.f, false, m.maxRange);
    const auto& px = ppi.rgba();
    REQUIRE(px.size() == 128u * 128u * 4u);
    const std::size_t centre = (64u * 128u + 64u) * 4u;
    CHECK(px[centre] == 255);// own ship
}

// ── Calibration: the sphere ────────────────────────────────────────────────
// Real sonars are calibrated with a tungsten-carbide or copper sphere and
// radars with a metal one: a target whose echo is known in closed form. A
// smooth sphere of radius a backscatters
//     sonar:  TS = 10 log10(a^2 / 4)        (per steradian; Urick)
//     radar:  RCS = pi a^2                   (geometric optics; Skolnik)
// so the synthesizers, end to end, must return the sonar and radar equations
// for it: rays dense enough to resolve its glint spot, speckle averaged over
// seeds (each seed its own ray jitter and scatterer phases).

namespace {

    constexpr double kPiD = 3.14159265358979323846;

    // The first hit of the ray o + t d (d a unit vector) on the sphere |p - c| = a.
    // Hits past 20 deg incidence are left as misses: with the glint's rms
    // slope of 0.1 they carry under 2e-6 of its echo, and skipping them keeps
    // the test quick.
    bool hitSphere(const Vector3& o, const Vector3& d, const Vector3& c, double a, int id, LidarReturn& out) {
        const double ox = o.x - c.x, oy = o.y - c.y, oz = o.z - c.z;
        const double b = ox * d.x + oy * d.y + oz * d.z;
        const double disc = b * b - (ox * ox + oy * oy + oz * oz - a * a);
        if (disc < 0.0) return false;
        const double t = -b - std::sqrt(disc);
        if (t <= 0.0) return false;
        const double px = o.x + t * d.x, py = o.y + t * d.y, pz = o.z + t * d.z;
        const double cosInc = -((px - c.x) * d.x + (py - c.y) * d.y + (pz - c.z) * d.z) / a;
        if (cosInc < 0.9397) return false;// cos 20 deg
        out = LidarReturn{};
        out.position = Vector3(static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz));
        out.normal = Vector3(static_cast<float>((px - c.x) / a), static_cast<float>((py - c.y) / a),
                             static_cast<float>((pz - c.z) / a));
        out.distance = static_cast<float>(t);
        out.intensity = 1.f;
        out.hitInstanceId = id;
        out.returnNo = 1;
        return true;
    }

    // A sphere's backscatter from a per-area law: 2 pi a^2 int_0^{pi/2} sigma0 sin(theta) d theta.
    template<class Sigma0>
    double sphereIntegral(Sigma0&& sigma0, double a) {
        constexpr int n = 200000;
        const double h = 0.5 * kPiD / n;
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            const double th = (i + 0.5) * h;
            s += sigma0(std::cos(th)) * std::sin(th);
        }
        return 2.0 * kPiD * a * a * s * h;
    }

    // The matched filter's output for a scatterer at a bin centre, averaged
    // over the bin's four taps as the synthesizers deposit it.
    double binWeight(const echo::RangeResponse& resp, double binW) {
        double w = 0.0;
        for (int t = 0; t < 4; ++t) w += resp(((t + 0.5) * 0.25 - 0.5) * binW);
        return 0.25 * w;
    }

    // A slightly rough sphere: the small-slope correction to the
    // geometric-optics limit, (1 + s^2 / 2) for an rms slope s.
    double roughSphere(double s) { return 1.0 + 0.5 * s * s; }

}// namespace

TEST_CASE("Calibration: the facet glint over a sphere is the geometric-optics sphere") {
    const double a = 0.5, s = 0.1;
    const AcousticMaterial steel{-300.f, 0.f, static_cast<float>(s)};
    const double ts = 10.0 * std::log10(sphereIntegral([&](double c) { return steel.sigma0(c); }, a));
    INFO("sonar sphere TS " << ts << " dB");
    CHECK_THAT(ts, WithinAbs(10.0 * std::log10(a * a / 4.0 * roughSphere(s)), 0.05));

    const RadarMaterial metal{-300.f, 0.f, static_cast<float>(s), -400.f, false};
    const double rcs = 10.0 * std::log10(sphereIntegral([&](double c) { return metal.sigma0(c); }, a));
    INFO("radar sphere RCS " << rcs << " dBsm");
    CHECK_THAT(rcs, WithinAbs(10.0 * std::log10(kPiD * a * a * roughSphere(s)), 0.05));
}

TEST_CASE("ImagingSonar: a calibration sphere echoes at the sonar equation's level") {
    // EL = SL - 40 log10 R - 2 alpha R + TS.
    ImagingSonarModel m = quietSonar();
    m.horizontalFov = 1.5f;
    m.beams = 7;// beam 3 looks straight ahead
    m.horizontalBeamwidth = 1.f;
    m.verticalBeamwidth = 1.f;
    m.azimuthOversample = 24;// rays 1.4 mm apart at 10 m; the glint spot is 20 mm across
    m.verticalSamples = 180;
    m.minRange = 5.f;
    m.maxRange = 15.f;
    m.rangeBins = 800;// 12.5 mm bins, four to the 50 mm resolution
    m.rangeResolution = 0.05f;
    const unsigned beam = 3, bin = 400;
    const double a = 0.1, s = 0.1;
    const double front = m.minRange + (bin + 0.5) * m.binWidth();
    const Vector3 origin(0, 0, 0);
    const Vector3 centre(0, 0, static_cast<float>(-(front + a)));
    AcousticMaterials mats;
    mats.set(7, AcousticMaterial{-300.f, 0.f, static_cast<float>(s)});

    constexpr int seeds = 192;
    double sum = 0.0;
    int labelled = 0;
    SonarFrame f;
    // One ray table; each seed draws the scatterers' phases afresh.
    const ImagingSonarSynthesizer syn(m);
    const auto& dirs = syn.rayDirections();
    std::vector<LidarReturn> returns(dirs.size());
    for (std::size_t i = 0; i < dirs.size(); ++i) hitSphere(origin, dirs[i], centre, a, 7, returns[i]);
    for (int k = 0; k < seeds; ++k) {
        m.seed = 1000 + k;
        syn.synthesize(m, mats, returns, origin, Matrix3{}, 1, f, 1);
        sum += std::pow(10.0, 0.1 * f.levelDb[f.index(beam, bin)]);
        labelled += f.label[f.index(beam, bin)] == 7;
    }
    REQUIRE(std::abs(f.bearingsDeg[beam]) < 1e-4f);
    const double alpha = m.water.absorptionDbPerKm(m.frequency) * 1e-3;// dB/m
    const double ts = 10.0 * std::log10(a * a / 4.0 * roughSphere(s));
    const double w = binWeight(echo::RangeResponse(m.rangeResolution, m.pulse), m.binWidth());
    const double expected =
            m.sourceLevelDb - 40.0 * std::log10(front) - 2.0 * alpha * front + ts + 20.0 * std::log10(w);
    const double measured = 10.0 * std::log10(sum / seeds);
    INFO("sphere echo " << measured << " dB re 1 uPa, sonar equation " << expected << " dB");
    CHECK_THAT(measured, WithinAbs(expected, 1.0));
    CHECK(labelled == seeds);
}

TEST_CASE("MarineRadar: a calibration sphere returns the radar equation's SNR") {
    // SNR = Pt G^2 lambda^2 sigma / ((4 pi)^3 R^4 L k T0 B F), B = 1 / tau.
    MarineRadarModel m;
    m.multipath = false;
    m.verticalBeamwidth = 2.f;
    m.azimuthSamples = 64;// rays 1 cm apart across, 5 cm up, at 200 m; the glint spot is 20 cm
    m.elevationSamples = 300;
    m.horizonShare = 0.f;
    m.aboveHorizonShare = 0.f;
    m.minRange = 20.f;
    m.maxRange = 420.f;
    m.rangeBins = 4000;// 0.1 m bins
    const unsigned bin = 1800;
    const double a = 1.0, s = 0.1;
    const double binW = (m.maxRange - m.minRange) / m.rangeBins;
    const double front = m.minRange + (bin + 0.5) * binW;
    const Vector3 origin(0, 10, 0);
    const Vector3 centre(0, 10, static_cast<float>(-(front + a)));
    RadarMaterials mats;
    mats.set(7, RadarMaterial{-300.f, 0.f, static_cast<float>(s), -400.f, false});

    constexpr int seeds = 192;
    double sum = 0.0;
    int labelled = 0;
    std::vector<std::complex<float>> zero(m.rangeBins);
    std::vector<std::int32_t> none(m.rangeBins, -1);
    std::vector<float> nil(m.rangeBins, 0.f);
    for (int k = 0; k < seeds; ++k) {
        m.seed = 2000 + k;
        MarineRadarSynthesizer syn(m);
        const unsigned n = syn.raysPerSlot();
        // Slot 0 looks straight at the sphere; its neighbours see its limb.
        std::array<std::vector<std::complex<float>>, 3> col;
        std::array<std::vector<std::int32_t>, 3> lab;
        std::array<std::vector<float>, 3> amp;
        for (int sl = -1; sl <= 1; ++sl) {
            std::vector<Vector3> dirs;
            std::vector<float> solid;
            syn.slotRays(sl, origin.y - m.seaLevel, dirs, solid);
            std::vector<LidarReturn> ret(n);
            for (unsigned i = 0; i < n; ++i) hitSphere(origin, dirs[i], centre, a, 7, ret[i]);
            const auto c = static_cast<std::size_t>(sl + 1);
            col[c].resize(m.rangeBins);
            lab[c].resize(m.rangeBins);
            amp[c].resize(m.rangeBins);
            syn.foldSlot(m, mats, sl, ret.data(), dirs.data(), solid.data(), origin, Matrix3{}, 1, col[c].data(),
                         lab[c].data(), amp[c].data());
        }
        auto at = [](std::int64_t sl) { return static_cast<std::size_t>(sl + 1); };
        auto inRing = [](std::int64_t sl) { return sl >= -1 && sl <= 1; };
        RadarSpoke sp;
        syn.formSpoke(
                m, 0, [&](std::int64_t sl) { return inRing(sl) ? col[at(sl)].data() : zero.data(); },
                [&](std::int64_t sl) { return inRing(sl) ? lab[at(sl)].data() : none.data(); },
                [&](std::int64_t sl) { return inRing(sl) ? amp[at(sl)].data() : nil.data(); }, 1, sp);
        sum += std::pow(10.0, 0.1 * sp.snrDb[bin]) - 1.0;// less the noise's own share
        labelled += sp.label[bin] == 7;
    }
    const double lambda = m.wavelength();
    const double G = m.antennaGain();
    const double sigma = kPiD * a * a * roughSphere(s);
    const double kTBF = 1.380649e-23 * 290.0 / m.pulseLength * std::pow(10.0, 0.1 * m.noiseFigureDb);
    const double pr = m.peakPowerW * G * G * lambda * lambda * sigma /
                      (std::pow(4.0 * kPiD, 3.0) * std::pow(front, 4.0) * std::pow(10.0, 0.1 * m.systemLossDb));
    const double w = binWeight(echo::RangeResponse(0.5 * 299792458.0 * m.pulseLength, echo::PulseKind::CW), binW);
    const double expected = 10.0 * std::log10(pr * w * w / kTBF);
    const double measured = 10.0 * std::log10(sum / seeds);
    INFO("sphere SNR " << measured << " dB, radar equation " << expected << " dB");
    CHECK_THAT(measured, WithinAbs(expected, 1.0));
    CHECK(labelled == seeds);
}
