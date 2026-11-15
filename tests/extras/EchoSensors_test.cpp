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
#include "threepp/scenes/Weather.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <chrono>
#include <cstdio>
#include <utility>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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

TEST_CASE("ImagingSonar: a surface just past max range stays in the last bins of its beams") {
    // The default model with only the range cut to 1 m: the chirp's response
    // (~0.1 m each side) is far wider than a bin (0.7 mm), so a surface 1.04 m
    // out, past the window but inside the response, is recentred on its core.
    ImagingSonarModel model;
    model.maxRange = 1.f;
    model.noiseSpectrumLevelDb = -150.f;
    model.volumeBackscatterDb = -300.f;
    model.surfaceMultipath = false;
    const ImagingSonarSynthesizer syn(model);
    const Vector3 origin(0, 0, 0);

    // Two targets inside the window, one each side, and the wall dead ahead.
    auto sides = targetColumn(syn, origin, -40.f, 2.f, 0.6f, 3);
    const auto right = targetColumn(syn, origin, 40.f, 2.f, 0.6f, 4);
    for (std::size_t i = 0; i < sides.size(); ++i) {
        if (right[i].returnNo > 0) sides[i] = right[i];
    }
    auto withWall = sides;
    const auto wall = targetColumn(syn, origin, 0.f, 2.f, 1.04f, 5);
    for (std::size_t i = 0; i < wall.size(); ++i) {
        if (wall[i].returnNo > 0) withWall[i] = wall[i];
    }

    SonarFrame a, b;
    syn.synthesize(model, AcousticMaterials{}, sides, origin, Matrix3{}, 1, a, 1);
    syn.synthesize(model, AcousticMaterials{}, withWall, origin, Matrix3{}, 1, b, 1);
    REQUIRE(a.levelDb.size() == b.levelDb.size());
    // The wall only adds to the bins its response reaches, the last ~150.
    const unsigned firstReached = model.rangeBins - 200;
    unsigned changedEarly = 0, changedLate = 0;
    for (unsigned beam = 0; beam < b.beams; ++beam) {
        for (unsigned j = 0; j < b.bins; ++j) {
            const std::size_t c = b.index(beam, j);
            REQUIRE(std::isfinite(b.levelDb[c]));
            if (a.levelDb[c] == b.levelDb[c]) continue;
            (j < firstReached ? changedEarly : changedLate) += 1;
        }
    }
    CHECK(changedEarly == 0u);
    CHECK(changedLate > 0u);
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

    // The sea state matters only to the sea material and to multipath; the
    // cases that use neither fold with the default weather.
    double columnPower(MarineRadarSynthesizer& syn, const MarineRadarModel& m, const RadarMaterials& mats,
                       const Vector3& origin, const Vector3& p, std::int32_t* labelOut = nullptr,
                       const Weather& weather = Weather{}) {
        const auto o = oneRayAt(syn, origin, p);
        std::vector<std::complex<float>> col(m.rangeBins);
        std::vector<std::int32_t> lab(m.rangeBins);
        std::vector<float> amp(m.rangeBins);
        syn.foldSlot(m, mats, weather, 0.0, 0, o.returns.data(), o.dirs.data(), o.solid.data(), origin, Matrix3{}, 1, col.data(),
                     lab.data(), amp.data(), m.rangeBins);
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

TEST_CASE("MarineRadar: a fold writes no further than the rows it is given") {
    MarineRadarModel m;
    m.multipath = false;
    m.rangeBins = 2048;// the model asks for more bins than the rows hold
    m.minRange = 20.f;
    m.maxRange = 1020.f;
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    const Vector3 origin(0, 5, 0);
    // A wall at the far end of the range, where bins past 1024 would be written.
    const auto o = oneRayAt(syn, origin, Vector3(0, 5, -1000.f));
    constexpr unsigned rows = 1024, guard = 1024;
    std::vector<std::complex<float>> col(rows + guard, {7.f, 7.f});
    std::vector<std::int32_t> lab(rows + guard, 77);
    std::vector<float> amp(rows + guard, 7.f);
    const unsigned extent = syn.foldSlot(m, mats, Weather{}, 0.0, 0, o.returns.data(), o.dirs.data(),
                                         o.solid.data(), origin, Matrix3{}, 1, col.data(), lab.data(), amp.data(), rows);
    CHECK(extent <= rows);
    for (unsigned j = rows; j < rows + guard; ++j) {
        REQUIRE(col[j] == std::complex<float>(7.f, 7.f));
        REQUIRE(lab[j] == 77);
        REQUIRE(amp[j] == 7.f);
    }
}

TEST_CASE("MarineRadar: rain is attenuated by the showers nearer than it, whatever their order in the list") {
    MarineRadarModel m;
    m.multipath = false;
    m.minRange = 20.f;
    m.maxRange = 6020.f;
    m.rangeBins = 1000;// 6 m bins
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    const Vector3 origin(0, 5, 0);
    const auto o = oneRayAt(syn, origin, Vector3(0, 5, -5900.f));

    // Two heavy showers on the ray's bearing, 1.5 and 4 km out, apart.
    RainCell nearCell, farCell, none;
    nearCell.centre = Vector3(0, 0, -1500.f);
    farCell.centre = Vector3(0, 0, -4000.f);
    nearCell.radius = farCell.radius = 300.f;
    nearCell.rainRate = farCell.rainRate = 100.f;
    none.rainRate = 0.f;// skipped: keeps the near shower second in both lists

    auto fold = [&](const std::vector<RainCell>& rain) {
        Weather weather;
        weather.rain = rain;
        std::vector<std::complex<float>> col(m.rangeBins);
        std::vector<std::int32_t> lab(m.rangeBins);
        std::vector<float> amp(m.rangeBins);
        syn.foldSlot(m, mats, weather, 0.0, 0, o.returns.data(), o.dirs.data(), o.solid.data(), origin, Matrix3{},
                     1, col.data(), lab.data(), amp.data(), m.rangeBins);
        return col;
    };
    const auto alone = fold({none, nearCell});
    const auto farFirst = fold({farCell, nearCell});

    // The near shower's own echo (0.6 to 2.4 km) is the same with the far one
    // listed first: nothing lies between it and the antenna.
    double pAlone = 0.0, pFarFirst = 0.0;
    for (unsigned j = 0; j < m.rangeBins; ++j) {
        const double R = m.minRange + (j + 0.5) * m.binWidth();
        if (R < 700.0 || R > 2300.0) continue;
        pAlone += std::norm(std::complex<double>(alone[j]));
        pFarFirst += std::norm(std::complex<double>(farFirst[j]));
    }
    REQUIRE(pAlone > 0.0);
    INFO("near shower's echo: " << 10.0 * std::log10(pFarFirst / pAlone) << " dB with the far one listed first");
    CHECK_THAT(10.0 * std::log10(pFarFirst / pAlone), WithinAbs(0.0, 0.01));
}

TEST_CASE("MarineRadar: a moving shower is folded where it is at the trace's time") {
    MarineRadarModel m;
    m.multipath = false;
    m.minRange = 20.f;
    m.maxRange = 6020.f;
    m.rangeBins = 1000;
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    const Vector3 origin(0, 5, 0);
    const auto o = oneRayAt(syn, origin, Vector3(0, 5, -5900.f));

    auto fold = [&](const RainCell& cell, double time) {
        Weather weather;
        weather.rain = {cell};
        std::vector<std::complex<float>> col(m.rangeBins);
        std::vector<std::int32_t> lab(m.rangeBins);
        std::vector<float> amp(m.rangeBins);
        syn.foldSlot(m, mats, weather, time, 0, o.returns.data(), o.dirs.data(), o.solid.data(), origin,
                     Matrix3{}, 1, col.data(), lab.data(), amp.data(), m.rangeBins);
        return std::make_pair(col, lab);
    };
    RainCell moving;
    moving.centre = Vector3(150.f, 0, -2000.f);
    moving.velocity = Vector3(-6.f, 0, -9.f);
    moving.radius = 300.f;
    moving.rainRate = 30.f;
    const double t = 40.0;
    RainCell placed = moving;
    placed.centre = moving.centreAt(t);
    placed.velocity = Vector3();
    CHECK(placed.centre.x == -90.f);
    CHECK(placed.centre.z == -2360.f);

    const auto atT = fold(moving, t);
    const auto atZero = fold(placed, 0.0);
    unsigned nonZero = 0;
    for (unsigned j = 0; j < m.rangeBins; ++j) {
        REQUIRE(atT.first[j] == atZero.first[j]);
        REQUIRE(atT.second[j] == atZero.second[j]);
        nonZero += atT.first[j] != std::complex<float>(0.f, 0.f);
    }
    CHECK(nonZero > 0u);
    // And it is not where it started: the fold at time 0 differs.
    const auto start = fold(moving, 0.0);
    bool differs = false;
    for (unsigned j = 0; j < m.rangeBins; ++j) differs = differs || start.first[j] != atT.first[j];
    CHECK(differs);
}

TEST_CASE("Rain: volume reflectivity and attenuation at 9.41 GHz against IEC 62388 Table D.5") {
    const double lambda = 299792458.0 / 9.41e9;
    const double eta4 = 10.0 * std::log10(rainVolumeReflectivity(4.0, lambda));
    const double eta16 = 10.0 * std::log10(rainVolumeReflectivity(16.0, lambda));
    INFO("eta: " << eta4 << " dB at 4 mm/h, " << eta16 << " dB at 16 mm/h; attenuation "
                 << rainSpecificAttenuation(4.0) << ", " << rainSpecificAttenuation(16.0) << " dB/km");
    CHECK_THAT(eta4, WithinAbs(-63.0, 0.2));
    CHECK_THAT(eta16, WithinAbs(-53.3, 0.2));
    CHECK_THAT(rainSpecificAttenuation(4.0), WithinAbs(0.059, 0.002));
    CHECK_THAT(rainSpecificAttenuation(16.0), WithinAbs(0.347, 0.005));
}

TEST_CASE("Weather: the significant wave height of the ocean's spectrum") {
    auto hs = [](float U, float fetch, float swell = 0.f) {
        Weather w;
        w.windSpeed = U;
        w.fetch = fetch;
        w.swellHeight = swell;
        return w.significantWaveHeight();
    };
    const double a = hs(4.5f, 1500.f), b = hs(9.6f, 8000.f), c = hs(10.f, 0.f);
    INFO("Hs: " << a << " m (4.5 m/s, 1.5 km), " << b << " m (9.6 m/s, 8 km), " << c << " m (10 m/s, developed)");
    CHECK_THAT(a, WithinRel(0.096, 0.03));
    CHECK_THAT(b, WithinRel(0.48, 0.03));
    CHECK_THAT(c, WithinRel(2.141, 0.03));

    // Monotone in fetch at a fixed wind up to 0.9 of the fully developed
    // fetch 1609 U^2, and fully developed from there on. In between it is
    // not: the peak's share of the variance (1 / A_gamma) falls faster than
    // r^0.5567 rises as gamma goes to 1, so the height peaks about 3.5% over
    // the developed one near r = 0.955 and comes down to it at r = 1.
    double last = 0.0;
    for (const float f : {100.f, 1000.f, 5000.f, 20000.f, 60000.f, 120000.f, 144810.f}) {
        const double h = hs(10.f, f);
        CHECK(h >= last);
        last = h;
    }
    const double developed = hs(10.f, 0.f);
    CHECK_THAT(hs(10.f, 1609.f * 100.f), WithinRel(developed, 1e-9));
    CHECK_THAT(hs(10.f, 400000.f), WithinRel(developed, 1e-9));

    // Swell adds in quadrature.
    CHECK_THAT(hs(9.6f, 8000.f, 1.5f), WithinRel(std::sqrt(b * b + 1.5 * 1.5), 1e-9));
    CHECK_THAT(hs(0.f, 0.f, 1.5f), WithinRel(1.5, 1e-9));
}

TEST_CASE("Douglas sea state: seaStateFromWaveHeight inverts douglasWaveHeight") {
    for (const double s : {0.0, 0.5, 1.0, 2.3, 4.0, 6.0}) {
        INFO("sea state " << s);
        CHECK_THAT(seaStateFromWaveHeight(douglasWaveHeight(s)), WithinAbs(s, 1e-9));
    }
    CHECK(seaStateFromWaveHeight(9.0) == 6.0);
    CHECK(seaStateFromWaveHeight(-1.0) == 0.0);
}

TEST_CASE("MarineRadar: two-ray lobing over a glassy sea") {
    MarineRadarModel m;
    m.rangeBins = 1024;
    Weather calm;
    calm.windSpeed = 0.f;// no waves, no roughness: the full Fresnel reflection
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
        return 10.0 * std::log10(columnPower(a, on, mats, origin, p, nullptr, calm) /
                                 columnPower(b, off, mats, origin, p, nullptr, calm));
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
            syn.foldSlot(m, mats, Weather{}, 0.0, sl, ret.data(), dirs.data(), solid.data(), origin, Matrix3{}, 1,
                         col[c].data(), lab[c].data(), amp[c].data(), m.rangeBins);
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

// Not a test of a result: what one slot costs to fold, with and without a
// shower on its bearing. Hidden; run with "[.bench]".
TEST_CASE("MarineRadar: the cost of folding one slot, dry and through a shower", "[.bench]") {
    MarineRadarModel m = MarineRadarModel::Radome24in();
    m.minRange = 25.f;
    m.maxRange = 1852.f;
    m.rangeBins = 1024;
    m.pulseLength = 80e-9f;
    MarineRadarSynthesizer syn(m);
    RadarMaterials mats;
    mats.set(1, RadarMaterial::Sea());
    const Vector3 origin(0, 4, 0);
    std::vector<Vector3> dirs;
    std::vector<float> solid;
    syn.slotRays(0, origin.y, dirs, solid);
    const unsigned n = syn.raysPerSlot();

    // Every ray that reaches the sea inside the range hits it; the rest miss.
    std::vector<LidarReturn> miss(n), sea(n);
    unsigned seaHits = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (dirs[i].y >= 0.f) continue;
        const float t = -origin.y / dirs[i].y;
        if (t > m.maxRange) continue;
        sea[i] = hitAt(origin, dirs[i], t, 1);
        sea[i].normal = Vector3(0, 1, 0);
        ++seaHits;
    }

    Weather dry, wet;
    dry.windSpeed = wet.windSpeed = 9.6f;
    dry.fetch = wet.fetch = 8000.f;
    RainCell cell;// the USV demo's squall, on the slot's bearing
    cell.centre = Vector3(0, 0, -900.f);
    cell.radius = 150.f;
    cell.top = 1400.f;
    cell.rainRate = 16.f;
    wet.rain = {cell};

    std::vector<std::complex<float>> col(m.rangeBins);
    std::vector<std::int32_t> lab(m.rangeBins);
    std::vector<float> amp(m.rangeBins);
    auto ms = [&](const std::vector<LidarReturn>& returns, const Weather& w) {
        constexpr int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) {
            syn.foldSlot(m, mats, w, 0.0, 0, returns.data(), dirs.data(), solid.data(), origin, Matrix3{}, 1,
                         col.data(), lab.data(), amp.data(), m.rangeBins);
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double a = ms(miss, dry), b = ms(sea, dry), c = ms(miss, wet), d = ms(sea, wet);
    std::printf("rays per slot %u, sea hits %u\n", n, seaHits);
    std::printf("ms per slot: all miss dry %.4f | sea dry %.4f | all miss shower %.4f | sea shower %.4f\n", a, b, c, d);
    std::printf("at 24 rpm, 2048 spokes, 60 fps (13.65 slots a frame): dry %.2f ms, shower %.2f ms a frame\n",
                13.65 * b, 13.65 * d);
    // A wide shower that covers the whole range: the cost must not grow with its size.
    Weather wide = wet;
    wide.rain[0].radius = 1500.f;
    const double e = ms(sea, wide);
    std::printf("ms per slot: sea wide shower (radius 1500 m) %.4f\n", e);
    SUCCEED();
}

namespace {

    // The radar of the USV demo: a 24 in radome on a 1 nmi range scale.
    MarineRadarModel rainRadar() {
        MarineRadarModel m = MarineRadarModel::Radome24in();
        m.minRange = 25.f;
        m.maxRange = 1852.f;
        m.rangeBins = 1024;
        m.pulseLength = 80e-9f;
        return m;
    }

    Weather showerAhead(float radius, float top, float rainRate) {
        Weather w;
        w.windSpeed = 9.6f;
        w.fetch = 8000.f;
        RainCell cell;// on slot 0's bearing
        cell.centre = Vector3(0, 0, -900.f);
        cell.radius = radius;
        cell.top = top;
        cell.rainRate = rainRate;
        w.rain = {cell};
        return w;
    }

    struct RainWindows {
        std::vector<double> energy;// mean over the keys of the sum of |column|^2 in each window
        double rayleigh = 0.0;     // (mean |x|)^2 / mean |x|^2 of the bin at `probeRange`
    };

    enum class RainHits { None, Sea, Wall };

    // Slot 0 folded once per revolution key. Every ray misses (None), or every
    // ray that reaches the sea inside the range hits it (Sea), or every ray
    // hits a metal wall 1600 m out along itself (Wall).
    RainWindows foldRainWindows(const Vector3& origin, const Weather& w, RainHits hits,
                                const std::vector<std::pair<double, double>>& windows, unsigned firstKey,
                                unsigned keys, double probeRange = 900.0) {
        const MarineRadarModel m = rainRadar();
        MarineRadarSynthesizer syn(m);
        RadarMaterials mats;
        mats.set(1, RadarMaterial::Sea());
        mats.set(9, RadarMaterial::Metal());
        std::vector<Vector3> dirs;
        std::vector<float> solid;
        syn.slotRays(0, origin.y, dirs, solid);
        std::vector<LidarReturn> returns(syn.raysPerSlot());
        for (unsigned i = 0; i < returns.size(); ++i) {
            if (hits == RainHits::Wall) {
                returns[i] = hitAt(origin, dirs[i], 1600.f, 9);
            } else if (hits == RainHits::Sea && dirs[i].y < 0.f && -origin.y / dirs[i].y <= m.maxRange) {
                returns[i] = hitAt(origin, dirs[i], -origin.y / dirs[i].y, 1);
                returns[i].normal = Vector3(0, 1, 0);
            }
        }
        std::vector<std::complex<float>> col(m.rangeBins);
        std::vector<std::int32_t> lab(m.rangeBins);
        std::vector<float> amp(m.rangeBins);
        RainWindows out;
        out.energy.assign(windows.size(), 0.0);
        const auto probe = static_cast<std::size_t>((probeRange - m.minRange) / m.binWidth());
        double sumAbs = 0.0, sumSq = 0.0;
        for (unsigned k = 0; k < keys; ++k) {
            syn.foldSlot(m, mats, w, 0.0, 0, returns.data(), dirs.data(), solid.data(), origin, Matrix3{},
                         firstKey + k, col.data(), lab.data(), amp.data(), m.rangeBins);
            for (unsigned j = 0; j < m.rangeBins; ++j) {
                const double R = m.minRange + (j + 0.5) * m.binWidth();
                const double e = std::norm(std::complex<double>(col[j]));
                for (std::size_t q = 0; q < windows.size(); ++q) {
                    if (R >= windows[q].first && R < windows[q].second) out.energy[q] += e;
                }
            }
            sumAbs += std::abs(std::complex<double>(col[probe]));
            sumSq += std::norm(std::complex<double>(col[probe]));
        }
        for (double& e : out.energy) e /= keys;
        if (sumSq > 0.0) out.rayleigh = (sumAbs / keys) * (sumAbs / keys) / (sumSq / keys);
        return out;
    }

}// namespace

// Prints the numbers the rain cases below are pinned to. Hidden; run with "[.rainref]".
TEST_CASE("MarineRadar: reference energies of the rain fold", "[.rainref]") {
    const std::vector<std::pair<double, double>> thirds = {{450, 750}, {750, 1050}, {1050, 1350}};
    const std::vector<std::pair<double, double>> wide = {{200, 733}, {733, 1266}, {1266, 1800}};
    const std::vector<std::pair<double, double>> behind = {{1400, 1800}};
    auto show = [](const char* name, const RainWindows& a) {
        std::printf("%s:", name);
        for (const double e : a.energy) std::printf(" %.4e", e);
        std::printf(" rayleigh %.4f\n", a.rayleigh);
    };
    auto behindDb = [&](const Vector3& origin, const Weather& wet, RainHits hits) {
        Weather dry = wet;
        dry.rain.clear();
        return 10.0 * std::log10(foldRainWindows(origin, wet, hits, behind, 1, 1).energy[0] /
                                 foldRainWindows(origin, dry, hits, behind, 1, 1).energy[0]);
    };
    const Vector3 low(0, 4, 0), high(0, 200, 0);
    show("demo 16", foldRainWindows(low, showerAhead(150.f, 1400.f, 16.f), RainHits::None, thirds, 1, 800));
    show("demo 100", foldRainWindows(low, showerAhead(150.f, 1400.f, 100.f), RainHits::None, thirds, 1, 800));
    show("low top 16", foldRainWindows(low, showerAhead(600.f, 60.f, 16.f), RainHits::None, wide, 1, 800));
    show("from above 100", foldRainWindows(high, showerAhead(600.f, 100.f, 100.f), RainHits::None, wide, 1, 800));
    std::printf("sea behind, 100 mm/h, past the echo (1560 m on): %.4f dB\n",
                10.0 * std::log10(foldRainWindows(low, showerAhead(150.f, 1400.f, 100.f), RainHits::Sea, {{1560, 1800}}, 1, 1).energy[0] /
                                  foldRainWindows(low, showerAhead(150.f, 1400.f, 0.f), RainHits::Sea, {{1560, 1800}}, 1, 1).energy[0]));
    std::printf("wall behind, 100 mm/h: %.4f dB\n", behindDb(low, showerAhead(150.f, 1400.f, 100.f), RainHits::Wall));
    std::printf("wall behind, low top 100 mm/h: %.4f dB\n", behindDb(low, showerAhead(150.f, 60.f, 100.f), RainHits::Wall));
    SUCCEED();
}

// The rain fold's output, pinned to what the march of one echo per ray per
// range step gave (2026-11-15, the numbers "[.rainref]" prints): the mean echo
// energy of a shower in three range windows, over 800 revolutions, and what a
// shower takes from the echoes behind it. A fold that sums the rays' rain per
// range cell draws other noise, so the energies are compared as means (the
// two sets of 800 draws differ by about 2% from sampling alone) and the single
// bin's amplitudes as a Rayleigh distribution, (mean |x|)^2 / mean |x|^2 = pi / 4.
TEST_CASE("MarineRadar: a shower's mean echo, range window by range window") {
    const std::vector<std::pair<double, double>> thirds = {{450, 750}, {750, 1050}, {1050, 1350}};
    const std::vector<std::pair<double, double>> wide = {{200, 733}, {733, 1266}, {1266, 1800}};
    const Vector3 low(0, 4, 0), high(0, 200, 0);
    struct Case {
        const char* name;
        Vector3 origin;
        Weather weather;
        std::vector<std::pair<double, double>> windows;
        std::array<double, 3> energy;
    };
    const std::vector<Case> cases = {
            // The USV demo's squall: the near flank, the core, the far flank.
            {"16 mm/h", low, showerAhead(150.f, 1400.f, 16.f), thirds, {1.7472e-11, 8.7174e-11, 8.7819e-12}},
            // Heavy: the far flank is also dimmed by the rain before it.
            {"100 mm/h", low, showerAhead(150.f, 1400.f, 100.f), thirds, {3.1867e-10, 1.3174e-09, 1.0426e-10}},
            // A wide shower with its top at 60 m: the rays that climb leave it.
            {"top at 60 m", low, showerAhead(600.f, 60.f, 16.f), wide, {3.8560e-10, 4.3033e-11, 5.8193e-12}},
            // An antenna 200 m up over rain that tops out at 100 m: the rays enter it from above.
            {"from above", high, showerAhead(600.f, 100.f, 100.f), wide, {8.5335e-10, 7.8497e-10, 1.6656e-10}},
    };
    for (const Case& c : cases) {
        const auto got = foldRainWindows(c.origin, c.weather, RainHits::None, c.windows, 1, 800);
        for (std::size_t q = 0; q < 3; ++q) {
            INFO(c.name << ", window " << q << ": " << got.energy[q] << " against " << c.energy[q]);
            CHECK_THAT(got.energy[q], WithinRel(c.energy[q], 0.07));
        }
        INFO(c.name << ": Rayleigh ratio " << got.rayleigh);
        CHECK_THAT(got.rayleigh, WithinAbs(0.7854, 0.04));
    }
}

TEST_CASE("MarineRadar: what a shower takes from the echoes behind it") {
    const std::vector<std::pair<double, double>> behind = {{1400, 1800}};
    // Past the reach of the shower's own echo: the rays that climb carry it
    // out to 1350 m / cos(elevation), about 1516 m, plus the pulse's 12 m.
    const std::vector<std::pair<double, double>> clear = {{1560, 1800}};
    const Vector3 low(0, 4, 0);
    auto behindDb = [&](const Weather& wet, RainHits hits, const std::vector<std::pair<double, double>>& window) {
        Weather dry = wet;
        dry.rain.clear();
        return 10.0 * std::log10(foldRainWindows(low, wet, hits, window, 1, 1).energy[0] /
                                 foldRainWindows(low, dry, hits, window, 1, 1).energy[0]);
    };
    // The sea past the shower: every ray that reaches it crossed the whole
    // shower, so the loss is the shower's own, two ways. Its attenuation
    // 0.0101 R^1.276 dB/km is a Gaussian across of sigma 150 / sqrt(1.276) m:
    // 3.600 dB/km at the centre times 0.1328 sqrt(2 pi) km = 1.198 dB one way.
    // (A window that still held the tail of the shower's echo, a fiftieth of
    // the sea's energy, moved with the draw by tenths of a dB: the first
    // version of this case pinned one draw.)
    CHECK_THAT(behindDb(showerAhead(150.f, 1400.f, 100.f), RainHits::Sea, clear), WithinAbs(-2.396, 0.05));
    // A wall 1600 m along every ray: the rays aimed below the horizon leave
    // the rain at the sea before the shower and lose nothing.
    CHECK_THAT(behindDb(showerAhead(150.f, 1400.f, 100.f), RainHits::Wall, behind), WithinAbs(-1.5999, 0.05));
    // The same with the shower's top at 60 m: the rays that climb leave it too.
    CHECK_THAT(behindDb(showerAhead(150.f, 60.f, 100.f), RainHits::Wall, behind), WithinAbs(-0.0261, 0.01));
}
