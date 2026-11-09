#include "threepp/helpers/MarineRadarModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace threepp;
using namespace threepp::echo;

namespace {

    constexpr double kDeg = kPi / 180.0;
    constexpr double kBoltzmann = 1.380649e-23;
    constexpr double kT0 = 290.0;
    constexpr double kC = 299792458.0;
    // Sea water at X band and ~15 deg C, eps = eps' - j eps''.
    const std::complex<double> kSeaPermittivity{55.0, -37.0};
    constexpr unsigned kTiltBands = 16;

    std::complex<double> phasor(double amp, double phase) {
        return {amp * std::cos(phase), amp * std::sin(phase)};
    }

    // Fresnel reflection of sea water at grazing angle psi.
    std::complex<double> seaFresnel(double psi, RadarPolarization pol) {
        const double s = std::sin(psi), c = std::cos(psi);
        const std::complex<double> root = std::sqrt(kSeaPermittivity - c * c);
        if (pol == RadarPolarization::HH) return (s - root) / (s + root);
        return (kSeaPermittivity * s - root) / (kSeaPermittivity * s + root);
    }

    double wrapDeg(double d) {
        d = std::fmod(d, 360.0);
        return d < 0.0 ? d + 360.0 : d;
    }

    std::int64_t floorMod(std::int64_t a, std::int64_t n) {
        const std::int64_t m = a % n;
        return m < 0 ? m + n : m;
    }

}// namespace

// ── Sea clutter ────────────────────────────────────────────────────────────

double threepp::nrlSeaClutterDb(double grazingDeg, double frequencyGHz, double seaState,
                                RadarPolarization pol) {
    // Gregers-Hansen & Mital, "An Improved Empirical Model for Radar Sea
    // Clutter Reflectivity", NRL/MR/5310-12-9346 (2012):
    //   sigma0 = c1 + c2 log10(sin a) + (27.5 + c3 a) log10(f) / (1 + 0.95 a)
    //          + c4 (1 + SS)^(1 / (2 + 0.085 a + 0.033 SS)) + c5 a^2
    // a in degrees, f in GHz. HH reproduces the published worked value
    // (a = 10, f = 30, SS 2 -> -36.66 dB); VV is as recalled, unverified.
    const double a = std::clamp(grazingDeg, 0.1, 60.0);
    const double f = std::clamp(frequencyGHz, 0.5, 35.0);
    const double ss = std::clamp(seaState, 0.0, 6.0);
    double c1, c2, c3, c4, c5;
    if (pol == RadarPolarization::HH) {
        c1 = -73.00, c2 = 20.78, c3 = 7.351, c4 = 25.65, c5 = 0.00540;
    } else {
        c1 = -50.79, c2 = 25.93, c3 = 0.7093, c4 = 21.58, c5 = 0.00211;
    }
    return c1 + c2 * std::log10(std::sin(a * kDeg)) + (27.5 + c3 * a) * std::log10(f) / (1.0 + 0.95 * a) +
           c4 * std::pow(1.0 + ss, 1.0 / (2.0 + 0.085 * a + 0.033 * ss)) + c5 * a * a;
}

double threepp::douglasWaveHeight(double seaState) {
    // Mid-band significant wave heights of the Douglas sea scale.
    static const double hs[] = {0.0, 0.05, 0.3, 0.875, 1.875, 3.25, 5.0};
    const double s = std::clamp(seaState, 0.0, 6.0);
    const auto i = static_cast<unsigned>(std::floor(s));
    if (i >= 6) return hs[6];
    const double f = s - i;
    return hs[i] + f * (hs[i + 1] - hs[i]);
}

double RadarMaterial::sigma0(double cosTheta) const {
    const double c = std::clamp(cosTheta, 0.0, 1.0);
    const double c2 = c * c;
    double s = dbToPower(lambertDb) * c2;
    if (rmsSlope > 0.f && c > 1e-3) {
        const double s2 = static_cast<double>(rmsSlope) * rmsSlope;
        s += dbToPower(reflectionLossDb) * std::exp(-(1.0 - c2) / c2 / s2) / (s2 * c2 * c2);
    }
    if (isotropicDb > -300.f) s += dbToPower(isotropicDb);
    return s;
}

double MarineRadarModel::antennaGain() const {
    const double th = std::max(0.05f, horizontalBeamwidth) * kDeg;
    const double tv = std::max(0.5f, verticalBeamwidth) * kDeg;
    return std::max(0.05f, antennaEfficiency) * 4.0 * kPi / (th * tv);
}

MarineRadarModel MarineRadarModel::OpenArray4ft() { return MarineRadarModel{}; }

MarineRadarModel MarineRadarModel::OpenArray6ft() {
    MarineRadarModel m;
    m.horizontalBeamwidth = 1.2f;
    m.peakPowerW = 12000.f;
    return m;
}

MarineRadarModel MarineRadarModel::Radome24in() {
    MarineRadarModel m;
    m.horizontalBeamwidth = 3.9f;
    m.verticalBeamwidth = 25.f;
    m.peakPowerW = 4000.f;
    return m;
}

// ── Polar image ────────────────────────────────────────────────────────────

void RadarPolarImage::reset(unsigned nSpokes, unsigned nBins, float r0, float r1) {
    spokes = nSpokes;
    bins = nBins;
    minRange = r0;
    maxRange = r1;
    const std::size_t n = static_cast<std::size_t>(spokes) * bins;
    display.assign(n, 0.f);
    snrDb.assign(n, 0.f);
    label.assign(n, -1);
    time.assign(spokes, -1.0);
    worldBearingDeg.assign(spokes, 0.f);
    byWorldBearing.assign(spokes, -1);
}

void RadarPolarImage::store(const RadarSpoke& s) {
    if (spokes == 0) return;
    const auto idx = static_cast<std::size_t>(floorMod(s.index, spokes));
    const std::size_t base = idx * bins;
    const std::size_t n = std::min<std::size_t>(bins, s.display.size());
    std::copy_n(s.display.begin(), n, display.begin() + static_cast<std::ptrdiff_t>(base));
    std::copy_n(s.snrDb.begin(), std::min<std::size_t>(bins, s.snrDb.size()), snrDb.begin() + static_cast<std::ptrdiff_t>(base));
    std::copy_n(s.label.begin(), std::min<std::size_t>(bins, s.label.size()), label.begin() + static_cast<std::ptrdiff_t>(base));
    time[idx] = s.time;
    worldBearingDeg[idx] = s.worldBearingDeg;
    const auto w = static_cast<std::size_t>(
            floorMod(static_cast<std::int64_t>(std::floor(wrapDeg(s.worldBearingDeg) / 360.0 * spokes)), spokes));
    byWorldBearing[w] = static_cast<std::int32_t>(idx);
}

// ── Synthesizer ────────────────────────────────────────────────────────────

MarineRadarSynthesizer::MarineRadarSynthesizer(const MarineRadarModel& model) {
    spokes_ = std::max(8u, model.spokes);
    azimuthSamples_ = std::max(1u, model.azimuthSamples);
    elevationSamples_ = std::max(4u, model.elevationSamples);
    horizonSamples_ = std::min(elevationSamples_ - 2u,
                               static_cast<unsigned>(std::lround(std::clamp(model.horizonShare, 0.f, 1.f) *
                                                                 static_cast<float>(elevationSamples_))));
    aboveSamples_ = std::min(elevationSamples_ - 2u - horizonSamples_,
                             static_cast<unsigned>(std::lround(std::clamp(model.aboveHorizonShare, 0.f, 1.f) *
                                                               static_cast<float>(elevationSamples_))));

    const double hbw = std::clamp(static_cast<double>(model.horizontalBeamwidth), 0.1, 45.0);
    const double spanU = std::max(1.0, static_cast<double>(model.patternSpanBeamwidths)) * 2.0 *
                         std::sin(0.5 * hbw * kDeg);
    hPattern_ = ArrayPattern(hbw, model.horizontalTaper, std::min(spanU, 0.99));
    vPattern_ = ArrayPattern(std::clamp(static_cast<double>(model.verticalBeamwidth), 1.0, 120.0), Taper::Uniform, 1.0);
    elevationSpan_ = std::asin(std::min(0.999, 2.26 * vPattern_.halfPowerU()));

    const double slotAngle = 2.0 * kPi / spokes_;
    // One antenna transmits and receives: the two-way voltage is the one-way
    // pattern squared. Its tail below -70 dB two-way is dropped from the kernel.
    auto twoWay = [&](int k) {
        const double v = hPattern_(std::sin(k * slotAngle));
        return v * v;
    };
    int K = static_cast<int>(std::ceil(std::asin(hPattern_.span()) / slotAngle));
    while (K > 1 && std::abs(twoWay(K)) < 3e-4 && std::abs(twoWay(K - 1)) < 3e-4) --K;
    kernelHalf_ = static_cast<unsigned>(K);
    kernel_.resize(2 * kernelHalf_ + 1);
    for (int k = -K; k <= K; ++k) kernel_[static_cast<std::size_t>(k + K)] = static_cast<float>(twoWay(k));
    mainLobeHalf_ = static_cast<unsigned>(std::ceil(std::asin(hPattern_.halfPowerU()) / slotAngle));
    tiltMean_.assign(kTiltBands, 1.0);
    minRange_ = std::max(1.f, model.minRange);
    maxRange_ = std::max(model.maxRange, minRange_ + 1.f);
    seed_ = model.seed;
}

void MarineRadarSynthesizer::slotRays(std::int64_t slot, double antennaHeight, std::vector<Vector3>& dirs,
                                      std::vector<float>& solidAngle) const {
    // Hard-coded geometry only: the ray table of a slot depends on the slot,
    // the antenna height and the seed baked at construction.
    const double slotAngle = 2.0 * kPi / spokes_;
    const auto s = static_cast<std::uint64_t>(floorMod(slot, spokes_));
    const double theta0 = static_cast<double>(s) * slotAngle;
    const double h = std::max(0.2, antennaHeight);
    const unsigned nA = aboveSamples_;
    const unsigned nU = elevationSamples_ - horizonSamples_ - nA;
    const unsigned nH = horizonSamples_;
    const double N = static_cast<double>(elevationSamples_);
    const double span = elevationSpan_;
    // The horizon samples map uniformly onto the sea plane between these ranges.
    const double rLo = std::max(1.0, static_cast<double>(minRange_));
    const double rHi = std::max(rLo + 1.0, static_cast<double>(maxRange_));
    const double eHi = -std::atan(h / rHi), eLo = -std::atan(h / rLo);
    // The band above: the sea band's mirror, aimed out to twice the range.
    const double aHi = 2.0 * rHi;
    const double uLo = std::atan(h / aHi), uHi = std::atan(h / rLo);

    auto pdf = [&](double el) {
        double p = 0.0;
        if (nU && el >= -span && el <= span) p += (nU / N) / (2.0 * span);
        if (nH && el >= eLo && el <= eHi) {
            const double se = std::sin(-el);
            p += (nH / N) * h / ((rHi - rLo) * se * se);
        }
        if (nA && el >= uLo && el <= uHi) {
            const double se = std::sin(el);
            p += (nA / N) * h / ((aHi - rLo) * se * se);
        }
        return p;
    };

    for (unsigned a = 0; a < azimuthSamples_; ++a) {
        for (unsigned e = 0; e < elevationSamples_; ++e) {
            const std::uint64_t key = combineKeys(combineKeys(combineKeys(seed_, s), a), e);
            const double theta = theta0 + (stratum(a, azimuthSamples_, key, true) - 0.5) * slotAngle;
            double el;
            if (e < nU) {
                el = -span + 2.0 * span * stratum(e, nU, key ^ 0x51ULL, true);
            } else if (e < nU + nH) {
                const double R = rLo + (rHi - rLo) * stratum(e - nU, nH, key ^ 0x52ULL, true);
                el = -std::atan(h / R);
            } else {
                const double R = rLo + (aHi - rLo) * stratum(e - nU - nH, nA, key ^ 0x53ULL, true);
                el = std::atan(h / R);
            }
            const double ce = std::cos(el);
            dirs.emplace_back(static_cast<float>(ce * std::sin(theta)), static_cast<float>(std::sin(el)),
                              static_cast<float>(-ce * std::cos(theta)));
            const double p = pdf(el);
            solidAngle.push_back(p > 0.0 ? static_cast<float>(slotAngle / azimuthSamples_ * ce / (N * p)) : 0.f);
        }
    }
}

unsigned MarineRadarSynthesizer::foldSlot(const MarineRadarModel& model, const RadarMaterials& materials,
                                      std::int64_t slot, const LidarReturn* returns, const Vector3* dirs,
                                      const float* solidAngle, const Vector3& origin,
                                      const Matrix3& worldFromLevel, std::uint64_t revolutionKey,
                                      std::complex<float>* column, std::int32_t* label, float* labelAmp,
                                      unsigned columnBins, const Matrix3* antennaFromLevel) {
    // The bins are laid out by the model; only those the rows hold are written.
    const unsigned modelBins = std::max(1u, model.rangeBins);
    const unsigned bins = std::min(modelBins, columnBins);
    if (bins == 0) return 0;
    std::fill(column, column + bins, std::complex<float>(0.f, 0.f));
    std::fill(label, label + bins, -1);
    std::fill(labelAmp, labelAmp + bins, 0.f);

    const double r0 = std::max(1.f, model.minRange);
    const double r1 = std::max(model.maxRange, model.minRange + 1.f);
    const double binW = (r1 - r0) / modelBins;
    const double lambda = model.wavelength();
    const double k = 2.0 * kPi / lambda;
    const double G = model.antennaGain();
    const double C = model.peakPowerW * G * G * lambda * lambda /
                     (std::pow(4.0 * kPi, 3.0) * dbToPower(model.systemLossDb));
    const RangeResponse resp(0.5 * kC * model.pulseLength, PulseKind::CW);
    const double hs = resp.halfSpan();
    const double h = origin.y - model.seaLevel;
    const double sigmaH = 0.25 * douglasWaveHeight(model.seaState);
    const double fGHz = model.frequency * 1e-9;
    const double sinMin = std::sin(0.1 * kDeg);
    const auto sIdx = static_cast<std::uint64_t>(floorMod(slot, spokes_));

    unsigned extent = 0;
    auto deposit = [&](double range, std::complex<double> a, std::int32_t id) {
        if (range + hs < r0 || range - hs > r1) return;
        const double x = (range - r0) / binW;
        const int jlo = std::max(0, static_cast<int>(std::floor(x - hs / binW)));
        const int jhi = std::min(static_cast<int>(bins) - 1, static_cast<int>(std::floor(x + hs / binW)));
        if (jhi >= jlo) extent = std::max(extent, static_cast<unsigned>(jhi + 1));
        for (int j = jlo; j <= jhi; ++j) {
            double w = 0.0;
            for (int t = 0; t < 4; ++t) w += resp(r0 + (j + (t + 0.5) * 0.25) * binW - range);
            column[j] += std::complex<float>(a * (0.25 * w));
        }
        const int jc = std::clamp(static_cast<int>(std::floor(x)), 0, static_cast<int>(bins) - 1);
        const float amp = static_cast<float>(std::abs(a));
        if (amp > labelAmp[jc]) {
            labelAmp[jc] = amp;
            label[jc] = id;
        }
    };

    auto band = [&](double R) {
        const double t = std::log(std::max(R, r0) / r0) / std::log(r1 / r0);
        return std::min(kTiltBands - 1, static_cast<unsigned>(std::max(0.0, t) * kTiltBands));
    };
    std::array<double, kTiltBands> tiltW{}, tiltSum{};

    // One ray's rain, every shower's steps together, walked in range order:
    // an echo is dimmed by the rain nearer than it, whichever shower that
    // rain belongs to and wherever that shower sits in the list (showers
    // overlap, so sorting the showers would not do).
    struct RainStep {
        double t, dt, rate;
        std::uint32_t cell, step;
    };
    std::vector<RainStep> rainSteps;

    const unsigned n = raysPerSlot();
    for (unsigned i = 0; i < n; ++i) {
        const double dOmega = solidAngle[i];
        if (!(dOmega > 0.0)) continue;
        Vector3 dW = dirs[i];
        dW.applyMatrix3(worldFromLevel).normalize();
        // The vertical pattern tilts with the antenna; the rays do not.
        Vector3 dA = dirs[i];
        if (antennaFromLevel) dA.applyMatrix3(*antennaFromLevel);
        const double gv = vPattern_(dA.y);
        const double gv2 = gv * gv;

        const LidarReturn& r = returns[i];
        const bool hit = r.returnNo > 0 && r.returnKind == 0 && r.hitInstanceId != -2;
        double R = 0.0;
        if (hit) {
            Vector3 d = r.position;
            d.sub(origin);
            R = d.length();
        }

        // ── Rain on the way: volume echoes, and the attenuation they cost ──
        double rainAttDb = 0.0;// one-way, accumulated to the hit
        const double tEnd = hit ? std::min(R, r1) : r1;
        rainSteps.clear();
        for (std::size_t c = 0; c < model.rain.size(); ++c) {
            const RainCell& cell = model.rain[c];
            if (cell.rainRate <= 0.f || cell.radius <= 0.f) continue;
            // Horizontal approach to the shower's axis: |o + t d - c|^2 in xz.
            const double ox = origin.x - cell.centre.x, oz = origin.z - cell.centre.z;
            const double dx = dW.x, dz = dW.z;
            const double qa = dx * dx + dz * dz;
            const double rr = 3.0 * cell.radius;
            double tA = r0, tB = tEnd;
            if (qa > 1e-12) {
                const double qb = 2.0 * (ox * dx + oz * dz);
                const double qc = ox * ox + oz * oz - rr * rr;
                const double disc = qb * qb - 4.0 * qa * qc;
                if (disc <= 0.0) continue;
                const double sq = std::sqrt(disc);
                tA = std::max(tA, (-qb - sq) / (2.0 * qa));
                tB = std::min(tB, (-qb + sq) / (2.0 * qa));
            } else if (ox * ox + oz * oz > rr * rr) {
                continue;
            }
            // Between the sea and the shower's top.
            if (std::abs(dW.y) > 1e-9) {
                const double tSea = (model.seaLevel - origin.y) / dW.y;
                const double tTop = (cell.top - origin.y) / dW.y;
                tA = std::max(tA, std::min(tSea, tTop));
                tB = std::min(tB, std::max(tSea, tTop));
            } else if (origin.y < model.seaLevel || origin.y > cell.top) {
                continue;
            }
            if (tB <= tA) continue;
            const double step = std::max(binW, 0.5 * resp.resolution());
            const int steps = std::min(256, static_cast<int>(std::ceil((tB - tA) / step)));
            const double dt = (tB - tA) / steps;
            for (int st = 0; st < steps; ++st) {
                const double t = tA + (st + 0.5) * dt;
                const double px = ox + t * dx, pz = oz + t * dz;
                const double rate = cell.rainRate * std::exp(-(px * px + pz * pz) / (2.0 * cell.radius * cell.radius));
                if (rate < 0.05) continue;
                rainSteps.push_back({t, dt, rate, static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(st)});
            }
        }
        std::sort(rainSteps.begin(), rainSteps.end(), [](const RainStep& a, const RainStep& b) {
            if (a.t != b.t) return a.t < b.t;
            return a.cell != b.cell ? a.cell < b.cell : a.step < b.step;
        });
        for (const RainStep& rs : rainSteps) {
            // Marshall-Palmer reflectivity, |K|^2 = 0.93 for water.
            const double Z = 200.0 * std::pow(rs.rate, 1.6) * 1e-18;// m^6/m^3
            const double eta = std::pow(kPi, 5.0) * 0.93 * Z / std::pow(lambda, 4.0);
            const double dV = rs.t * rs.t * dOmega * rs.dt;
            const double p = C * eta * dV / std::pow(rs.t, 4.0) * std::pow(10.0, -0.2 * rainAttDb);
            const std::uint64_t key = combineKeys(combineKeys(combineKeys(revolutionKey, sIdx), i),
                                                  static_cast<std::uint64_t>(rs.step) * 31 + rs.cell);
            deposit(rs.t, std::sqrt(p) * gv2 * complexGaussian(key), -3);
            // Specific attenuation near 9.4 GHz (ITU-R P.838 class values).
            rainAttDb += 0.0101 * std::pow(rs.rate, 1.276) * rs.dt * 1e-3;
        }

        if (!hit || R < r0 || R > r1 + hs || R < 1e-3) continue;

        const RadarMaterial& mat = materials.of(r.hitInstanceId);
        double cosT = 1.0;
        if (r.normal.lengthSq() > 0.f) cosT = std::abs(static_cast<double>(r.normal.dot(dW)));

        double sigma;
        std::complex<double> F2(1.0, 0.0);
        if (mat.sea) {
            // NRL at the mean grazing angle; the rendered wave facet's own tilt
            // redistributes it (normalised so the band's mean stays on NRL).
            const double sinM = std::clamp(h / R, sinMin, 1.0);
            const double s0 = dbToPower(nrlSeaClutterDb(std::asin(sinM) / kDeg, fGHz, model.seaState,
                                                        model.polarization));
            const double m = std::clamp(cosT / sinM, 0.0, 30.0);
            const unsigned b = band(R);
            tiltW[b] += dOmega;
            tiltSum[b] += dOmega * m;
            sigma = s0 * (m / std::max(tiltMean_[b], 1e-3)) * R * R * dOmega / sinM;
        } else {
            sigma = mat.sigma0(cosT) * R * R * dOmega / std::max(cosT, 0.03);
            if (model.multipath && h > 0.0) {
                // Two-ray: the direct path and the one reflected off the sea.
                const double ht = std::max(0.02, static_cast<double>(r.position.y) - model.seaLevel);
                const double gx = r.position.x - origin.x, gz = r.position.z - origin.z;
                const double Rg = std::max(1.0, std::sqrt(gx * gx + gz * gz));
                const double Rd = std::sqrt(Rg * Rg + (ht - h) * (ht - h));
                const double Rr = std::sqrt(Rg * Rg + (ht + h) * (ht + h));
                const double psi = std::atan((h + ht) / Rg);
                const double kss = k * sigmaH * std::sin(psi);
                const double rho = std::exp(-2.0 * kss * kss);
                // The reflected leg arrives from below the horizon at psi.
                Vector3 dR(dirs[i].x, 0.f, dirs[i].z);
                const float hl = dR.length();
                if (hl > 1e-6f) dR.multiplyScalar(static_cast<float>(std::cos(psi)) / hl);
                dR.y = static_cast<float>(-std::sin(psi));
                if (antennaFromLevel) dR.applyMatrix3(*antennaFromLevel);
                const double gvr = vPattern_(dR.y);
                const double ratio = std::abs(gv) > 1e-6 ? gvr / gv : 0.0;
                const std::complex<double> F =
                        1.0 + seaFresnel(psi, model.polarization) * rho * ratio * phasor(1.0, -k * (Rr - Rd));
                F2 = F * F;
            }
        }
        const double p = C * sigma / std::pow(R, 4.0) * std::pow(10.0, -0.2 * rainAttDb);
        if (!(p > 0.0)) continue;
        // Each ray its own scatterer (see ImagingSonarSynthesizer::synthesize).
        const double phase = 2.0 * k * R + scattererPhase(r.position, lambda, combineKeys(combineKeys(seed_, sIdx), i));
        deposit(R, phasor(std::sqrt(p) * gv2, phase) * F2, r.hitInstanceId);
    }

    for (unsigned b = 0; b < kTiltBands; ++b) {
        if (tiltW[b] > 0.0) tiltMean_[b] = 0.95 * tiltMean_[b] + 0.05 * (tiltSum[b] / tiltW[b]);
    }
    return extent;
}

void MarineRadarSynthesizer::finishSpoke(const MarineRadarModel& model, std::vector<std::complex<double>>& sum,
                                         std::uint64_t revolutionKey, RadarSpoke& out) const {
    const unsigned bins = static_cast<unsigned>(sum.size());
    const double noise = kBoltzmann * kT0 * (1.0 / std::max(1e-9f, model.pulseLength)) *
                         dbToPower(model.noiseFigureDb);
    const double sqN = std::sqrt(noise);
    const double r0 = std::max(1.f, model.minRange);
    const double r1 = std::max(model.maxRange, model.minRange + 1.f);
    const double binW = (r1 - r0) / bins;
    out.snrDb.resize(bins);
    out.display.resize(bins);
    const auto sIdx = static_cast<std::uint64_t>(out.index);
    for (unsigned j = 0; j < bins; ++j) {
        const std::complex<double> z = sum[j] + sqN * complexGaussian(combineKeys(combineKeys(revolutionKey, sIdx), j));
        out.snrDb[j] = static_cast<float>(powerToDb(std::norm(z) / noise, -60.0));
    }

    // The receiver's controls: STC eases the near range, FTC differentiates
    // along range (a log-video high-pass that leaves edges, drops rain).
    const double stcSpan = std::log10(std::max(model.stcRange, model.minRange + 1.f) / r0);
    const int ftcWin = std::max(3, static_cast<int>(std::lround(2.0 * 0.5 * kC * model.pulseLength / binW)));
    double run = 0.0;
    std::vector<float> x(bins);
    for (unsigned j = 0; j < bins; ++j) {
        const double R = r0 + (j + 0.5) * binW;
        const double stc = model.seaClutterDb * std::clamp(1.0 - std::log10(R / r0) / stcSpan, 0.0, 1.0);
        x[j] = static_cast<float>(out.snrDb[j] - stc);
    }
    const double black = model.thresholdDb - model.gainDb;
    const double invDyn = 1.0 / std::max(1.f, model.dynamicRangeDb);
    for (unsigned j = 0; j < bins; ++j) {
        run += x[j];
        if (static_cast<int>(j) >= ftcWin) run -= x[j - ftcWin];
        const double mean = run / std::min<int>(static_cast<int>(j) + 1, ftcWin);
        const double v = x[j] - model.rainClutter * std::max(0.0, mean);
        out.display[j] = static_cast<float>(std::clamp((v - black) * invDyn, 0.0, 1.0));
    }
}

// ── PPI ────────────────────────────────────────────────────────────────────

RadarPpiDisplay::RadarPpiDisplay(unsigned size): size_(std::max(64u, size)) {
    const std::size_t n = static_cast<std::size_t>(size_) * size_;
    rgba_.assign(n * 4, 0);
    trail_.assign(n, 0.f);
    pixRadius_.resize(n);
    pixBearing_.resize(n);
    const double c = 0.5 * (size_ - 1);
    for (unsigned y = 0; y < size_; ++y) {
        for (unsigned x = 0; x < size_; ++x) {
            const double dx = x - c, dy = c - y;
            const std::size_t i = static_cast<std::size_t>(y) * size_ + x;
            pixRadius_[i] = static_cast<float>(std::sqrt(dx * dx + dy * dy));
            pixBearing_[i] = static_cast<float>(wrapDeg(std::atan2(dx, dy) / kDeg));
        }
    }
}

void RadarPpiDisplay::draw(const RadarPolarImage& img, double now, float headingDeg, float sweepDeg, bool northUp,
                           float rangeShown) {
    const unsigned S = size_;
    const std::size_t npx = static_cast<std::size_t>(S) * S;
    const double radius = 0.5 * S - 3.0;
    const double pxPerM = radius / std::max(1.f, rangeShown);
    const double dt = lastDraw_ < 0.0 ? 0.0 : std::max(0.0, now - lastDraw_);
    lastDraw_ = now;
    const float trailDecay = style.trailSeconds > 0.f ? static_cast<float>(std::exp(-dt / style.trailSeconds)) : 0.f;
    const double sweepScreen = wrapDeg(northUp ? sweepDeg + headingDeg : sweepDeg);
    const double headScreen = northUp ? wrapDeg(headingDeg) : 0.0;
    const unsigned spokes = std::max(1u, img.spokes);
    const double spokeDeg = 360.0 / spokes;
    // Trails live in screen space: a new frame of reference (north up / head
    // up, another range) makes the old ones meaningless.
    if (northUp != keyNorthUp_ || rangeShown != keyRange_) std::fill(trail_.begin(), trail_.end(), 0.f);
    keyNorthUp_ = northUp;

    // What depends only on the scope's range and the image's layout: each
    // pixel's range bin and its static overlay (rings, the scope's edge).
    if (rangeShown != keyRange_ || img.bins != keyBins_ || img.minRange != keyMin_ || img.maxRange != keyMax_ ||
        img.spokes != keySpokes_ || style.ringStep != keyRing_) {
        keyRange_ = rangeShown, keyBins_ = img.bins, keyMin_ = img.minRange, keyMax_ = img.maxRange;
        keySpokes_ = img.spokes, keyRing_ = style.ringStep;
        pixBin_.assign(npx, -1);
        pixOverlay_.assign(npx, 0);
        pixSlot_.resize(npx);
        const double ring = style.ringStep > 0.f ? style.ringStep : rangeShown / 4.0;
        const double ringPx = ring * pxPerM;
        for (std::size_t i = 0; i < npx; ++i) {
            const double rp = pixRadius_[i];
            pixSlot_[i] = static_cast<std::int32_t>(static_cast<unsigned>(pixBearing_[i] / spokeDeg) % spokes);
            if (rp > radius + 2.0) {
                pixOverlay_[i] = 3;
                continue;
            }
            const double range = rp / pxPerM;
            if (img.bins && range >= img.minRange && range < img.maxRange) {
                const auto bin = static_cast<int>((range - img.minRange) / (img.maxRange - img.minRange) * img.bins);
                pixBin_[i] = std::min(bin, static_cast<int>(img.bins) - 1);
            }
            if (rp > radius - 0.5) pixOverlay_[i] = 2;
            else if (rp > 4.0 && std::abs(rp - ringPx * std::round(rp / ringPx)) < 0.55) pixOverlay_[i] = 1;
        }
        // Navy -> amber -> warm white, in sqrt of the display value.
        for (int k = 0; k < 256; ++k) {
            const double t = k / 255.0, tt = std::sqrt(t);
            double r = 6 + (255 - 6) * tt, g = 20 + (186 - 20) * tt, b = 38 + (52 - 38) * tt;
            if (t > 0.75) {
                const double w = 0.6 * (t - 0.75) / 0.25;
                r += (255 - r) * w, g += (244 - g) * w, b += (200 - b) * w;
            }
            lut_[k] = {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b)};
        }
    }

    glow_.resize(spokes);
    for (unsigned s = 0; s < img.spokes; ++s) {
        const double age = img.time[s] >= 0.0 ? std::max(0.0, now - img.time[s]) : 1e9;
        glow_[s] = img.time[s] >= 0.0
                           ? static_cast<float>(0.62 + 0.38 * std::exp(-age / std::max(0.05f, style.afterglowSeconds)))
                           : 0.f;
    }
    // North up: which spoke paints each world-bearing slot, if it is current.
    if (northUp) {
        worldSpoke_.assign(spokes, -1);
        for (unsigned w = 0; w < img.spokes; ++w) {
            const std::int32_t idx = img.byWorldBearing[w];
            if (idx < 0) continue;
            const double d = std::abs(wrapDeg(img.worldBearingDeg[static_cast<std::size_t>(idx)] - (w + 0.5) * spokeDeg + 180.0) - 180.0);
            if (d <= 2.5 * spokeDeg) worldSpoke_[w] = idx;
        }
    }

    auto angDiff = [](double a, double b) {
        double d = a - b;
        if (d > 180.0) d -= 360.0;
        if (d < -180.0) d += 360.0;
        return std::abs(d);
    };

    for (std::size_t pi = 0; pi < npx; ++pi) {
        std::uint8_t* px = &rgba_[pi * 4];
        const std::uint8_t ov = pixOverlay_[pi];
        if (ov == 3) {
            px[0] = 14, px[1] = 17, px[2] = 22, px[3] = 255;
            continue;
        }
        float v = 0.f;
        std::int32_t lab = -1;
        const std::int32_t bin = pixBin_[pi];
        if (bin >= 0 && img.spokes) {
            const std::int32_t idx = northUp ? worldSpoke_[static_cast<std::size_t>(pixSlot_[pi])] : pixSlot_[pi];
            if (idx >= 0) {
                const std::size_t ci = static_cast<std::size_t>(idx) * img.bins + static_cast<std::size_t>(bin);
                v = img.display[ci] * glow_[static_cast<std::size_t>(idx)];
                lab = img.label[ci];
            }
        }
        float& tr = trail_[pi];
        tr = std::max(tr * trailDecay, v * 0.8f);

        double r, g, b;
        if (style.showLabels && lab != -1 && v > 0.f) {
            const std::uint64_t hsh = mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(lab)) + 77);
            const double tt = std::sqrt(std::min(1.0, static_cast<double>(v)));
            r = 6 + (90 + (hsh & 0xA5) - 6) * tt, g = 20 + (90 + ((hsh >> 8) & 0xA5) - 20) * tt,
            b = 38 + (90 + ((hsh >> 16) & 0xA5) - 38) * tt;
        } else {
            const auto& c = lut_[static_cast<std::size_t>(std::clamp(static_cast<int>(v * 255.f), 0, 255))];
            r = c[0], g = c[1], b = c[2];
        }
        if (style.trailSeconds > 0.f && tr > v) {
            const double t = std::min(1.0, tr * 1.4) * (1.0 - std::min(1.0, static_cast<double>(v) * 3.0));
            r += (30 - r) * t, g += (120 - g) * t, b += (150 - b) * t;
        }

        const double rp = pixRadius_[pi];
        if (ov == 2) {
            r = 110, g = 130, b = 150;
        } else if (ov == 1) {
            r = std::max(r, 38.0), g = std::max(g, 70.0), b = std::max(b, 88.0);
        }
        if (rp < radius) {
            const double angPx = rp * kDeg;// pixels per degree at this radius
            const double scr = pixBearing_[pi];
            // One-pixel lines, anti-aliased by their distance from the pixel.
            const double hd = angDiff(scr, headScreen) * angPx;
            if (hd < 1.0) {
                const double w = 0.85 * (1.0 - hd);
                r += (210 - r) * w, g += (220 - g) * w, b += (228 - b) * w;
            }
            const double sd = angDiff(scr, sweepScreen) * angPx;
            if (sd < 1.4) {
                const double w = 0.7 * (1.0 - sd / 1.4);
                r += (120 - r) * w, g += (255 - g) * w, b += (170 - b) * w;
            }
        }
        if (rp < 3.5) r = 255, g = 255, b = 255;
        px[0] = static_cast<std::uint8_t>(std::clamp(r, 0.0, 255.0));
        px[1] = static_cast<std::uint8_t>(std::clamp(g, 0.0, 255.0));
        px[2] = static_cast<std::uint8_t>(std::clamp(b, 0.0, 255.0));
        px[3] = 255;
    }
}
