#include "threepp/helpers/ImagingSonarModel.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

using namespace threepp;
using namespace threepp::echo;

// ── Water ──────────────────────────────────────────────────────────────────

double WaterProperties::soundSpeed() const {
    // Mackenzie, "Nine-term equation for sound speed in the oceans", JASA 70 (1981).
    const double T = temperature, S = salinity, D = depth;
    return 1448.96 + 4.591 * T - 5.304e-2 * T * T + 2.374e-4 * T * T * T + 1.340 * (S - 35.0) +
           1.630e-2 * D + 1.675e-7 * D * D - 1.025e-2 * T * (S - 35.0) - 7.139e-13 * T * D * D * D;
}

double WaterProperties::absorptionDbPerKm(double frequencyHz) const {
    // Francois & Garrison, JASA 72 (1982): boric acid and magnesium sulphate
    // relaxations plus pure-water viscous absorption. f in kHz, result dB/km.
    const double f = frequencyHz * 1e-3;
    const double T = temperature, S = salinity, z = depth;
    const double c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z;
    const double theta = T + 273.0;

    const double A1 = 8.86 / c * std::pow(10.0, 0.78 * pH - 5.0);
    const double f1 = 2.8 * std::sqrt(S / 35.0) * std::pow(10.0, 4.0 - 1245.0 / theta);

    const double A2 = 21.44 * S / c * (1.0 + 0.025 * T);
    const double P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z * z;
    const double f2 = 8.17 * std::pow(10.0, 8.0 - 1990.0 / theta) / (1.0 + 0.0018 * (S - 35.0));

    const double P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z * z;
    const double A3 = T <= 20.0 ? 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T * T - 1.50e-8 * T * T * T
                                : 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T * T - 6.5e-10 * T * T * T;
    const double ff = f * f;
    return A1 * f1 * ff / (f1 * f1 + ff) + A2 * P2 * f2 * ff / (f2 * f2 + ff) + A3 * P3 * ff;
}

// ── Materials ──────────────────────────────────────────────────────────────

double AcousticMaterial::sigma0(double cosTheta) const {
    const double c = std::clamp(cosTheta, 0.0, 1.0);
    const double c2 = c * c;
    double s = dbToPower(lambertDb) * c2;
    if (rmsSlope > 0.f && c > 1e-3) {
        // Per steradian, as acoustics counts it: the radar's form over 4 pi.
        const double s2 = static_cast<double>(rmsSlope) * rmsSlope;
        const double tan2 = (1.0 - c2) / c2;
        s += dbToPower(reflectionLossDb) * std::exp(-tan2 / s2) / (4.0 * kPi * s2 * c2 * c2);
    }
    return s;
}

// ── Presets ────────────────────────────────────────────────────────────────

ImagingSonarModel ImagingSonarModel::OculusM750d() {
    ImagingSonarModel m;
    m.frequency = 750e3;
    m.horizontalFov = 130.f;
    m.beams = 512;
    m.horizontalBeamwidth = 1.0f;
    m.verticalBeamwidth = 20.f;
    m.maxRange = 40.f;
    m.rangeResolution = 0.025f;
    return m;
}

ImagingSonarModel ImagingSonarModel::OculusM750dHigh() {
    ImagingSonarModel m;
    m.frequency = 1.2e6;
    m.horizontalFov = 70.f;
    m.beams = 512;
    m.horizontalBeamwidth = 0.6f;
    m.verticalBeamwidth = 12.f;
    m.maxRange = 30.f;
    m.rangeResolution = 0.012f;
    m.sourceLevelDb = 204.f;
    m.noiseSpectrumLevelDb = 49.f;// thermal noise rises 20 dB per decade
    return m;
}

ImagingSonarModel ImagingSonarModel::Gemini720is() {
    ImagingSonarModel m;
    m.frequency = 720e3;
    m.horizontalFov = 120.f;
    m.beams = 512;
    m.horizontalBeamwidth = 0.5f;
    m.verticalBeamwidth = 20.f;
    m.maxRange = 50.f;
    m.rangeResolution = 0.012f;
    return m;
}

ImagingSonarModel ImagingSonarModel::BlueViewM900() {
    ImagingSonarModel m;
    m.frequency = 900e3;
    m.horizontalFov = 130.f;
    m.beams = 768;
    m.horizontalBeamwidth = 1.0f;
    m.verticalBeamwidth = 20.f;
    m.maxRange = 60.f;
    m.rangeResolution = 0.013f;
    m.noiseSpectrumLevelDb = 47.f;
    return m;
}

// ── Synthesizer ────────────────────────────────────────────────────────────

namespace {

    constexpr double kDeg = kPi / 180.0;

    // amp * e^(i phase) for a signed amplitude (std::polar needs rho >= 0, and
    // the patterns' sidelobes are negative).
    std::complex<double> phasor(double amp, double phase) {
        return {amp * std::cos(phase), amp * std::sin(phase)};
    }

    struct Contribution {
        double u;
        std::complex<float> amp;// peak amplitude (for labels)
        std::int32_t id;
        std::uint32_t firstBin;
        std::uint32_t weightsOffset;
        std::uint16_t nbins;
        std::uint16_t centreBin;// offset of the bin holding the range, within [0, nbins)
        std::uint8_t kind;
    };

}// namespace

ImagingSonarSynthesizer::ImagingSonarSynthesizer(const ImagingSonarModel& model) {
    columns_ = std::max(1u, model.beams * std::max(1u, model.azimuthOversample));
    rows_ = std::max(1u, model.verticalSamples);

    const double hbw = std::clamp(static_cast<double>(model.horizontalBeamwidth), 0.05, 60.0);
    const double spanU = std::max(1.0, static_cast<double>(model.patternSpanBeamwidths)) * 2.0 *
                         std::sin(0.5 * hbw * kDeg);
    hPattern_ = ArrayPattern(hbw, model.horizontalTaper, spanU);
    // The vertical fan: one transducer stave, an untapered line source,
    // tabulated over every elevation.
    vPattern_ = ArrayPattern(std::clamp(static_cast<double>(model.verticalBeamwidth), 0.5, 120.0),
                             Taper::Uniform, 1.0);
    // Rays out to the vertical pattern's first null (2.26 x the half-power
    // point for an untapered line source): the skirts matter, past the null
    // the sidelobes are 13 dB down twice over.
    elevationSpan_ = std::asin(std::min(0.999, 2.26 * vPattern_.halfPowerU()));

    const double fov = std::clamp(static_cast<double>(model.horizontalFov), 1.0, 179.0) * kDeg;
    columnWidth_ = fov / static_cast<double>(columns_);

    dirs_.reserve(static_cast<std::size_t>(columns_) * rows_);
    for (unsigned c = 0; c < columns_; ++c) {
        for (unsigned r = 0; r < rows_; ++r) {
            const std::uint64_t key = combineKeys(combineKeys(model.seed, c), r);
            const double az = -0.5 * fov + fov * stratum(c, columns_, key, true);
            const double el = -elevationSpan_ + 2.0 * elevationSpan_ * stratum(r, rows_, key ^ 0x9E37ULL, true);
            const double ce = std::cos(el);
            dirs_.emplace_back(static_cast<float>(ce * std::sin(az)), static_cast<float>(std::sin(el)),
                               static_cast<float>(-ce * std::cos(az)));
        }
    }
}

void ImagingSonarSynthesizer::synthesize(const ImagingSonarModel& model, const AcousticMaterials& materials,
                                         const std::vector<LidarReturn>& returns, const Vector3& origin,
                                         const Matrix3& worldFromSensor, std::uint64_t pingKey, SonarFrame& out,
                                         unsigned threads) const {
    const unsigned beams = model.beams;
    const unsigned bins = std::max(1u, model.rangeBins);
    const double r0 = std::max(0.f, model.minRange);
    const double r1 = std::max(model.maxRange, model.minRange + 0.01f);
    const double binW = (r1 - r0) / bins;

    out.beams = beams;
    out.bins = bins;
    out.minRange = static_cast<float>(r0);
    out.maxRange = static_cast<float>(r1);
    out.bearingsDeg.resize(beams);
    for (unsigned b = 0; b < beams; ++b) out.bearingsDeg[b] = model.beamBearingDeg(b);
    const std::size_t cells = static_cast<std::size_t>(beams) * bins;
    out.levelDb.assign(cells, 0.f);
    out.display.assign(cells, 0.f);
    out.label.assign(cells, -1);
    out.labelKind.assign(cells, static_cast<std::uint8_t>(SonarEchoKind::None));

    // ── Acoustics of this ping ───────────────────────────────────────────
    const double c = model.water.soundSpeed();
    const double lambda = c / model.frequency;
    const double k = 2.0 * kPi / lambda;
    const double alpha = model.water.absorptionDbPerKm(model.frequency) * 1e-3;// dB/m
    const double SL = dbToPower(model.sourceLevelDb);
    const RangeResponse resp(model.rangeResolution, model.pulse);
    const double hs = resp.halfSpan();
    const double cellSize = std::max(lambda, 0.25 * static_cast<double>(model.rangeResolution));
    const double dEl = 2.0 * elevationSpan_ / rows_;

    Matrix3 sensorFromWorld = worldFromSensor;
    sensorFromWorld.transpose();

    // ── Scatterers -> contributions ──────────────────────────────────────
    std::vector<Contribution> contrib;
    std::vector<std::complex<float>> weighted;
    contrib.reserve(returns.size() * 2);
    weighted.reserve(returns.size() * 8);

    auto add = [&](double u, double range, std::complex<double> a, std::int32_t id, SonarEchoKind kind) {
        if (range + hs < r0 || range - hs > r1) return;
        const double x = (range - r0) / binW;
        int jlo = static_cast<int>(std::floor(x - hs / binW));
        int jhi = static_cast<int>(std::floor(x + hs / binW));
        jlo = std::max(jlo, 0);
        jhi = std::min(jhi, static_cast<int>(bins) - 1);
        if (jhi < jlo) return;
        if (jhi - jlo > 63) {// a response far wider than a bin: keep its core
            // The core of a range just outside the window is the window's
            // edge: clamped into [jlo, jhi], the core never leaves it.
            const int jc = std::clamp(static_cast<int>(std::floor(x)), jlo, jhi);
            jlo = std::max(jc - 31, jlo);
            jhi = std::min(jc + 32, jhi);
        }
        Contribution ct{};
        ct.u = u;
        ct.amp = std::complex<float>(a);
        ct.id = id;
        ct.kind = static_cast<std::uint8_t>(kind);
        ct.firstBin = static_cast<std::uint32_t>(jlo);
        ct.nbins = static_cast<std::uint16_t>(jhi - jlo + 1);
        ct.centreBin = static_cast<std::uint16_t>(std::clamp(static_cast<int>(std::floor(x)), jlo, jhi) - jlo);
        ct.weightsOffset = static_cast<std::uint32_t>(weighted.size());
        // The matched filter's output averaged over each bin (4 taps), so a
        // pulse shorter than a bin still lands its energy, wherever in the
        // bin the scatterer sits.
        for (int j = jlo; j <= jhi; ++j) {
            double w = 0.0;
            for (int s = 0; s < 4; ++s) {
                const double rc = r0 + (j + (s + 0.5) * 0.25) * binW;
                w += resp(rc - range);
            }
            weighted.emplace_back(std::complex<float>(a * (0.25 * w)));
        }
        contrib.push_back(ct);
    };

    const std::size_t rays = std::min(returns.size(), dirs_.size());
    const bool multipath = model.surfaceMultipath && origin.y < model.surfaceY;
    const double ys = model.surfaceY;
    const double incoherent = dbToPower(-std::abs(model.incoherentSurfaceLossDb));

    for (std::size_t i = 0; i < rays; ++i) {
        const LidarReturn& r = returns[i];
        if (r.returnNo <= 0 || r.returnKind != 0 || r.hitInstanceId == -2) continue;

        Vector3 dW = r.position;
        dW.sub(origin);
        const double R = dW.length();
        if (R < r0 || R > r1 + hs || R < 1e-4) continue;
        dW.multiplyScalar(static_cast<float>(1.0 / R));

        Vector3 dS = dW;
        dS.applyMatrix3(sensorFromWorld);
        const double u = dS.x;
        const double gv = vPattern_(dS.y);

        const AcousticMaterial& mat = materials.of(r.hitInstanceId);
        double cosT = 1.0;
        if (r.normal.lengthSq() > 0.f) cosT = std::abs(static_cast<double>(r.normal.dot(dW)));
        const double sig0 = mat.sigma0(cosT);
        if (!(sig0 > 0.0)) continue;

        const double ey = dirs_[i].y;
        const double dOmega = columnWidth_ * dEl * std::sqrt(std::max(0.0, 1.0 - ey * ey));
        const double dA = R * R * dOmega / std::max(cosT, 0.03);

        const double I = SL * std::pow(R, -4.0) * std::pow(10.0, -0.2 * alpha * R) * sig0 * dA;
        // Each ray is its own scatterer: rays sharing a phase cell would add
        // coherently and their energy would grow with the ray density.
        const double phase0 = scattererPhase(r.position, cellSize, combineKeys(model.seed, i));
        add(u, R, phasor(std::sqrt(I) * gv * gv, 2.0 * k * R + phase0), r.hitInstanceId, SonarEchoKind::Direct);

        if (!multipath || r.position.y >= ys) continue;

        // ── The sea surface as a mirror: the image of the transducer ────
        Vector3 oi = origin;
        oi.y = static_cast<float>(2.0 * ys - origin.y);
        Vector3 toP = r.position;
        toP.sub(oi);
        const double rs = toP.length();
        const double t = (ys - oi.y) / (static_cast<double>(r.position.y) - oi.y);
        Vector3 s = toP;
        s.multiplyScalar(static_cast<float>(t)).add(oi);// the specular point
        Vector3 leg = s;
        leg.sub(origin);
        const double ls = leg.length();
        if (ls < 1e-4 || rs < 1e-4) continue;
        leg.multiplyScalar(static_cast<float>(1.0 / ls));
        Vector3 legS = leg;
        legS.applyMatrix3(sensorFromWorld);
        const double us = legS.x;
        const double gvs = vPattern_(legS.y);
        if (gvs == 0.0) continue;

        // Rayleigh roughness: the coherent bounce dies as k sigma sin(psi) grows.
        const double sinPsi = (ys - origin.y) / ls;
        const double kss = k * model.surfaceRmsHeight * sinPsi;
        const double rho = std::exp(-2.0 * kss * kss);
        const double incPart = std::sqrt((1.0 - rho * rho) * incoherent);
        const double incPhase = 2.0 * kPi * unitFromKey(combineKeys(combineKeys(pingKey, i), 0x5EAULL));
        const std::complex<double> gamma = -rho + phasor(incPart, incPhase);

        Vector3 inc = r.position;
        inc.sub(s);
        inc.normalize();
        double cosTs = 1.0;
        if (r.normal.lengthSq() > 0.f) cosTs = std::abs(static_cast<double>(r.normal.dot(inc)));
        const double sig0s = mat.sigma0(cosTs);

        // One leg each way: path R + rs, twice (out direct / back via the
        // surface, and the reverse), received from two directions.
        const double Lc = R + rs;
        const double Ic = SL / (R * R * rs * rs) * std::pow(10.0, -0.1 * alpha * Lc) * std::sqrt(sig0 * sig0s) * dA;
        const std::complex<double> ac = phasor(std::sqrt(Ic) * gv * gvs, k * Lc + phase0) * gamma;
        add(us, 0.5 * Lc, ac, r.hitInstanceId, SonarEchoKind::Multipath);
        add(u, 0.5 * Lc, ac, r.hitInstanceId, SonarEchoKind::Multipath);
        // Both legs via the surface.
        const double Id = SL * std::pow(rs, -4.0) * std::pow(10.0, -0.2 * alpha * rs) * sig0s * dA;
        const std::complex<double> ad = phasor(std::sqrt(Id) * gvs * gvs, 2.0 * k * rs + phase0) * gamma * gamma;
        add(us, rs, ad, r.hitInstanceId, SonarEchoKind::Multipath);
    }

    // Bucketed by u (a counting sort: stable, so a bucket keeps ray order), so
    // each beam reads the buckets its pattern spans and sums them in a fixed
    // order. The pattern itself is 0 past its span, so whole buckets suffice.
    constexpr unsigned kBuckets = 8192;
    auto bucketOf = [](double u) {
        return static_cast<unsigned>(std::clamp((u + 1.0) * 0.5 * kBuckets, 0.0, kBuckets - 1.0));
    };
    std::vector<std::uint32_t> start(kBuckets + 1, 0u);
    for (const auto& ct : contrib) ++start[bucketOf(ct.u) + 1];
    for (unsigned b = 0; b < kBuckets; ++b) start[b + 1] += start[b];
    std::vector<std::uint32_t> order(contrib.size());
    {
        std::vector<std::uint32_t> fill(start.begin(), start.end() - 1);
        for (std::uint32_t i = 0; i < contrib.size(); ++i) order[fill[bucketOf(contrib[i].u)]++] = i;
    }

    // ── Reverberation and noise levels ───────────────────────────────────
    const double hbwRad = model.horizontalBeamwidth * kDeg;
    const double vbwRad = model.verticalBeamwidth * kDeg;
    const double bandwidth = c / (2.0 * std::max(1e-4, static_cast<double>(model.rangeResolution)));
    const double di = 10.0 * std::log10(4.0 * kPi / (hbwRad * vbwRad));
    const double noiseDb = model.noiseSpectrumLevelDb + 10.0 * std::log10(bandwidth) - di;
    const double noiseP = dbToPower(noiseDb);
    const double sv = dbToPower(model.volumeBackscatterDb);
    out.noiseLevelDb = static_cast<float>(noiseDb);

    auto tvg = [&](double R) {
        return model.tvgSpreading * std::log10(std::max(R, 0.1)) + 2.0 * alpha * R;
    };
    // Black sits 14 dB under the noise floor's level at full range after TVG:
    // the near water column is black, far noise a dim grain, a seabed mid-scale.
    const double black = noiseDb + tvg(r1) - 14.0 - model.gainDb;
    const double invDyn = 1.0 / std::max(1.f, model.dynamicRangeDb);
    const double span = hPattern_.span();

    std::vector<std::complex<float>> acc(cells);
    std::vector<float> labelAmp(cells, 0.f);
    // Per range bin: the reverberation level without the beam's solid angle, and the TVG.
    std::vector<double> pvBase(bins), tvgBin(bins);
    for (unsigned j = 0; j < bins; ++j) {
        const double R = r0 + (j + 0.5) * binW;
        pvBase[j] = SL / (R * R) * std::pow(10.0, -0.2 * alpha * R) * sv *
                    std::max(static_cast<double>(model.rangeResolution), binW);
        tvgBin[j] = tvg(R);
    }

    auto work = [&](unsigned b0, unsigned b1) {
        for (unsigned b = b0; b < b1; ++b) {
            const double ub = std::sin(model.beamBearingDeg(b) * kDeg);
            const std::size_t base = static_cast<std::size_t>(b) * bins;
            const std::size_t lo = start[bucketOf(ub - span)];
            const std::size_t hi = start[bucketOf(ub + span) + 1];
            for (std::size_t o = lo; o < hi; ++o) {
                const Contribution& ct = contrib[order[o]];
                const float w = static_cast<float>(hPattern_(ct.u - ub));
                if (w == 0.f) continue;
                std::complex<float>* cell = &acc[base + ct.firstBin];
                const std::complex<float>* wt = &weighted[ct.weightsOffset];
                for (unsigned j = 0; j < ct.nbins; ++j) cell[j] += w * wt[j];
                const float a = std::abs(w) * std::abs(ct.amp);
                const std::size_t lc = base + ct.firstBin + ct.centreBin;
                if (a > labelAmp[lc]) {
                    labelAmp[lc] = a;
                    out.label[lc] = ct.id;
                    out.labelKind[lc] = ct.kind;
                }
            }

            // Steered beams widen as 1/cos: so does the volume they insonify.
            const double steer = std::max(0.2, std::cos(model.beamBearingDeg(b) * kDeg));
            const double psi = (1.06 * hbwRad / steer) * (0.75 * vbwRad);
            const std::uint64_t beamKey = combineKeys(pingKey, b);
            for (unsigned j = 0; j < bins; ++j) {
                const std::size_t ci = base + j;
                const double pv = pvBase[j] * psi;
                // Reverberation and noise are independent complex Gaussians:
                // their sum is one, with the summed variance.
                std::complex<double> z(acc[ci]);
                z += std::sqrt(pv + noiseP) * complexGaussian(combineKeys(beamKey, j));
                const double p = std::norm(z);
                const double level = powerToDb(p, -100.0);
                out.levelDb[ci] = static_cast<float>(level);
                out.display[ci] = static_cast<float>(std::clamp((level + tvgBin[j] - black) * invDyn, 0.0, 1.0));
                const double echoP = static_cast<double>(labelAmp[ci]) * labelAmp[ci];
                if (echoP < 0.5 * pv) {
                    out.label[ci] = pv > noiseP ? -2 : -1;
                    out.labelKind[ci] = static_cast<std::uint8_t>(pv > noiseP ? SonarEchoKind::Volume
                                                                               : SonarEchoKind::None);
                }
            }
        }
    };

    const unsigned nT = std::max(1u, std::min(threads ? threads : workerCount(16), beams));
    if (nT == 1) {
        work(0, beams);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(nT - 1);
    const unsigned chunk = (beams + nT - 1) / nT;
    for (unsigned t = 1; t < nT; ++t) {
        const unsigned b0 = std::min(beams, t * chunk), b1 = std::min(beams, (t + 1) * chunk);
        if (b0 < b1) pool.emplace_back(work, b0, b1);
    }
    work(0, std::min(beams, chunk));
    for (auto& th : pool) th.join();
}

// ── Fan display ────────────────────────────────────────────────────────────

SonarFanDisplay::SonarFanDisplay(unsigned width, unsigned height)
    : w_(std::max(64u, width)), h_(std::max(64u, height)) {
    rgba_.assign(static_cast<std::size_t>(w_) * h_ * 4, 0);
}

void SonarFanDisplay::draw(const SonarFrame& f, float fovDeg) {
    if (f.beams == 0 || f.bins == 0) return;
    const std::size_t npx = static_cast<std::size_t>(w_) * h_;
    if (f.beams != keyBeams_ || f.bins != keyBins_ || fovDeg != keyFov_ || f.minRange != keyMin_ ||
        f.maxRange != keyMax_ || ringStep != keyRing_) {
        keyBeams_ = f.beams, keyBins_ = f.bins, keyFov_ = fovDeg;
        keyMin_ = f.minRange, keyMax_ = f.maxRange, keyRing_ = ringStep;
        cellOf_.assign(npx, -1);
        pixBeam_.assign(npx, -1);
        pixFrac_.assign(npx, 0.f);
        pixBin0_.assign(npx, 0);
        pixBins_.assign(npx, 1);
        overlay_.assign(npx, 0);
        const double half = 0.5 * fovDeg * kDeg;
        const double margin = 10.0;
        const double scale = std::min((h_ - 2.0 * margin) / f.maxRange,
                                      (w_ - 2.0 * margin) / (2.0 * std::sin(std::min(half, 0.5 * kPi)) * f.maxRange));
        const double apexX = 0.5 * (w_ - 1), apexY = h_ - margin;// image rows grow downward
        double ring = ringStep;
        if (ring <= 0.f) {
            const double target = f.maxRange / 5.0;
            const double p = std::pow(10.0, std::floor(std::log10(target)));
            ring = target / p < 2.0 ? p : (target / p < 5.0 ? 2.0 * p : 5.0 * p);
        }
        for (unsigned y = 0; y < h_; ++y) {
            for (unsigned x = 0; x < w_; ++x) {
                const double dx = x - apexX, dy = apexY - y;
                const double rp = std::sqrt(dx * dx + dy * dy);
                const double r = rp / scale;
                const double brg = std::atan2(dx, dy);
                const std::size_t i = static_cast<std::size_t>(y) * w_ + x;
                const bool inside = dy > 0.0 && std::abs(brg) <= half && r <= f.maxRange;
                if (inside && r >= f.minRange) {
                    const auto b = static_cast<int>((brg + half) / (2.0 * half) * f.beams);
                    const auto j = static_cast<int>((r - f.minRange) / (f.maxRange - f.minRange) * f.bins);
                    cellOf_[i] = std::clamp(b, 0, static_cast<int>(f.beams) - 1) * static_cast<int>(f.bins) +
                                 std::clamp(j, 0, static_cast<int>(f.bins) - 1);
                    // Between the two nearest beam centres, and over every bin
                    // the pixel spans in range (a pixel is often several bins).
                    // A one-beam sonar keeps the nearest-cell lookup above.
                    if (f.beams >= 2) {
                        const double bf = (brg + half) / (2.0 * half) * f.beams - 0.5;
                        const int b0 = std::clamp(static_cast<int>(std::floor(bf)), 0, static_cast<int>(f.beams) - 2);
                        pixBeam_[i] = b0;
                        pixFrac_[i] = static_cast<float>(std::clamp(bf - b0, 0.0, 1.0));
                        const double binsPerPx = f.bins / ((f.maxRange - f.minRange) * scale);
                        const int nj = std::clamp(static_cast<int>(std::lround(binsPerPx)), 1,
                                                  std::min(12, static_cast<int>(f.bins)));
                        const int jc = static_cast<int>((r - f.minRange) / (f.maxRange - f.minRange) * f.bins);
                        pixBin0_[i] = std::clamp(jc - nj / 2, 0, static_cast<int>(f.bins) - nj);
                        pixBins_[i] = static_cast<std::uint8_t>(nj);
                    }
                }
                if (!inside) continue;
                const double dRing = std::abs(rp - ring * scale * std::round(r / ring));
                const double dEdge = (half - std::abs(brg)) * rp;
                if (dEdge < 1.0 || f.maxRange * scale - rp < 1.0) overlay_[i] = 2;
                else if (r > 0.5 * ring && dRing < 0.5) overlay_[i] = 1;
            }
        }
    }

    for (std::size_t i = 0; i < npx; ++i) {
        std::uint8_t* px = &rgba_[i * 4];
        const std::int32_t c = cellOf_[i];
        double r = 9, g = 11, b = 14;
        if (c >= 0) {
            double v = f.display[static_cast<std::size_t>(c)];
            if (pixBeam_[i] >= 0) {
                const std::size_t b0 = static_cast<std::size_t>(pixBeam_[i]) * f.bins + pixBin0_[i];
                const std::size_t b1 = b0 + f.bins;
                double s0 = 0.0, s1 = 0.0;
                for (unsigned j = 0; j < pixBins_[i]; ++j) {
                    s0 += f.display[b0 + j];
                    s1 += f.display[b1 + j];
                }
                const double fr = pixFrac_[i];
                v = ((1.0 - fr) * s0 + fr * s1) / pixBins_[i];
            }
            if (showLabels && f.label[static_cast<std::size_t>(c)] >= 0 && v > 0.05) {
                const std::uint64_t hsh = mix64(static_cast<std::uint64_t>(f.label[static_cast<std::size_t>(c)]) + 91);
                const double s = std::sqrt(v);
                r = (70 + (hsh & 0xB9)) * s, g = (70 + ((hsh >> 8) & 0xB9)) * s, b = (70 + ((hsh >> 16) & 0xB9)) * s;
            } else {
                // Copper: black -> rust -> amber -> near white.
                const double t = std::clamp(v, 0.0, 1.0);
                if (t < 0.45) {
                    const double u = t / 0.45;
                    r = 4 + 150 * u, g = 3 + 52 * u, b = 2 + 12 * u;
                } else if (t < 0.8) {
                    const double u = (t - 0.45) / 0.35;
                    r = 154 + 96 * u, g = 55 + 120 * u, b = 14 + 40 * u;
                } else {
                    const double u = (t - 0.8) / 0.2;
                    r = 250 + 5 * u, g = 175 + 70 * u, b = 54 + 160 * u;
                }
            }
        }
        if (overlay_[i] == 2) r = std::max(r, 120.0), g = std::max(g, 104.0), b = std::max(b, 80.0);
        else if (overlay_[i] == 1) r = std::max(r, 62.0), g = std::max(g, 54.0), b = std::max(b, 42.0);
        px[0] = static_cast<std::uint8_t>(std::clamp(r, 0.0, 255.0));
        px[1] = static_cast<std::uint8_t>(std::clamp(g, 0.0, 255.0));
        px[2] = static_cast<std::uint8_t>(std::clamp(b, 0.0, 255.0));
        px[3] = 255;
    }
}
