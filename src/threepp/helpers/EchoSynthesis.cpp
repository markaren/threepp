#include "threepp/helpers/EchoSynthesis.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

using namespace threepp;
using namespace threepp::echo;

double echo::scattererPhase(const Vector3& p, double cell, std::uint64_t seed) {
    const double inv = cell > 0.0 ? 1.0 / cell : 1.0;
    const auto qx = static_cast<std::int64_t>(std::floor(static_cast<double>(p.x) * inv));
    const auto qy = static_cast<std::int64_t>(std::floor(static_cast<double>(p.y) * inv));
    const auto qz = static_cast<std::int64_t>(std::floor(static_cast<double>(p.z) * inv));
    std::uint64_t k = combineKeys(seed, static_cast<std::uint64_t>(qx));
    k = combineKeys(k, static_cast<std::uint64_t>(qy));
    k = combineKeys(k, static_cast<std::uint64_t>(qz));
    return 2.0 * kPi * unitFromKey(k);
}

std::complex<double> echo::complexGaussian(std::uint64_t key) {
    const double u1 = unitFromKey(key);
    const double u2 = unitFromKey(key ^ 0xD1B54A32D192ED03ULL);
    // |z|^2 ~ Exp(1): unit mean power, Rayleigh amplitude, uniform phase.
    const double r = std::sqrt(-std::log(1.0 - u1));
    const double ph = 2.0 * kPi * u2;
    return {r * std::cos(ph), r * std::sin(ph)};
}

unsigned echo::workerCount(unsigned cap) {
    const unsigned hw = std::thread::hardware_concurrency();
    return std::clamp(hw == 0 ? 1u : hw, 1u, std::max(1u, cap));
}

// ── Tapers ─────────────────────────────────────────────────────────────────

namespace {

    // Taylor n-bar line-source weights at x in [-1/2, 1/2] (Taylor 1955, the
    // form in every antenna text): w(x) = 1 + 2 sum_m F_m cos(2 pi m x).
    double taylorWeight(double x, double sllDb, int nbar) {
        const double r = std::pow(10.0, sllDb / 20.0);
        const double A = std::acosh(r) / kPi;
        const double sigma2 = static_cast<double>(nbar * nbar) /
                              (A * A + (nbar - 0.5) * (nbar - 0.5));
        double w = 1.0;
        for (int m = 1; m < nbar; ++m) {
            double num = 1.0, den = 1.0;
            for (int n = 1; n < nbar; ++n) {
                num *= 1.0 - static_cast<double>(m * m) / (sigma2 * (A * A + (n - 0.5) * (n - 0.5)));
                if (n != m) den *= 1.0 - static_cast<double>(m * m) / static_cast<double>(n * n);
            }
            const double Fm = ((m % 2) ? 1.0 : -1.0) * num / (2.0 * den);
            w += 2.0 * Fm * std::cos(2.0 * kPi * m * x);
        }
        return w;
    }

    // Array factor of `w` at half-wavelength spacing, normalised to 1 at u = 0.
    double arrayFactor(const std::vector<double>& w, double sumW, double u) {
        const std::size_t M = w.size();
        const double c = 0.5 * static_cast<double>(M - 1);
        double s = 0.0;
        for (std::size_t m = 0; m < M; ++m) {
            s += w[m] * std::cos(2.0 * kPi * 0.5 * (static_cast<double>(m) - c) * u);
        }
        return s / sumW;
    }

    // -3 dB point of the array factor in u, by bisection inside the main lobe.
    double halfPowerPoint(const std::vector<double>& w, double sumW) {
        const double target = std::sqrt(0.5);
        // The main lobe of an M-element half-wavelength array ends before
        // u = 2 * 2 / M even for the heaviest taper here.
        double hi = std::min(1.0, 8.0 / static_cast<double>(w.size()));
        double lo = 0.0;
        while (arrayFactor(w, sumW, hi) > target && hi < 1.0) hi = std::min(1.0, hi * 1.5);
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            (arrayFactor(w, sumW, mid) > target ? lo : hi) = mid;
        }
        return 0.5 * (lo + hi);
    }

}// namespace

std::vector<double> echo::taperWeights(Taper taper, unsigned n) {
    std::vector<double> w(std::max(1u, n));
    for (unsigned m = 0; m < w.size(); ++m) {
        const double x = (static_cast<double>(m) + 0.5) / static_cast<double>(w.size()) - 0.5;
        switch (taper) {
            case Taper::Uniform: w[m] = 1.0; break;
            case Taper::Hann: w[m] = 0.5 * (1.0 + std::cos(2.0 * kPi * x)); break;
            case Taper::Hamming: w[m] = 0.54 + 0.46 * std::cos(2.0 * kPi * x); break;
            case Taper::Taylor25: w[m] = taylorWeight(x, 25.0, 4); break;
        }
    }
    return w;
}

ArrayPattern::ArrayPattern(double beamwidthDeg, Taper taper, double spanU, unsigned tableSize) {
    if (!(beamwidthDeg > 0.0) || beamwidthDeg >= 180.0) {
        throw std::invalid_argument("ArrayPattern: the beamwidth must be in (0, 180) degrees");
    }
    // The beamwidth factor of this taper: u_3dB * M is (nearly) a constant of
    // the taper once M is large, so measure it once on a long array and solve
    // for the element count that gives the requested beamwidth.
    constexpr unsigned kRef = 512;
    const auto wRef = taperWeights(taper, kRef);
    double sumRef = 0.0;
    for (double v : wRef) sumRef += v;
    const double kFactor = halfPowerPoint(wRef, sumRef) * static_cast<double>(kRef);

    const double uTarget = std::sin(0.5 * beamwidthDeg * kPi / 180.0);
    elements_ = std::max(2u, static_cast<unsigned>(std::lround(kFactor / uTarget)));
    aperture_ = 0.5 * static_cast<double>(elements_);

    const auto w = taperWeights(taper, elements_);
    double sumW = 0.0;
    for (double v : w) sumW += v;

    span_ = std::max(spanU, 1e-6);
    const unsigned n = std::max(16u, tableSize);
    table_.resize(n + 2);
    invStep_ = static_cast<double>(n) / span_;
    for (unsigned i = 0; i < table_.size(); ++i) {
        table_[i] = arrayFactor(w, sumW, static_cast<double>(i) / invStep_);
    }

    halfPowerU_ = halfPowerPoint(w, sumW);
    // Peak sidelobe: the largest |AF| past the first null of the main lobe.
    std::size_t i = 1;
    while (i + 1 < table_.size() && std::abs(table_[i + 1]) < std::abs(table_[i])) ++i;
    double peak = 0.0;
    for (; i < table_.size(); ++i) peak = std::max(peak, std::abs(table_[i]));
    peakSidelobeDb_ = peak > 0.0 ? 20.0 * std::log10(peak) : -400.0;
}

// ── Range response ─────────────────────────────────────────────────────────

namespace {
    double sinc(double x) {
        if (std::abs(x) < 1e-9) return 1.0;
        const double px = kPi * x;
        return std::sin(px) / px;
    }
}// namespace

RangeResponse::RangeResponse(double resolution, PulseKind kind)
    : resolution_(std::max(resolution, 1e-6)), kind_(kind) {
    if (kind_ == PulseKind::CW) {
        // Autocorrelation of a rectangular pulse: a triangle whose half-base
        // is c tau / 2, the range resolution of the pulse.
        halfSpan_ = resolution_;
        scale_ = 1.0 / resolution_;
    } else {
        // Hann-weighted LFM: 0.5 sinc(x) + 0.25 [sinc(x - 1) + sinc(x + 1)]
        // with x = dr / (c / 2B); `resolution` is c / 2B. Main lobe to
        // |x| = 2, kept to |x| = 4 (the first two sidelobes, -31 and -41 dB).
        halfSpan_ = 4.0 * resolution_;
        scale_ = 1.0 / resolution_;
    }
}

double RangeResponse::operator()(double dr) const {
    const double a = std::abs(dr);
    if (a >= halfSpan_) return 0.0;
    if (kind_ == PulseKind::CW) return 1.0 - a * scale_;
    const double x = dr * scale_;
    return (0.5 * sinc(x) + 0.25 * (sinc(x - 1.0) + sinc(x + 1.0))) / 0.5;
}
