#include "threepp/scenes/Weather.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

Vector3 RainCell::centreAt(double time) const {
    return {static_cast<float>(centre.x + velocity.x * time), centre.y,
            static_cast<float>(centre.z + velocity.z * time)};
}

double Weather::significantWaveHeight() const {
    // The constants are phillips_spectrum.comp's. Fully developed, its
    // Pierson-Moskowitz level and peak give Hs = 0.21 U^2 / g.
    constexpr double g = 9.81;
    const double U = std::max(0.0, static_cast<double>(windSpeed));
    double hsWind = 0.21 * U * U / g;
    if (fetch > 0.f && U > 0.0) {
        // JONSWAP fetch scaling: the shader scales the length L by r^(2/3)
        // and the level by r^-0.22, so the variance (level * L^2) goes as
        // r^(4/3 - 0.22) and Hs as its square root. The peak enhancement
        // gamma raises the variance by 1 / A_gamma, A_gamma = 1 - 0.287 ln
        // gamma (DNV-RP-C205's normalising factor): 1 at full development
        // (r = 1, gamma = 1), where the two branches meet.
        const double r = std::clamp(static_cast<double>(fetch) / (1609.0 * U * U), 0.005, 1.0);
        const double gamma = 1.0 + 2.3 * std::sqrt(1.0 - r);
        hsWind *= std::pow(r, 0.5 * (4.0 / 3.0 - 0.22)) * std::sqrt(1.0 / (1.0 - 0.287 * std::log(gamma)));
    }
    if (U > 0.0) {
        // A light wind has not grown its sea: the shader scales the wind
        // sea's variance at k by windGrowth(k, U) (phillips_spectrum.comp,
        // where the derivation is cited), 0 below 1 m/s and 1 from 4.5 m/s
        // at every k. Most of the variance is at the peak kp = 0.894 / L
        // (L as there, the fetch scaling included), so Hs goes as the
        // square root of the factor there.
        double L = 1.1624764 * U * U / g;
        if (fetch > 0.f) L *= std::pow(std::clamp(static_cast<double>(fetch) / (1609.0 * U * U), 0.005, 1.0), 2.0 / 3.0);
        const double kp = 0.894 / L;
        const double onset = std::clamp(0.36 * std::sqrt(std::sqrt(g * kp)), 1.0, 3.4);
        const double t = std::clamp((U - onset) / (4.5 - onset), 0.0, 1.0);
        hsWind *= std::sqrt(t * t * (3.0 - 2.0 * t));
    }
    if (U > 0.0 && U < 4.5) {
        // A light air's scales (phillips_spectrum.comp, lightAirScales): a
        // curvature band of mean square slope m about k_m = 370 rad/m, a
        // Gaussian in ln k of deviation s = 0.35, whose height variance is
        // m E[k^-2] = m exp(2 s^2) / k_m^2: under half a millimetre of Hs.
        const double a = std::clamp(U - 0.5, 0.0, 1.0);
        const double on = std::clamp(0.36 * std::sqrt(std::sqrt(g * 370.0)), 1.0, 3.4);
        const double tg = std::clamp((U - on) / (4.5 - on), 0.0, 1.0);
        const double m = 1.0e-3 * a * a * (1.0 - tg * tg * (3.0 - 2.0 * tg));
        const double hs = 4.0 * std::sqrt(m * std::exp(2.0 * 0.35 * 0.35)) / 370.0;
        hsWind = std::sqrt(hsWind * hsWind + hs * hs);
    }
    const double swell = std::max(0.0, static_cast<double>(swellHeight));
    return std::sqrt(hsWind * hsWind + swell * swell);
}

double Weather::rainRateAt(double x, double z, double time) const {
    double sum = 0.0;
    for (const RainCell& cell : rain) {
        if (cell.rainRate <= 0.f || cell.radius <= 0.f) continue;
        const Vector3 c = cell.centreAt(time);
        const double dx = x - c.x, dz = z - c.z;
        sum += cell.rainRate * std::exp(-(dx * dx + dz * dz) / (2.0 * cell.radius * cell.radius));
    }
    return sum;
}
