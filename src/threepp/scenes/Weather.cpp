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
        // sea's variance at k by windGrowth(k, kp, U) (phillips_spectrum.comp,
        // where the derivation is cited): the light air's peak ripples (a
        // quarter of the level) from 0.4 m/s, handed over to the grown sea from 1 to 2.5 m/s, and 1 from
        // 4.5 m/s at every k. Most of the variance is at the peak
        // kp = 0.894 / L (L as there, the fetch scaling included), so Hs goes
        // as the square root of the factor there.
        double L = 1.1624764 * U * U / g;
        if (fetch > 0.f) L *= std::pow(std::clamp(static_cast<double>(fetch) / (1609.0 * U * U), 0.005, 1.0), 2.0 / 3.0);
        const double kp = 0.894 / L;
        auto smooth = [](double e0, double e1, double x) {
            const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        };
        const double onset = std::clamp(0.36 * std::sqrt(std::sqrt(g * kp)), 1.0, 3.4);
        const double air   = 0.25 * smooth(0.4, 0.8, U) * (1.0 - smooth(1.0, 2.5, U));
        hsWind *= std::sqrt(std::max(smooth(onset, 4.5, U), air));
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
