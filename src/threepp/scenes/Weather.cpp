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
