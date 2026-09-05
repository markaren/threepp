#include "threepp/helpers/SonarModel.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

void SonarImage::reset(const SonarModel& model) {
    beams = model.beams;
    bins = model.rangeBins;
    maxRange = model.maxRange;
    intensity.assign(static_cast<std::size_t>(beams) * bins, 0.f);
}

float SonarReflectivity::of(const LidarReturn& r) const {
    if (r.returnKind == static_cast<std::int32_t>(LidarReturnKind::VolumeScatter) || r.hitInstanceId == -2) {
        return volume;
    }
    const auto it = byInstance.find(r.hitInstanceId);
    return it == byInstance.end() ? defaultValue : it->second;
}

std::vector<Vector3> threepp::sonarRayDirections(const SonarModel& model) {
    std::vector<Vector3> dirs;
    dirs.reserve(model.rayCount());

    // Beam and sample CENTRES: the first beam is half a step inside the swath
    // edge, so no ray sits exactly on the fan boundary and the fan is
    // symmetric about forward whatever the beam count.
    for (unsigned int b = 0; b < model.beams; ++b) {
        const float azDeg = -0.5f * model.horizontalFov +
                            model.horizontalFov * (static_cast<float>(b) + 0.5f) / static_cast<float>(model.beams);
        const float az = azDeg * math::DEG2RAD;
        const float sinA = std::sin(az);
        const float cosA = std::cos(az);
        for (unsigned int s = 0; s < model.verticalSamples; ++s) {
            const float elDeg = -0.5f * model.verticalAperture +
                                model.verticalAperture * (static_cast<float>(s) + 0.5f) /
                                        static_cast<float>(model.verticalSamples);
            const float el = elDeg * math::DEG2RAD;
            const float cosE = std::cos(el);
            const float sinE = std::sin(el);
            // azimuth 0 -> local -Z (forward), positive toward local +X: the
            // LIDAR frame. Beam 0 is therefore the left-most beam.
            dirs.emplace_back(cosE * sinA, sinE, -cosE * cosA);
        }
    }
    return dirs;
}

void threepp::sonarAccumulate(const SonarModel& model, const SonarReflectivity& reflectivity,
                              const std::vector<LidarReturn>& returns, const Vector3& origin,
                              SonarImage& out) {
    out.reset(model);
    const std::size_t rays = model.rayCount();
    if (rays == 0 || model.rangeBins == 0 || returns.empty()) return;
    // Several returns per ray share the ray's beam; anything that is not a
    // whole multiple is a layout the caller did not produce with this model.
    const std::size_t perRay = std::max<std::size_t>(1, returns.size() / rays);
    const std::size_t usable = std::min(returns.size(), rays * perRay);

    const float invRange = 1.f / model.maxRange;
    const float floor = std::clamp(model.incidenceFloor, 0.f, 1.f);

    for (std::size_t i = 0; i < usable; ++i) {
        const LidarReturn& r = returns[i];
        if (r.returnNo <= 0) continue;
        const float range = r.distance;
        if (!(range > 0.f) || range < model.minRange || range > model.maxRange) continue;

        const unsigned int beam = static_cast<unsigned int>((i / perRay) / model.verticalSamples);
        if (beam >= model.beams) break;

        Vector3 d = r.position;
        d.sub(origin);
        const float len = d.length();
        // |n . d| with d the ray direction; a zero normal (a source that does
        // not report one) is treated as head-on rather than as no echo.
        float cosInc = 1.f;
        if (len > 1e-6f && r.normal.lengthSq() > 0.f) {
            cosInc = std::abs(r.normal.dot(d) / len);
        }
        const float strength = reflectivity.of(r) * (floor + (1.f - floor) * cosInc) *
                               std::exp(-2.f * model.attenuation * range);

        const unsigned int bin = std::min(model.rangeBins - 1,
                                          static_cast<unsigned int>(range * invRange * static_cast<float>(model.rangeBins)));
        float& cell = out.at(beam, bin);
        cell = std::max(cell, strength);
    }
}
