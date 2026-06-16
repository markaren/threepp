
#include "threepp/helpers/PathTracedLidarSensor.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

PathTracedLidarSensor::PathTracedLidarSensor(unsigned int hRes, unsigned int vRes, float maxRange) {
    params.maxRange = maxRange;
    buildDenseBeams(hRes, vRes);
}

PathTracedLidarSensor::PathTracedLidarSensor(const LidarModel& model, float maxRange) {
    params.maxRange = maxRange;
    buildModelBeams(model);
}

PathTracedLidarSensor::PathTracedLidarSensor(float fovY, unsigned int width, unsigned int height, float maxRange) {
    params.maxRange = maxRange;
    buildCameraBeams(fovY, width, height);
}

void PathTracedLidarSensor::buildDenseBeams(unsigned int hRes, unsigned int vRes) {
    directions_.clear();
    directions_.reserve(static_cast<size_t>(hRes) * vRes);

    // Full-sphere equirectangular sampling. Sample centres ((i+0.5)/N) so no
    // beam lands exactly at the poles where multiple beams would degenerate
    // to the same direction.
    for (unsigned vi = 0; vi < vRes; ++vi) {
        const float v = (static_cast<float>(vi) + 0.5f) / static_cast<float>(vRes);
        const float elevation = (v - 0.5f) * math::PI;
        const float cosE = std::cos(elevation);
        const float sinE = std::sin(elevation);

        for (unsigned hi = 0; hi < hRes; ++hi) {
            const float u = (static_cast<float>(hi) + 0.5f) / static_cast<float>(hRes);
            const float azimuth = (u - 0.5f) * 2.f * math::PI;
            directions_.emplace_back(
                    cosE * std::sin(azimuth),
                    sinE,
                    -cosE * std::cos(azimuth));
        }
    }
}

void PathTracedLidarSensor::buildModelBeams(const LidarModel& model) {
    directions_.clear();

    const int numAzSteps = std::max(1, static_cast<int>(std::round(
                                                 (model.azimuthMax - model.azimuthMin) /
                                                 model.azimuthResolution)));
    directions_.reserve(static_cast<size_t>(numAzSteps) * model.elevationAngles.size());

    for (int ai = 0; ai < numAzSteps; ++ai) {
        const float azimuth = (model.azimuthMin + static_cast<float>(ai) * model.azimuthResolution) * math::DEG2RAD;
        const float sinA = std::sin(azimuth);
        const float cosA = std::cos(azimuth);
        for (float elevDeg : model.elevationAngles) {
            const float elevation = elevDeg * math::DEG2RAD;
            const float cosE = std::cos(elevation);
            const float sinE = std::sin(elevation);
            // azimuth = 0 → sensor-local -Z (forward); CCW from above.
            directions_.emplace_back(cosE * sinA, sinE, -cosE * cosA);
        }
    }
}

void PathTracedLidarSensor::buildCameraBeams(float fovY, unsigned int width, unsigned int height) {
    directions_.clear();
    directions_.reserve(static_cast<size_t>(width) * height);

    // Pinhole grid through pixel centres, matching DepthSensor's xDir_/yDir_
    // precompute: view direction = (dx, dy, -1), dx/dy in tan space.
    const float tanHalfY = std::tan(math::degToRad(fovY) * 0.5f);
    const float tanHalfX = tanHalfY * static_cast<float>(width) / static_cast<float>(height);

    for (unsigned y = 0; y < height; ++y) {
        const float dy = ((static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.f - 1.f) * tanHalfY;
        for (unsigned x = 0; x < width; ++x) {
            const float dx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.f - 1.f) * tanHalfX;
            Vector3 d(dx, dy, -1.f);
            directions_.emplace_back(d.normalize());
        }
    }
}

void PathTracedLidarSensor::scan(VulkanRenderer& renderer, std::vector<LidarReturn>& out) {
    out.clear();
    if (directions_.empty()) return;

    if (!parent) updateMatrixWorld();

    // Origin = sensor's world translation; orientation = upper-3x3 of world
    // matrix. Sensors don't typically carry scale, so the upper-3x3 is a
    // rotation matrix (no inverse-transpose required for direction vectors).
    Vector3 origin;
    getWorldPosition(origin);
    Matrix3 rot;
    rot.setFromMatrix4(*matrixWorld);

    beamScratch_.resize(directions_.size());
    for (size_t i = 0; i < directions_.size(); ++i) {
        Vector3 d = directions_[i];
        d.applyMatrix3(rot).normalize();
        beamScratch_[i].origin = origin;
        beamScratch_[i].direction = d;
    }

    renderer.scanLidar(beamScratch_, out, params);
}
