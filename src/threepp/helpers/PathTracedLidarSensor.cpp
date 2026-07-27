
#include "threepp/helpers/PathTracedLidarSensor.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

PathTracedLidarSensor::PathTracedLidarSensor(unsigned int hRes, unsigned int vRes, float maxRange)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Noise off by default: this back-end reports the tracer's own range.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0xD1342543DE82EF95ULL}) {
    params.maxRange = maxRange;
    buildDenseBeams(hRes, vRes);
}

PathTracedLidarSensor::PathTracedLidarSensor(const LidarModel& model, float maxRange)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Noise off by default: this back-end reports the tracer's own range.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0xD1342543DE82EF95ULL}) {
    params.maxRange = maxRange;
    buildModelBeams(model);
}

PathTracedLidarSensor::PathTracedLidarSensor(float fovY, unsigned int width, unsigned int height, float maxRange)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Noise off by default: this back-end reports the tracer's own range.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0xD1342543DE82EF95ULL}) {
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
    beginScan();
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

    // Queues these beams and returns the PREVIOUS submit's results, so the
    // returns must be perturbed about the origin they were traced from.
    renderer.scanLidar(beamScratch_, out, params);

    if (haveInFlight_) applyNoise(out, inFlightOrigin_);
    else out.clear();// nothing was in flight ⇒ nothing to hand back
    inFlightOrigin_ = origin;
    haveInFlight_   = true;
}

// Perturb each return along its own beam. Applied CPU-side rather than in the
// rgen shader on purpose: the GPU's per-beam RNG is seeded from launch index +
// frame, so it could not honour a caller's seed, and a scan replayed on another
// GPU would not reproduce it. Here the stream is the sensor's own.
//
// A return whose noisy range leaves (0, maxRange] becomes a miss rather than a
// point at a range the sensor could not have reported — same rule the raster
// sensors apply when noise pushes a depth past the far plane.
void PathTracedLidarSensor::applyNoise(std::vector<LidarReturn>& out, const Vector3& origin) {
    if (!rangeNoise.active()) return;

    for (auto& ret : out) {
        if (ret.returnNo <= 0) continue;// miss: nothing was measured to corrupt

        Vector3 dir = ret.position;
        dir.sub(origin);
        const float dist = dir.length();
        if (dist <= 1e-6f) continue;

        const float noisy = applyRangeNoise(dist);
        if (noisy <= 0.f || noisy > params.maxRange) {
            // Full miss per the LidarTypes.hpp contract (normal = 0 too), so a
            // consumer that ignores the sentinel rules still sees miss-shaped
            // data rather than the stale hit fields.
            ret.position.set(0.f, 0.f, 0.f);
            ret.normal.set(0.f, 0.f, 0.f);
            ret.hitInstanceId = -1;
            ret.returnNo = 0;
            ret.distance = 0.f;
            ret.intensity = 0.f;
            continue;
        }
        ret.position = dir.multiplyScalar(noisy / dist).add(origin);
        // `distance` is the slant range the consumer reads; keep it consistent
        // with the point that was just moved.
        ret.distance = noisy;
    }
}
