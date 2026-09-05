#include "threepp/helpers/SonarSensor.hpp"

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

SonarSensor::SonarSensor(const SonarModel& model)
    // The transducer frame is this node's own world frame. Noise off by
    // default, as for the path-traced lidar: the tracer's range is the truth.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0xA3C59AC3B7F0D1E5ULL}),
      model_(model), directions_(sonarRayDirections(model)),
      speckleRng_(speckleSeed()), speckleSeededWith_(rangeNoise.seed) {
    params.maxRange = model.maxRange;
    params.minRange = model.minRange;
    // The lidar equation's threshold gates on OPTICAL return strength; a
    // sonar target may be black paint. Every hit inside the range window is a
    // return; the acoustic model decides how bright.
    params.detectorThreshold = 0.f;
}

std::uint64_t SonarSensor::speckleSeed() const {
    // A fixed offset of the caller's seed: distinct from the range stream,
    // still a pure function of the seed the dataset records.
    return rangeNoise.seed ^ 0x5DEECE66D1CE4E5BULL;
}

void SonarSensor::resetNoise() {
    VisionSensor::resetNoise();
    speckleRng_ = SplitMix64{speckleSeed()};
    speckleSeededWith_ = rangeNoise.seed;
}

void SonarSensor::scan(VulkanRenderer& renderer, SonarImage& out) {
    scanBegin(renderer);
    if (!scanCollect(renderer, out)) out.reset(model_);
}

void SonarSensor::scanBegin(VulkanRenderer& renderer) {
    beginScan();
    if (rangeNoise.seed != speckleSeededWith_) {
        speckleRng_ = SplitMix64{speckleSeed()};
        speckleSeededWith_ = rangeNoise.seed;
    }
    scanHandle_ = VulkanRenderer::kNoLidarScan;
    if (directions_.empty()) return;

    if (!parent) updateMatrixWorld();

    // Origin = world translation; orientation = the world matrix's upper 3x3.
    // A sensor carries no scale, so that block is a rotation and direction
    // vectors go through it directly.
    getWorldPosition(scanOrigin_);
    Matrix3 rot;
    rot.setFromMatrix4(*matrixWorld);

    beamScratch_.resize(directions_.size());
    for (std::size_t i = 0; i < directions_.size(); ++i) {
        Vector3 d = directions_[i];
        d.applyMatrix3(rot).normalize();
        beamScratch_[i].origin = scanOrigin_;
        beamScratch_[i].direction = d;
    }

    scanHandle_ = renderer.scanLidarBegin(beamScratch_, params);
}

bool SonarSensor::scanReady(const VulkanRenderer& renderer) const {
    return scanHandle_ != VulkanRenderer::kNoLidarScan && renderer.scanLidarReady(scanHandle_);
}

bool SonarSensor::scanCollect(VulkanRenderer& renderer, SonarImage& out) {
    returns_.clear();
    if (scanHandle_ == VulkanRenderer::kNoLidarScan) {
        out.reset(model_);
        return false;
    }
    const int handle = scanHandle_;
    scanHandle_ = VulkanRenderer::kNoLidarScan;

    if (!renderer.scanLidarCollect(handle, returns_)) {
        out.reset(model_);
        return false;
    }

    // Range noise along each ray from where it was FIRED, then the fold, then
    // speckle on the folded image: noise on the measurement, speckle on the
    // picture, each from its own seeded stream.
    applyRangeNoise(returns_, scanOrigin_);
    sonarAccumulate(model_, reflectivity, returns_, scanOrigin_, out);
    out.time = lastScanTime();
    applySpeckle(out);
    return true;
}

// The path-traced lidar's rule, restated for the sonar: a return whose noisy
// range leaves [minRange, maxRange] becomes a miss, never a point at a range
// the sensor could not have reported.
void SonarSensor::applyRangeNoise(std::vector<LidarReturn>& returns, const Vector3& origin) {
    if (!rangeNoise.active()) return;

    for (auto& ret : returns) {
        if (ret.returnNo <= 0) continue;

        Vector3 dir = ret.position;
        dir.sub(origin);
        const float dist = dir.length();
        if (dist <= 1e-6f) continue;

        const float noisy = VisionSensor::applyRangeNoise(dist);
        if (noisy <= 0.f || noisy < params.minRange || noisy > params.maxRange) {
            ret.position.set(0.f, 0.f, 0.f);
            ret.normal.set(0.f, 0.f, 0.f);
            ret.hitInstanceId = -1;
            ret.returnNo = 0;
            ret.distance = 0.f;
            ret.intensity = 0.f;
            continue;
        }
        ret.position = dir.multiplyScalar(noisy / dist).add(origin);
        ret.distance = noisy;
    }
}

void SonarSensor::applySpeckle(SonarImage& image) {
    if (speckle <= 0.f) return;
    const float s = std::min(speckle, 1.f);
    // Every bin draws, echo or not, so the draw count — and therefore the
    // stream position of every later bin — depends only on the image size.
    for (float& v : image.intensity) {
        const float k = 1.f - s + 2.f * s * static_cast<float>(speckleRng_.nextDouble());
        v = std::min(1.f, v * k);
    }
}
