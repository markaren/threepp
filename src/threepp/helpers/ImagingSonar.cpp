#include "threepp/helpers/ImagingSonar.hpp"

#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

using namespace threepp;

ImagingSonar::ImagingSonar(const ImagingSonarModel& m)
    : VisionSensor(*this, RangeNoiseModel{0.f, 0.f, 0.f, 0x50A4ULL}), model(m) {
    // Every hit in the window is a scatterer: the acoustic model, not the
    // optical detector, decides how bright.
    params.detectorThreshold = 0.f;
    params.maxReturns = 1;
    params.samplesPerBeam = 1;
    // Keep the renderer's visual fog out of the trace: a LIDAR medium that
    // lives entirely below every ray (the tracer then never falls back to
    // the scene's fog). The sonar's own water column is its Sv.
    params.mediumExtinction = 1e-5f;
    params.mediumSurfaceY = -1e20f;
    params.mediumAlbedo = 0.f;
    rebuild();
}

ImagingSonar::~ImagingSonar() { waitFold(); }

void ImagingSonar::waitFold() {
    if (fold_.valid()) fold_.wait();
}

namespace {

    // The fields ImagingSonarSynthesizer bakes into its ray table and patterns.
    bool sameGeometry(const ImagingSonarModel& a, const ImagingSonarModel& b) {
        return a.horizontalFov == b.horizontalFov && a.beams == b.beams &&
               a.horizontalBeamwidth == b.horizontalBeamwidth && a.horizontalTaper == b.horizontalTaper &&
               a.verticalBeamwidth == b.verticalBeamwidth && a.azimuthOversample == b.azimuthOversample &&
               a.verticalSamples == b.verticalSamples && a.patternSpanBeamwidths == b.patternSpanBeamwidths &&
               a.seed == b.seed;
    }

}// namespace

void ImagingSonar::rebuild() {
    waitFold();// a running fold reads the synthesizer
    synth_ = std::make_unique<ImagingSonarSynthesizer>(model);
    built_ = model;
    syncParams();
}

unsigned ImagingSonar::rayCount() const { return synth_->rayCount(); }

void ImagingSonar::syncParams() {
    const echo::RangeResponse resp(model.rangeResolution, model.pulse);
    params.minRange = std::max(0.001f, model.housingRange);// the blind zone is the synthesizer's
    params.maxRange = model.maxRange + static_cast<float>(resp.halfSpan());
}

void ImagingSonar::resetNoise() {
    VisionSensor::resetNoise();
    pings_ = 0;
}

void ImagingSonar::ping(VulkanRenderer& renderer, SonarFrame& out) {
    pingBegin(renderer);
    if (!pingCollect(renderer, out)) out = SonarFrame{};
}

void ImagingSonar::pingBegin(VulkanRenderer& renderer) {
    // New fan geometry re-bakes the ray table before it is fired.
    if (!sameGeometry(model, built_)) rebuild();
    beginScan();
    syncParams();
    handle_ = VulkanRenderer::kNoLidarScan;

    if (!parent) updateMatrixWorld();
    getWorldPosition(origin_);
    // The rotation alone: the synthesizer inverts it by transposing, which a
    // scaled parent (a centimetre glTF root) would turn into a scale of u.
    rotation_.setFromMatrix4(Matrix4().extractRotation(*matrixWorld));

    const auto& dirs = synth_->rayDirections();
    beams_.resize(dirs.size());
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        Vector3 d = dirs[i];
        d.applyMatrix3(rotation_).normalize();
        beams_[i].origin = origin_;
        beams_[i].direction = d;
    }
    pingKey_ = echo::combineKeys(model.seed ^ 0x9A11ULL, pings_++);
    handle_ = renderer.scanLidarBegin(beams_, params);
}

bool ImagingSonar::pingReady(const VulkanRenderer& renderer) const {
    return handle_ != VulkanRenderer::kNoLidarScan && renderer.scanLidarReady(handle_);
}

bool ImagingSonar::pingCollectAsync(VulkanRenderer& renderer) {
    if (handle_ == VulkanRenderer::kNoLidarScan || asyncPending_) return false;
    const int h = handle_;
    handle_ = VulkanRenderer::kNoLidarScan;
    returns_.clear();
    if (!renderer.scanLidarCollect(h, returns_)) return false;
    asyncReturns_ = returns_;
    asyncPending_ = true;
    fold_ = std::async(std::launch::async, [this, m = model, mats = materials, o = origin_, rot = rotation_,
                                            key = pingKey_, t = lastScanTime(), n = threads] {
        synth_->synthesize(m, mats, asyncReturns_, o, rot, key, asyncFrame_, n);
        asyncFrame_.time = t;
    });
    return true;
}

bool ImagingSonar::frameReady() const {
    return asyncPending_ && fold_.valid() &&
           fold_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

bool ImagingSonar::takeFrame(SonarFrame& out) {
    if (!frameReady()) return false;
    fold_.get();
    out = std::move(asyncFrame_);
    asyncFrame_ = SonarFrame{};
    asyncPending_ = false;
    return true;
}

bool ImagingSonar::pingCollect(VulkanRenderer& renderer, SonarFrame& out) {
    returns_.clear();
    if (handle_ == VulkanRenderer::kNoLidarScan) return false;
    const int h = handle_;
    handle_ = VulkanRenderer::kNoLidarScan;
    if (!renderer.scanLidarCollect(h, returns_)) return false;
    synth_->synthesize(model, materials, returns_, origin_, rotation_, pingKey_, out, threads);
    out.time = lastScanTime();
    return true;
}
