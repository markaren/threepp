
#include "threepp/extras/sensors/TracedRasterVisionSensor.hpp"

#include "sensor_scan_util.hpp"

#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Quaternion.hpp"

// The path-traced back-end is only *used* on a Vulkan build (there is no
// raster depth pass there), but its header is included unconditionally so the
// cached unique_ptr member has a complete type to destroy. The header itself
// is Vulkan-include-free; only the renderer header below is gated.
#include "threepp/helpers/PathTracedLidarSensor.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

using namespace threepp;

template<typename PointT>
TracedRasterVisionSensor<PointT>::TracedRasterVisionSensor(const RangeNoiseModel& noise, float near, float far)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    : VisionSensor(*this, noise), near_(near), far_(far) {}

template<typename PointT>
TracedRasterVisionSensor<PointT>::~TracedRasterVisionSensor() = default;

template<typename PointT>
void TracedRasterVisionSensor<PointT>::resetNoise() {
    VisionSensor::resetNoise();
    // On Vulkan the back-end draws the noise, so resetting only our own stream
    // would silently leave that path unreplayable.
    if (tracedBackend_) tracedBackend_->resetNoise();
}

template<typename PointT>
void TracedRasterVisionSensor<PointT>::scan(Renderer& renderer, Scene& scene, Cloud& cloud) {
    // Fire and take delivery in one call. Both halves run unconditionally: the
    // raster path has already filled `cloud`, and its collect is what clears
    // the "one delivery owed" flag so the pair stays balanced.
    const bool immediate = scanBegin(renderer, scene, cloud);
    const bool delivered = scanCollect(renderer, cloud);
    // Synchronous contract: this call's cloud, or nothing.
    if (!immediate && !delivered) cloud.clear();
}

template<typename PointT>
bool TracedRasterVisionSensor<PointT>::scanBegin(Renderer& renderer, Scene& scene, Cloud& cloud) {
    beginScan();
    scanPending_ = true;

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        // No raster depth to read back here: trace the same beam pattern
        // through the renderer's TLAS instead. Range noise is applied by the
        // back-end.
        auto& lidar = aimedTracedBackend();
        lidar.scanBegin(*vk);
        // Refused (too many scans already in flight): nothing is owed, so the
        // caller keeps whatever cloud it had rather than being handed an empty
        // one, and tries again next frame.
        scanPending_ = lidar.scanFired();
        // `cloud` is deliberately NOT cleared. It still holds the last
        // delivered scan, and a viewer reading it between fire and delivery
        // must see that rather than an empty one — clearing here made the
        // overlay blink and made "the cloud is not empty" a coin flip on the
        // frame you asked.
        return false;
    }
#endif

    // Raster: the depth render(s) and their readbacks ARE the scan, and they
    // block anyway. Do it here and let scanCollect hand the same cloud over.
    sensorscan::DataPassGuard guard(renderer);
    rasterScan(renderer, scene, cloud);
    return true;
}

template<typename PointT>
bool TracedRasterVisionSensor<PointT>::scanReady(const Renderer& renderer) const {
    if (!scanPending_) return false;
#ifdef THREEPP_WITH_VULKAN
    if (const auto* vk = dynamic_cast<const VulkanRenderer*>(&renderer)) {
        return tracedBackend_ && tracedBackend_->scanReady(*vk);
    }
#else
    (void) renderer;
#endif
    return true;// raster: filled by scanBegin
}

template<typename PointT>
bool TracedRasterVisionSensor<PointT>::scanCollect(Renderer& renderer, Cloud& cloud) {
    if (!scanPending_) return false;

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        scanPending_ = false;
        if (!tracedBackend_) return false;
        return collectTraced(*tracedBackend_, *vk, cloud);
    }
#else
    (void) renderer;
    (void) cloud;
#endif

    scanPending_ = false;
    return true;// raster: scanBegin already filled it
}

// Defined unconditionally (the explicit instantiations below need every
// member), but only ever CALLED from the Vulkan-gated branches above — on a
// raster-only build createTracedBackend() returns nullptr and this would
// dereference it. Nothing here touches a Vulkan-gated symbol: only data
// members and this class's own virtual hook.
template<typename PointT>
PathTracedLidarSensor& TracedRasterVisionSensor<PointT>::tracedBackend() {
    if (!tracedBackend_) tracedBackend_ = createTracedBackend();
    // This sensor owns the noise contract; the back-end just applies it, so
    // its model tracks ours (including a seed the caller re-rolled). Copying
    // an unchanged seed does not restart the stream — see VisionSensor.
    tracedBackend_->rangeNoise = rangeNoise;
    // The range shell, handed to the tracer as its [tMin, tMax] interval. This
    // is the SAME bound the raster path applies per pixel after unprojecting
    // (see shellRange) — near is a blind sphere on both backends, not a plane.
    // Without it every beam returns the sensor's own housing from the inside.
    tracedBackend_->params.minRange = near_;
    return *tracedBackend_;
}

template<typename PointT>
PathTracedLidarSensor& TracedRasterVisionSensor<PointT>::aimedTracedBackend() {
    if (!parent) updateMatrixWorld();

    auto& lidar = tracedBackend();
    Vector3 pos, scl;
    Quaternion quat;
    matrixWorld->decompose(pos, quat, scl);
    lidar.position = pos;
    lidar.quaternion = quat;
    lidar.scale = scl;
    return lidar;
}

namespace threepp {
    template class TracedRasterVisionSensor<Vector3>;
    template class TracedRasterVisionSensor<LidarReturn>;
}// namespace threepp
