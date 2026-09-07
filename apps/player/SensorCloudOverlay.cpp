
#include "SensorCloudOverlay.hpp"

#include "threepp/extras/editor/SensorPlaySession.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Points.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::player;

namespace {

    // The editor's band for measurement overlays, kept so a scene reads the
    // same in both front ends.
    constexpr int kCloudRenderOrder = 3000;

    // PIXELS (sizeAttenuation off), for the reason the editor's overlay
    // documents at length: a world-space size makes the far half of every scan
    // sub-pixel, and the returns out at the range limit are exactly the ones
    // worth looking at.
    constexpr float kPointSize = 4.f;

}// namespace


SensorCloudOverlay::SensorCloudOverlay(Group& parent, editor::SensorPlaySession& session)
    : parent_(parent), session_(session) {}

SensorCloudOverlay::~SensorCloudOverlay() {

    clear();
}

void SensorCloudOverlay::sync() {

    // entries() is empty between episodes: the session drops them in stop().
    int points = 0;
    for (const auto& entry : session_.entries()) {
        points += static_cast<int>(entry->pointCount());
    }
    if (points == 0) {
        // Keep the last cloud rather than blinking — a sensor at 10 Hz would
        // otherwise strobe the overlay at its own rate.
        if (cloud_) cloud_->visible = cloud_->geometry()->drawRange.count > 0;
        return;
    }

    if (!cloud_) {
        auto material = PointsMaterial::create(PointsMaterial::Params()
                                                       .size(kPointSize)
                                                       .sizeAttenuation(false)
                                                       .vertexColors(true));
        // Range colour is data, not shading: tone mapping would compress the
        // far end of the ramp into the near end.
        material->toneMapped = false;
        cloud_ = Points::create(BufferGeometry::create(), material);
        cloud_->renderOrder = kCloudRenderOrder;
        cloud_->frustumCulled = false;
        cloud_->matrixAutoUpdate = false;
        capacity_ = 0;
        parent_.add(cloud_);
    }

    if (points > capacity_) {
        const auto old = cloud_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>(points * 3), 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(
                                                std::vector<float>(points * 3), 3));
        geometry->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        geometry->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
        cloud_->setGeometry(geometry);
        if (old) old->dispose();
        capacity_ = points;
    }

    auto* position = cloud_->geometry()->getAttribute<float>("position");
    auto* color = cloud_->geometry()->getAttribute<float>("color");
    if (!position || !color) {
        cloud_->visible = false;
        return;
    }
    cloud_->visible = true;

    Color tint;
    int written = 0;
    for (const auto& entry : session_.entries()) {

        // Range normalised per sensor: a 5 m depth camera and a 100 m LIDAR in
        // one scene must not share one hue ramp.
        const float maxRange = std::max(entry->config.farPlane, 1e-3f);

        if (entry->lidar) {
            for (const auto& r : entry->returns) {
                if (written >= capacity_) break;
                position->setXYZ(written, r.position.x, r.position.y, r.position.z);
                // Green near, red far — the same ramp the editor and the lidar
                // example use, so one cloud reads the same everywhere.
                tint.setHSL(0.33f * (1.f - std::min(r.distance / maxRange, 1.f)), 1.f, 0.5f);
                color->setXYZ(written, tint.r, tint.g, tint.b);
                ++written;
            }
        } else if (entry->depth) {
            Vector3 origin;
            entry->depth->getWorldPosition(origin);
            for (const auto& p : entry->cloud) {
                if (written >= capacity_) break;
                position->setXYZ(written, p.x, p.y, p.z);
                const float range = p.distanceTo(origin);
                tint.setHSL(0.33f * (1.f - std::min(range / maxRange, 1.f)), 1.f, 0.5f);
                color->setXYZ(written, tint.r, tint.g, tint.b);
                ++written;
            }
        }
    }

    position->needsUpdate();
    color->needsUpdate();
    // The tail beyond `written` still holds whatever a busier frame left there.
    cloud_->geometry()->drawRange = {0, written};
}

void SensorCloudOverlay::clear() {

    if (!cloud_) return;

    cloud_->removeFromParent();
    // The renderer keys GPU buffers on geometry identity; an undisposed orphan
    // both leaks them and re-arms the recycled-pointer staleness the in-place
    // writes above exist to avoid.
    if (const auto geometry = cloud_->geometry()) geometry->dispose();
    cloud_.reset();
    capacity_ = 0;
}
