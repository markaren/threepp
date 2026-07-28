// Sensor point cloud: what the depth cameras and LIDARs are actually seeing.
//
// A sensor is otherwise entirely invisible. The inspector shows what was
// AUTHORED — rate, noise, beam pattern — and the Sensors tab shows counters, but
// neither answers the only question that matters when a scan looks wrong: where
// did the beams land. A sensor aimed a few degrees off, mounted inside its own
// robot, or clipped by its near plane produces a plausible-looking count and a
// useless cloud, and no number in a table says so.
//
// Every playing vision sensor's returns go into ONE Points under the editor
// overlay, coloured green-to-red by range. Same contract as the collider lines
// (PhysicsDebugOverlay.cpp) and for the same reason: the renderer caches GPU
// buffers by ATTRIBUTE IDENTITY, so a fresh attribute every frame can hand it a
// recycled pointer that reads as already uploaded and the overlay freezes. The
// attributes are rewritten in place, replaced only when the cloud outgrows them,
// and the orphaned geometry is disposed when they are.
//
// The cloud lives under the overlay, which SensorPlaySession hides for the
// duration of each scan — so the sensors never measure their own output. That is
// not a nicety: a lidar that sees last frame's points builds a shell around
// itself and the cloud collapses to arm's length within a second.

#include "EditorApp.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/SensorPlaySession.hpp"
#endif

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Points.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Over scene geometry, under the marker icons — the band the collider lines
    // and the spline curves are drawn in.
    constexpr int kCloudRenderOrder = 3000;

    // Big enough to read as a point at the ranges a robot sensor works at,
    // small enough not to paint over the geometry it is measuring.
    constexpr float kPointSize = 0.06f;

}// namespace


void EditorApp::syncSensorOverlay() {

#ifdef THREEPP_EDITOR_WITH_PHYSX

    // entries() is empty between plays: the session drops them in stop().
    const bool wanted = sensorCloudVisible_ && sensors_ && sensors_->sensorCount() > 0;
    if (!wanted) {
        clearSensorOverlay();
        return;
    }

    // Count first: the buffer is grown once for the whole rig rather than per
    // sensor, and a frame where nothing scanned keeps the last cloud (the same
    // choice the collider overlay makes, for the same "do not blink" reason).
    int points = 0;
    for (const auto& entry : sensors_->entries()) {
        points += static_cast<int>(entry->pointCount());
    }
    if (points == 0) {
        if (sensorCloud_) {
            sensorCloud_->visible = sensorCloud_->geometry()->drawRange.count > 0;
        }
        return;
    }

    if (!sensorCloud_) {
        auto material = PointsMaterial::create(PointsMaterial::Params()
                                                       .size(kPointSize)
                                                       .vertexColors(true));
        // Range colour is data, not shading: tone mapping would compress the far
        // end of the ramp into the near end.
        material->toneMapped = false;
        sensorCloud_ = Points::create(BufferGeometry::create(), material);
        sensorCloud_->renderOrder = kCloudRenderOrder;
        // In-place attribute writes never refresh cached bounds, and these are
        // world-space points that must not vanish at the edge of a stale one.
        sensorCloud_->frustumCulled = false;
        sensorCloud_->matrixAutoUpdate = false;
        sensorCloudCapacity_ = 0;
        overlay_->add(sensorCloud_);
    }

    if (points > sensorCloudCapacity_) {
        const auto old = sensorCloud_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>(points * 3), 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(
                                                std::vector<float>(points * 3), 3));
        geometry->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        geometry->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
        sensorCloud_->setGeometry(geometry);
        if (old) old->dispose();
        sensorCloudCapacity_ = points;
    }

    auto* position = sensorCloud_->geometry()->getAttribute<float>("position");
    auto* color = sensorCloud_->geometry()->getAttribute<float>("color");
    if (!position || !color) {
        sensorCloud_->visible = false;
        return;
    }
    sensorCloud_->visible = true;

    Color tint;
    int written = 0;
    for (const auto& entry : sensors_->entries()) {

        // Range is normalised per sensor: a 5 m depth camera and a 100 m LIDAR
        // in one scene would otherwise put the whole depth cloud in one hue.
        const float maxRange = std::max(entry->config.farPlane, 1e-3f);

        if (entry->lidar) {
            for (const auto& r : entry->returns) {
                if (written >= sensorCloudCapacity_) break;
                position->setXYZ(written, r.position.x, r.position.y, r.position.z);
                // Green near, red far — the ramp the lidar example uses, so a
                // cloud reads the same in the editor as in the demo.
                tint.setHSL(0.33f * (1.f - std::min(r.distance / maxRange, 1.f)), 1.f, 0.5f);
                color->setXYZ(written, tint.r, tint.g, tint.b);
                ++written;
            }
        } else if (entry->depth) {
            Vector3 origin;
            entry->depth->getWorldPosition(origin);
            for (const auto& p : entry->cloud) {
                if (written >= sensorCloudCapacity_) break;
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
    sensorCloud_->geometry()->drawRange = {0, written};

#else
    clearSensorOverlay();
#endif
}

void EditorApp::clearSensorOverlay() {

    if (!sensorCloud_) return;

    sensorCloud_->removeFromParent();
    // The renderer keys GPU buffers on geometry identity; an orphan that is never
    // disposed both leaks them and re-arms the recycled-pointer staleness the
    // in-place writes above exist to avoid.
    if (const auto geometry = sensorCloud_->geometry()) geometry->dispose();
    sensorCloud_.reset();
    sensorCloudCapacity_ = 0;
}
