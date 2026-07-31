// Script debug draw: the lines a playing script asked to see.
//
// Every controller script computes geometry nobody can look at — an altimeter
// ray, a contact normal, a drive target — and its only instrument was print().
// threepp.editor.draw_line / draw_ray / draw_point / draw_box / draw_sphere /
// draw_axes decompose to world-space segments in scripting::debugDraw()
// (Scripting.hpp), and this drains that list into ONE LineSegments under the
// editor overlay each rendered frame.
//
// Immediate mode: drained is gone, and a script that wants a line to persist
// draws it again next update() — which it is called every frame to do. A
// paused frame skips the drain, so the last picture stays up instead of
// blinking out the moment the scripts stop being asked.
//
// The geometry contract is PhysicsDebugOverlay's, for the same reason (the
// renderer caches GPU buffers by ATTRIBUTE IDENTITY, so a fresh attribute per
// frame can hand it a recycled pointer that reads as already uploaded): the
// attributes are rewritten in place, replaced only when outgrown, and the
// orphaned geometry is disposed when they are. Unlike the collider overlay this
// one carries a colour attribute too — each draw call picks its own.
//
// Under overlay_, deliberately: the overlay is hidden for the duration of
// every sensor scan, so a lidar can never range against somebody's debug
// arrow. Depth test off: debugging wants to see the ray inside the mesh.

#include "EditorApp.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Over the collider lines (3000): a script pointing at its own collider
    // should win the overdraw, since it is the one somebody is iterating on.
    constexpr int kDebugDrawRenderOrder = 3200;

}// namespace


void EditorApp::syncDebugDraw() {

#ifdef THREEPP_EDITOR_WITH_PYTHON

    auto& list = scripting::debugDraw();

    // Inactive means no session is running: not "nothing drawn this frame" but
    // "nobody there to draw". Take the node down with the session.
    if (!list.active) {
        clearDebugDraw();
        return;
    }

    // Paused: the scripts are not being asked, so the buffer is empty by now —
    // draining would blank the picture. Keep the last frame on screen; it is
    // exactly what was true when time stopped.
    if (play_.paused()) return;

    if (list.dropped > 0 && !debugDrawWarned_) {
        debugDrawWarned_ = true;
        log("debug draw capped at " + std::to_string(scripting::DebugDrawList::cap) +
            " segments - " + std::to_string(list.dropped) + " dropped this frame");
    }

    const auto segmentCount = static_cast<int>(list.segments.size());
    const int vertices = segmentCount * 2;

    if (!debugDrawLines_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        // On top of the scene: the whole point is seeing the ray that ends
        // inside a mesh. depthWrite off with it, so the lines cannot shadow
        // anything drawn after them either.
        material->depthTest = false;
        material->depthWrite = false;
        debugDrawLines_ = LineSegments::create(BufferGeometry::create(), material);
        debugDrawLines_->renderOrder = kDebugDrawRenderOrder;
        // In-place updates never refresh cached bounds, and the segments are
        // world-space; same rule as the collider overlay.
        debugDrawLines_->frustumCulled = false;
        debugDrawLines_->matrixAutoUpdate = false;
        debugDrawCapacity_ = 0;
        overlay_->add(debugDrawLines_);
    }

    if (vertices > debugDrawCapacity_) {
        const auto old = debugDrawLines_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>(vertices * 3), 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(
                                                std::vector<float>(vertices * 3), 3));
        debugDrawLines_->setGeometry(geometry);
        if (old) old->dispose();
        debugDrawCapacity_ = vertices;
    }

    // Empty is authoritative here, unlike the PhysX render buffer: the scripts
    // DID run this frame, and they drew nothing. Hide rather than hold.
    debugDrawLines_->visible = vertices > 0;
    if (vertices == 0) {
        debugDrawLines_->geometry()->drawRange = {0, 0};
        return;
    }

    auto* position = debugDrawLines_->geometry()->getAttribute<float>("position");
    auto* color = debugDrawLines_->geometry()->getAttribute<float>("color");
    if (!position || !color) {
        debugDrawLines_->visible = false;
        return;
    }

    for (int i = 0; i < segmentCount; ++i) {
        const auto& s = list.segments[static_cast<std::size_t>(i)];
        position->setXYZ(i * 2, s.ax, s.ay, s.az);
        position->setXYZ(i * 2 + 1, s.bx, s.by, s.bz);
        color->setXYZ(i * 2, s.r, s.g, s.b);
        color->setXYZ(i * 2 + 1, s.r, s.g, s.b);
    }
    position->needsUpdate();
    color->needsUpdate();
    debugDrawLines_->geometry()->drawRange = {0, vertices};

    // Drained: next frame starts from nothing, which is what makes a line that
    // stopped being drawn actually disappear.
    list.clear();

#else
    clearDebugDraw();
#endif
}

void EditorApp::clearDebugDraw() {

    debugDrawWarned_ = false;
    if (!debugDrawLines_) return;

    debugDrawLines_->removeFromParent();
    // Same leak-and-staleness rule as the collider overlay: the renderer keys
    // GPU buffers on geometry identity, so an undisposed orphan both leaks
    // them and re-arms the recycled-pointer trap.
    if (const auto geometry = debugDrawLines_->geometry()) geometry->dispose();
    debugDrawLines_.reset();
    debugDrawCapacity_ = 0;
}
