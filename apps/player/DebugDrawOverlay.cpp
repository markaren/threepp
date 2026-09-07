
#include "DebugDrawOverlay.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/LineSegments.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace threepp;
using namespace threepp::player;

namespace {

    // The player draws nothing else, so this only has to beat the scene. Kept at
    // the editor's number so a script that tuned its overdraw against one front
    // end sees the same thing in the other.
    constexpr int kRenderOrder = 3200;

}// namespace


DebugDrawOverlay::DebugDrawOverlay(Group& parent)
    : parent_(parent) {}

DebugDrawOverlay::~DebugDrawOverlay() {

    clear();
}

void DebugDrawOverlay::setLogger(std::function<void(const std::string&)> logger) {

    logger_ = std::move(logger);
}

void DebugDrawOverlay::sync() {

#ifdef THREEPP_EDITOR_WITH_PYTHON

    auto& list = editor::scripting::debugDraw();

    // Inactive means no script session is running: not "nothing drawn this
    // frame" but "nobody there to draw". Take the node down with the session.
    if (!list.active) {
        clear();
        return;
    }

    if (list.dropped > 0 && !warned_) {
        warned_ = true;
        if (logger_) {
            logger_("debug draw capped at " + std::to_string(editor::scripting::DebugDrawList::cap) +
                    " segments - " + std::to_string(list.dropped) + " dropped this frame");
        }
    }

    const auto segmentCount = static_cast<int>(list.segments.size());
    const int vertices = segmentCount * 2;

    if (!lines_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        material->depthTest = false;
        material->depthWrite = false;
        lines_ = LineSegments::create(BufferGeometry::create(), material);
        lines_->renderOrder = kRenderOrder;
        lines_->frustumCulled = false;
        lines_->matrixAutoUpdate = false;
        capacity_ = 0;
        parent_.add(lines_);
    }

    if (vertices > capacity_) {
        const auto old = lines_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>(vertices * 3), 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(
                                                std::vector<float>(vertices * 3), 3));
        lines_->setGeometry(geometry);
        if (old) old->dispose();
        capacity_ = vertices;
    }

    // Empty is authoritative: the scripts DID run this frame and drew nothing.
    // Hide rather than hold the last picture.
    lines_->visible = vertices > 0;
    if (vertices == 0) {
        lines_->geometry()->drawRange = {0, 0};
        return;
    }

    auto* position = lines_->geometry()->getAttribute<float>("position");
    auto* color = lines_->geometry()->getAttribute<float>("color");
    if (!position || !color) {
        lines_->visible = false;
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
    lines_->geometry()->drawRange = {0, vertices};

    // Drained: next frame starts from nothing, which is what makes a line that
    // stopped being drawn actually disappear — and what keeps the list off its
    // cap over a long run.
    list.clear();

#else
    clear();
#endif
}

void DebugDrawOverlay::clear() {

    warned_ = false;
    if (!lines_) return;

    lines_->removeFromParent();
    // Undisposed, the orphan both leaks its GPU buffers and re-arms the
    // recycled-pointer trap described in the header.
    if (const auto geometry = lines_->geometry()) geometry->dispose();
    lines_.reset();
    capacity_ = 0;
}
