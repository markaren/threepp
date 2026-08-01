// Conveyor path overlay: the centerline an authored conveyor actually follows —
// and, in the same pass, the generated content (belt, rollers, cleats, wall,
// frame) it produces. The conveyor twin of SplineOverlay.cpp, and the same
// rules apply: the overlay Line is editor furniture (never saved, never
// picked), the generated meshes are real document nodes, and everything is
// DERIVED STATE — the undoable step is the config or waypoint edit, and this
// pass follows the config wherever undo/redo/load leaves it.
//
// Rebuilt only when something it depends on moved: a hash over the waypoint
// count, their local positions, each waypoint's own config (arc centre /
// segment surface live on the waypoint node, not the owner) and the encoded
// conveyor config.
//
// Unlike a spline's single tube, regeneration here is WHOLESALE — the part
// count varies with the path — so the sync delegates to
// ConveyorConfig::syncDerived, which preserves the tagged group node and
// replaces its children. That keeps the library, the editor and any headless
// consumer generating identical content.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Drawn over scene geometry but under the marker icons, which are UI.
    constexpr int kCurveRenderOrder = 3000;

    void hashBytes(std::size_t& seed, const void* data, std::size_t size) {

        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            seed ^= static_cast<std::size_t>(bytes[i]) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }
    }

    // In-place sample writes, same contract as SplineOverlay's writeSamples
    // (the renderer caches GPU buffers by attribute identity; replacing the
    // attribute each rebuild can hand it a recycled pointer that reads as
    // already uploaded). The attribute is only replaced when outgrown.
    void writeSamples(Line& line, int& capacity, const std::vector<Vector3>& sampled) {

        const auto count = static_cast<int>(sampled.size());
        if (count > capacity) {
            const auto old = line.geometry();
            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(
                                                       std::vector<float>(sampled.size() * 3), 3));
            line.setGeometry(geometry);
            if (old) old->dispose();
            capacity = count;
        }
        auto* position = line.geometry()->getAttribute<float>("position");
        for (int i = 0; i < count; ++i) {
            position->setXYZ(i, sampled[i].x, sampled[i].y, sampled[i].z);
        }
        position->needsUpdate();
        line.geometry()->drawRange = {0, count};
    }

    // Everything the sampled line and the generated content are a function of.
    // Waypoints only — the generated group is an OUTPUT of this hash.
    std::size_t conveyorHash(const Object3D& conveyor, const std::string& encoded) {

        std::size_t points = 0;
        for (const auto* child : conveyor.children) {
            if (!ConveyorConfig::isDerived(*child)) ++points;
        }

        std::size_t seed = points;
        hashBytes(seed, encoded.data(), encoded.size());
        for (const auto* child : conveyor.children) {
            if (ConveyorConfig::isDerived(*child)) continue;
            const float xyz[3]{child->position.x, child->position.y, child->position.z};
            hashBytes(seed, xyz, sizeof(xyz));
            // Arc-centre / segment-surface flags live on the waypoint node.
            const auto wp = ConveyorWaypointConfig::read(*child).encode();
            hashBytes(seed, wp.data(), wp.size());
        }
        return seed;
    }

}// namespace


void EditorApp::syncConveyorOverlays() {

    if (!conveyors_) return;

    // --- who is a conveyor this frame ---------------------------------------
    static thread_local std::vector<Object3D*> owners;
    owners.clear();
    document_.scene().traverse([this](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        if (ConveyorConfig::isConveyor(object)) owners.push_back(&object);
    });

    // --- retire overlays whose conveyor is gone -----------------------------
    for (auto it = conveyorOverlays_.begin(); it != conveyorOverlays_.end();) {
        if (std::find(owners.begin(), owners.end(), it->owner) == owners.end()) {
            it->line->removeFromParent();
            it = conveyorOverlays_.erase(it);
        } else {
            ++it;
        }
    }

    if (owners.empty()) return;

    const auto tint = theme::accent();

    for (auto* owner : owners) {

        auto it = std::find_if(conveyorOverlays_.begin(), conveyorOverlays_.end(),
                               [owner](const SplineOverlay& o) { return o.owner == owner; });

        if (it == conveyorOverlays_.end()) {
            auto material = LineBasicMaterial::create(LineBasicMaterial::Params()
                                                              .color(Color(0xffffff))
                                                              .toneMapped(false));
            auto line = Line::create(BufferGeometry::create(), material);
            line->renderOrder = kCurveRenderOrder;
            line->frustumCulled = false;
            // The path is authored in the conveyor's space; adopt its world
            // matrix outright rather than reproducing it.
            line->matrixAutoUpdate = false;
            conveyors_->add(line);
            conveyorOverlays_.push_back(SplineOverlay{owner, line, material, 0});
            it = std::prev(conveyorOverlays_.end());
        }

        const auto config = ConveyorConfig::read(*owner).value_or(ConveyorConfig{});
        const auto hash = conveyorHash(*owner, config.encode());

        if (hash != it->hash) {
            it->hash = hash;

            const auto spec = config.spec(*owner);
            const auto sampled = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
            it->line->visible = sampled.size() >= 2;
            if (sampled.size() >= 2) writeSamples(*it->line, it->capacity, sampled);

            config.syncDerived(*owner);
        } else if (!ConveyorConfig::derivedGroup(*owner)) {
            // The hash only covers what the content is BUILT from, so it says
            // nothing about whether the group is still there — a deleted
            // derived group lands here and comes back.
            config.syncDerived(*owner);
        }

        owner->updateMatrixWorld();
        it->line->matrix->copy(*owner->matrixWorld);
        it->line->matrixWorldNeedsUpdate = true;

        // Tinted like a selected marker while the conveyor (or any of its
        // waypoints) is what the user is working on.
        auto* selected = selection_.get();
        const bool active = selected == owner ||
                            (selected && ConveyorConfig::conveyorOf(*selected) == owner);
        it->material->color.setRGB(tint.x, tint.y, tint.z);
        it->material->opacity = active ? 1.f : 0.55f;
        it->material->transparent = !active;
    }
}

void EditorApp::clearConveyorOverlays() {

    for (auto& overlay : conveyorOverlays_) overlay.line->removeFromParent();
    conveyorOverlays_.clear();
}
