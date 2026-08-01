// Conveyor path overlay: the centerline an authored conveyor actually follows —
// and, in the same pass, the generated content (belt, rollers, cleats, wall,
// frame) it produces. The conveyor twin of SplineOverlay.cpp, and the same
// rules apply: the overlay is editor furniture (never saved, never picked),
// the generated meshes are real document nodes, and everything is DERIVED
// STATE — the undoable step is the config or waypoint edit, and this pass
// follows the config wherever undo/redo/load leaves it.
//
// Rebuilt only when something it depends on moved: a hash over the waypoint
// count, their local positions, each waypoint's own config (corner radius /
// segment surface live on the waypoint node, not the owner) and the encoded
// conveyor config.
//
// Unlike a spline's single tube, regeneration here is WHOLESALE — the part
// count varies with the path — so the sync delegates to
// ConveyorConfig::syncDerived, which preserves the tagged group node and
// replaces its children. That keeps the library, the editor and any headless
// consumer generating identical content.
//
// Alongside the centerline each conveyor gets a DESIGN-AID buffer: chevrons
// along the path saying which way the belt runs (reverse included), and — for
// a selected rounded corner — the derived fillet centre and its two tangent
// spokes, so the bend the radius produces is visible while it is being tuned.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cmath>

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
    template<class L>
    void writePoints(L& line, int& capacity, const std::vector<Vector3>& points) {

        const auto count = static_cast<int>(points.size());
        if (count > capacity) {
            const auto old = line.geometry();
            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(
                                                       std::vector<float>(points.size() * 3), 3));
            line.setGeometry(geometry);
            if (old) old->dispose();
            capacity = count;
        }
        auto* position = line.geometry()->getAttribute<float>("position");
        for (int i = 0; i < count; ++i) {
            position->setXYZ(i, points[i].x, points[i].y, points[i].z);
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
            // Corner-radius / segment-surface flags live on the waypoint node.
            const auto wp = ConveyorWaypointConfig::read(*child).encode();
            hashBytes(seed, wp.data(), wp.size());
        }
        return seed;
    }

    // Travel-direction chevrons: a V every stretch of arc length, tip forward
    // along the flow (spec.reverse included), lifted a hair off the surface so
    // it never z-fights the belt. Emitted as segment PAIRS for a LineSegments.
    void appendChevrons(const conveyor::ConveyorSpec& spec,
                        const std::vector<Vector3>& sampled, std::vector<Vector3>& segments) {

        if (sampled.size() < 2) return;

        float total = 0.f;
        for (std::size_t i = 0; i + 1 < sampled.size(); ++i) {
            total += sampled[i].distanceTo(sampled[i + 1]);
        }
        if (total < 0.2f) return;

        const float pitch = std::clamp(total / 8.f, 0.6f, 2.5f);
        const Vector3 up(0, 1, 0);
        const float lift = 0.02f;
        const float size = std::clamp(spec.width * 0.22f, 0.08f, 0.3f);

        float next = pitch * 0.5f;
        float acc = 0.f;
        for (std::size_t i = 0; i + 1 < sampled.size() && next < total; ++i) {
            const float seg = sampled[i].distanceTo(sampled[i + 1]);
            if (seg < 1e-5f) continue;
            Vector3 dir;
            dir.subVectors(sampled[i + 1], sampled[i]).multiplyScalar(1.f / seg);
            for (; next < acc + seg; next += pitch) {
                Vector3 tip;
                tip.lerpVectors(sampled[i], sampled[i + 1], (next - acc) / seg);
                tip.addScaledVector(up, lift);
                Vector3 flow = dir;
                if (spec.reverse) flow.negate();
                Vector3 lat;
                if (std::abs(flow.dot(up)) > 0.999f) lat.set(0, 0, 1);
                else lat.crossVectors(flow, up).normalize();
                Vector3 head = tip;
                head.addScaledVector(flow, size);
                Vector3 left = tip;
                left.addScaledVector(flow, -size * 0.4f).addScaledVector(lat, size * 0.7f);
                Vector3 right = tip;
                right.addScaledVector(flow, -size * 0.4f).addScaledVector(lat, -size * 0.7f);
                segments.push_back(head);
                segments.push_back(left);
                segments.push_back(head);
                segments.push_back(right);
            }
            acc += seg;
        }
    }

    // The selected rounded corner's derived construction: a cross at the arc
    // centre and the two spokes to its tangent points — what the radius is
    // actually doing to the path, live while it is dragged.
    void appendCornerAid(const conveyor::CornerFillet& fillet, std::vector<Vector3>& segments) {

        if (!fillet.valid) return;

        const float cross = std::max(fillet.radius * 0.1f, 0.06f);
        for (const auto axis : {Vector3(1, 0, 0), Vector3(0, 0, 1)}) {
            Vector3 a = fillet.centre, b = fillet.centre;
            a.addScaledVector(axis, -cross);
            b.addScaledVector(axis, cross);
            segments.push_back(a);
            segments.push_back(b);
        }
        segments.push_back(fillet.centre);
        segments.push_back(fillet.t1);
        segments.push_back(fillet.centre);
        segments.push_back(fillet.t2);
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
            it->aids->removeFromParent();
            it = conveyorOverlays_.erase(it);
        } else {
            ++it;
        }
    }

    if (owners.empty()) return;

    const auto tint = theme::accent();

    for (auto* owner : owners) {

        auto it = std::find_if(conveyorOverlays_.begin(), conveyorOverlays_.end(),
                               [owner](const ConveyorOverlay& o) { return o.owner == owner; });

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
            // The aids share the line's material, so the chevrons and corner
            // helpers tint and fade exactly as the path does.
            auto aids = LineSegments::create(BufferGeometry::create(), material);
            aids->renderOrder = kCurveRenderOrder;
            aids->frustumCulled = false;
            aids->matrixAutoUpdate = false;
            conveyors_->add(aids);
            conveyorOverlays_.push_back(ConveyorOverlay{owner, line, material, 0, 0, aids});
            it = std::prev(conveyorOverlays_.end());
        }

        const auto config = ConveyorConfig::read(*owner).value_or(ConveyorConfig{});
        const auto hash = conveyorHash(*owner, config.encode());
        const auto spec = config.spec(*owner);

        if (hash != it->hash) {
            it->hash = hash;

            const auto sampled = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
            it->line->visible = sampled.size() >= 2;
            if (sampled.size() >= 2) writePoints(*it->line, it->capacity, sampled);

            config.syncDerived(*owner);
        } else if (!ConveyorConfig::derivedGroup(*owner)) {
            // The hash only covers what the content is BUILT from, so it says
            // nothing about whether the group is still there — a deleted
            // derived group lands here and comes back.
            config.syncDerived(*owner);
        }

        // --- design aids ----------------------------------------------------
        // Selection is deliberately not in the hash (it must not regenerate
        // parts), so the aids run on their own key: path hash + which of this
        // conveyor's waypoints is selected.
        auto* selected = selection_.get();
        const Object3D* selectedWaypoint =
                selected && ConveyorConfig::conveyorOf(*selected) == owner ? selected : nullptr;
        std::size_t aidsKey = hash;
        if (selectedWaypoint) {
            hashBytes(aidsKey, selectedWaypoint->uuid.data(), selectedWaypoint->uuid.size());
        }
        if (aidsKey != it->aidsKey) {
            it->aidsKey = aidsKey;
            std::vector<Vector3> segments;
            if (!spec.separator) {
                const auto sampled = conveyor::resamplePath(spec.waypoints, spec.smooth,
                                                            spec.samples);
                appendChevrons(spec, sampled, segments);
            }
            if (selectedWaypoint) {
                const auto index = ConveyorConfig::pointIndexOf(*owner, *selectedWaypoint);
                if (index < spec.waypoints.size()) {
                    appendCornerAid(conveyor::cornerFillet(spec.waypoints, index), segments);
                }
            }
            it->aids->visible = !segments.empty();
            if (!segments.empty()) writePoints(*it->aids, it->aidsCapacity, segments);
        }

        owner->updateMatrixWorld();
        it->line->matrix->copy(*owner->matrixWorld);
        it->line->matrixWorldNeedsUpdate = true;
        it->aids->matrix->copy(*owner->matrixWorld);
        it->aids->matrixWorldNeedsUpdate = true;

        // Tinted like a selected marker while the conveyor (or any of its
        // waypoints) is what the user is working on.
        const bool active = selected == owner || selectedWaypoint != nullptr;
        it->material->color.setRGB(tint.x, tint.y, tint.z);
        it->material->opacity = active ? 1.f : 0.55f;
        it->material->transparent = !active;
    }
}

void EditorApp::clearConveyorOverlays() {

    for (auto& overlay : conveyorOverlays_) {
        overlay.line->removeFromParent();
        overlay.aids->removeFromParent();
    }
    conveyorOverlays_.clear();
}
