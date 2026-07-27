// Spline curve overlay: the line an authored spline actually describes.
//
// A spline is a Group whose children are its control points (see SplineConfig),
// so the scene graph alone draws nothing — the points are empty Object3Ds and
// the curve between them exists only as a sampling of CatmullRomCurve3. This
// file samples it into one Line per spline, parented to the editor overlay, so
// it is never saved, never picked, and never appears in the camera preview.
//
// Rebuilt only when something it depends on moved: a hash over the point count,
// their local positions and the encoded config. Control points number in the
// tens, so hashing every frame is cheaper than tracking dirty flags through the
// gizmo, the command stack and the play snapshot.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cstring>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Drawn over scene geometry but under the marker icons, which are UI.
    constexpr int kCurveRenderOrder = 3000;

    // A curve with more samples than this per segment costs more than it shows.
    constexpr int kMaxSamples = 200;

    void hashBytes(std::size_t& seed, const void* data, std::size_t size) {

        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            seed ^= static_cast<std::size_t>(bytes[i]) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }
    }

    // Everything the sampled line is a function of. Float bytes rather than
    // values: a position that did not move hashes identically, and one that did
    // cannot collide with it however small the move was.
    std::size_t splineHash(const Object3D& spline, const std::string& encoded) {

        std::size_t seed = spline.children.size();
        hashBytes(seed, encoded.data(), encoded.size());
        for (const auto* child : spline.children) {
            const float xyz[3]{child->position.x, child->position.y, child->position.z};
            hashBytes(seed, xyz, sizeof(xyz));
        }
        return seed;
    }

}// namespace


void EditorApp::syncSplineOverlays() {

    if (!splines_) return;

    // --- who needs a curve this frame --------------------------------------
    static thread_local std::vector<Object3D*> owners;
    owners.clear();
    document_.scene().traverse([this](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        if (SplineConfig::isSpline(object)) owners.push_back(&object);
    });

    // --- retire curves whose spline is gone --------------------------------
    for (auto it = splineOverlays_.begin(); it != splineOverlays_.end();) {
        if (std::find(owners.begin(), owners.end(), it->owner) == owners.end()) {
            it->line->removeFromParent();
            it = splineOverlays_.erase(it);
        } else {
            ++it;
        }
    }

    if (owners.empty()) return;

    const auto tint = theme::accent();

    for (auto* owner : owners) {

        auto it = std::find_if(splineOverlays_.begin(), splineOverlays_.end(),
                               [owner](const SplineOverlay& o) { return o.owner == owner; });

        if (it == splineOverlays_.end()) {
            auto material = LineBasicMaterial::create(LineBasicMaterial::Params()
                                                              .color(Color(0xffffff))
                                                              .toneMapped(false));
            auto line = Line::create(BufferGeometry::create(), material);
            line->renderOrder = kCurveRenderOrder;
            // The curve is authored in the spline's space; the overlay is not
            // under it, so the spline's world matrix is adopted outright rather
            // than reproduced as position/rotation/scale.
            line->matrixAutoUpdate = false;
            splines_->add(line);
            splineOverlays_.push_back(SplineOverlay{owner, line, material, 0});
            it = std::prev(splineOverlays_.end());
        }

        const auto config = SplineConfig::read(*owner).value_or(SplineConfig{});
        const auto hash = splineHash(*owner, config.encode());

        if (hash != it->hash) {
            it->hash = hash;

            std::vector<Vector3> sampled;
            if (auto curve = config.curve(*owner)) {
                // samples-per-segment: the count the user authored is per
                // segment, and getPoints() divides the whole curve.
                const auto segments = static_cast<int>(owner->children.size()) - (config.closed ? 0 : 1);
                const int divisions = std::clamp(config.samples, 1, kMaxSamples) * std::max(segments, 1);
                sampled = curve->getPoints(static_cast<unsigned int>(divisions));
            }
            // A spline down to one point is legal while it is being authored;
            // it just has no curve to draw yet.
            it->line->visible = !sampled.empty();
            if (!sampled.empty()) it->line->geometry()->setFromPoints(sampled);
        }

        owner->updateMatrixWorld();
        it->line->matrix->copy(*owner->matrixWorld);
        it->line->matrixWorldNeedsUpdate = true;

        // Tinted like a selected marker while the spline (or any of its points)
        // is what the user is working on, so a scene of several splines still
        // says which one the inspector is editing.
        auto* selected = selection_.get();
        const bool active = selected == owner ||
                            (selected && SplineConfig::splineOf(*selected) == owner);
        it->material->color.setRGB(tint.x, tint.y, tint.z);
        it->material->opacity = active ? 1.f : 0.55f;
        it->material->transparent = !active;
    }
}

void EditorApp::clearSplineOverlays() {

    for (auto& overlay : splineOverlays_) overlay.line->removeFromParent();
    splineOverlays_.clear();
}
