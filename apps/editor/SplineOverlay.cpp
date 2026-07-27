// Spline curve overlay: the line an authored spline actually describes — and,
// in the same pass, the geometry it generates.
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
//
// The generated mesh (config.mesh) runs off the SAME hash, because it is a
// function of exactly the same inputs. Unlike the curve line it is a real
// document node — a child of the spline tagged userData["splineDerived"] — so a
// saved scene renders and collides with no editor present. It is DERIVED STATE,
// not a command: the undoable step is the config edit, and this pass follows the
// config wherever undo/redo/load leaves it. Undoing "mesh = tube" therefore
// removes the mesh on the next sync rather than through an undo entry of its
// own, which is what keeps the two from disagreeing about which exists.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RibbonGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cstring>

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

    // Writes the samples into the line IN PLACE, the way
    // examples/extras/curves/spline_editor.cpp always has: the renderer caches
    // GPU buffers by attribute identity, so replacing the attribute on every
    // rebuild (what setFromPoints does) can hand it a recycled pointer with the
    // same fresh version count — which reads as already uploaded, and the drawn
    // curve freezes at its first shape until something destroys the whole line.
    // The attribute is only replaced when the curve OUTGROWS it, and then the
    // whole geometry is swapped and the old one disposed, so the renderer
    // provably lets go of the stale buffers.
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
        // The tail beyond `count` still holds whatever a longer curve left there.
        line.geometry()->drawRange = {0, count};
    }

    // Everything the sampled line and the generated mesh are a function of.
    // Float bytes rather than values: a position that did not move hashes
    // identically, and one that did cannot collide with it however small the
    // move was.
    //
    // Control points only. The generated mesh is an OUTPUT of this hash, so
    // counting it as an input would make every regeneration dirty the hash that
    // triggered it and the pass would chase its own tail for a frame.
    std::size_t splineHash(const Object3D& spline, const std::string& encoded) {

        const auto points = SplineConfig::controlPointNodes(spline);

        std::size_t seed = points.size();
        hashBytes(seed, encoded.data(), encoded.size());
        for (const auto* child : points) {
            const float xyz[3]{child->position.x, child->position.y, child->position.z};
            hashBytes(seed, xyz, sizeof(xyz));
        }
        return seed;
    }

    // The geometry `config` asks for, or nullptr when there is no curve to
    // sweep along yet.
    std::shared_ptr<BufferGeometry> buildSplineGeometry(const Object3D& spline,
                                                        const SplineConfig& config) {

        auto curve = config.curve(spline);
        if (!curve) return nullptr;

        const auto divisions = config.divisions(spline);
        switch (config.mesh) {
            case SplineConfig::MeshKind::Tube:
                return TubeGeometry::create(
                        curve, TubeGeometry::Params(
                                       divisions, std::max(config.radius, 1e-3f),
                                       static_cast<unsigned int>(std::clamp(config.radialSegments, 3, 64)),
                                       config.closed));
            case SplineConfig::MeshKind::Road:
                return RibbonGeometry::create(
                        *curve, RibbonGeometry::Params(
                                        std::max(config.width, 1e-3f), divisions,
                                        std::max(config.uvLength, 1e-3f), config.closed));
            case SplineConfig::MeshKind::None:
                break;
        }
        return nullptr;
    }

    // Brings the spline's generated child in line with `config`. Called only
    // when the hash moved, or when what is actually parented disagrees with
    // what the config asks for — a loaded document already carries its mesh, and
    // ADOPTING that one rather than adding a second is the whole reason this
    // looks for the tag instead of remembering what it made.
    //
    // Regeneration preserves the NODE: same Object3D, same uuid, same material,
    // same userData, because the user may have put physics or a texture on it.
    // Only the geometry is swapped — and the orphaned one is disposed, since the
    // renderer keys GPU buffers on geometry identity and an undisposed one both
    // leaks them and re-arms the recycled-pointer staleness the overlay line
    // above was fixed for.
    void syncDerivedMesh(Object3D& spline, const SplineConfig& config) {

        // Exactly one tagged child, ever. Extras (a duplicate, a hand-edited
        // document) are removed rather than tolerated. Gathered up front
        // because removeFromParent() rewrites the vector being walked.
        std::vector<Object3D*> tagged;
        for (auto* child : spline.children) {
            if (SplineConfig::isDerived(*child)) tagged.push_back(child);
        }

        Mesh* derived = nullptr;
        if (config.mesh != SplineConfig::MeshKind::None) {
            for (auto* child : tagged) {
                if (auto* mesh = child->as<Mesh>()) {
                    derived = mesh;
                    break;
                }
            }
        }
        const Object3D* keep = derived;
        for (auto* child : tagged) {
            if (child != keep) child->removeFromParent();
        }

        if (config.mesh == SplineConfig::MeshKind::None) return;

        auto geometry = buildSplineGeometry(spline, config);
        if (!geometry) {
            // Fewer than two points: legal while authoring, and nothing to
            // sweep. The node keeps its last geometry (and everything the user
            // configured on it) and simply stops drawing until the curve is
            // back.
            if (derived) derived->visible = false;
            return;
        }

        if (!derived) {
            auto material = MeshStandardMaterial::create();
            auto mesh = Mesh::create(geometry, material);
            // Named once, at creation: switching tube to road preserves the
            // node, and clobbering a name the user typed is not a thing a sync
            // pass gets to do.
            mesh->name = ObjectFactory::uniqueName(spline, SplineConfig::label(config.mesh));
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            SplineConfig::markDerived(*mesh);
            spline.add(mesh);
            return;
        }

        derived->visible = true;
        const auto old = derived->geometry();
        derived->setGeometry(geometry);
        if (old && old != geometry) old->dispose();
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
            // In-place updates never refresh cached bounds, and a curve that
            // outgrew the bounds it was born with must not vanish at the edges.
            line->frustumCulled = false;
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
                sampled = curve->getPoints(config.divisions(*owner));
            }
            // A spline down to one point is legal while it is being authored;
            // it just has no curve to draw yet.
            it->line->visible = !sampled.empty();
            if (!sampled.empty()) writeSamples(*it->line, it->capacity, sampled);

            syncDerivedMesh(*owner, config);
        } else {
            // The hash only covers what the mesh is BUILT from, so it says
            // nothing about whether the mesh is still there. A loaded document,
            // a deleted derived child and a duplicated one all land here.
            const std::size_t wanted = config.mesh == SplineConfig::MeshKind::None ? 0 : 1;
            std::size_t derived = 0;
            for (const auto* child : owner->children) {
                if (SplineConfig::isDerived(*child)) ++derived;
            }
            if (derived != wanted) syncDerivedMesh(*owner, config);
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
