// Procedural tree regeneration: the trunk and foliage an authored TreeConfig
// actually describes.
//
// The tree twin of SplineOverlay's derived-mesh half — and only that half. A
// tree has no editor-only picture to draw: the generated meshes ARE what the
// config says, so there is nothing left for an overlay line to add. Everything
// this pass touches is real document content, tagged `userData["treeDerived"]`,
// which is what lets a saved scene render and collide with no editor present.
//
// DERIVED STATE, NOT A COMMAND, for SplineOverlay's reason: the undoable step is
// the config edit, and this pass follows the config wherever undo, redo and load
// leave it. Undoing "Crown Radius" therefore regrows the tree on the next sync
// rather than through an undo entry carrying a megabyte of vertices.
//
// TWO CLOCKS, because the two halves of a tree cost three orders of magnitude
// apart (measured, RelWithDebInfo, on the four presets):
//
//     geometry (skeleton + both skins)     1.4 - 4.6 ms
//     procedural atlases (bark + leaf)      34 - 77 ms
//
// So geometry is rebuilt the moment the config moves — a dragged crown radius
// regrows the tree live, which is the only way to author one. The atlases are a
// function of a handful of params (TreeConfig::textureKey) and are redrawn only
// once the user LETS GO, since a live ColorEdit3 drag would otherwise spend 77 ms
// a frame redrawing a texture nobody has finished choosing. The two are tracked
// separately, so a deferred atlas is not lost by a geometry rebuild that lands
// first.
//
// A tree this pass has not seen before is ADOPTED, not rebuilt: a loaded
// document — and the factory's own freshly built tree — already carries meshes
// generated from the config it also carries, and regrowing every tree in a
// forest on open would cost seconds and buy nothing.

#include "EditorApp.hpp"

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/TreeConfig.hpp"
#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr std::array kParts{TreeConfig::Part::Trunk, TreeConfig::Part::Leaves};

    // Exactly one tagged child per part, ever. Extras (a duplicated tree, a
    // hand-edited document) are removed rather than tolerated. Gathered up
    // front because removeFromParent() rewrites the vector being walked.
    void pruneDuplicateParts(Object3D& tree) {

        std::vector<Object3D*> keep;
        for (const auto part : kParts) {
            if (auto* found = TreeConfig::derivedPart(tree, part)) keep.push_back(found);
        }

        std::vector<Object3D*> tagged;
        for (auto* child : tree.children) {
            if (TreeConfig::isDerived(*child)) tagged.push_back(child);
        }
        for (auto* child : tagged) {
            if (std::find(keep.begin(), keep.end(), child) == keep.end()) child->removeFromParent();
        }
    }

    // Brings one half of the tree in line with `geometry`.
    //
    // Regeneration preserves the NODE: same Object3D, same uuid, same material,
    // same userData, because the user may have put physics or a texture on it —
    // only the geometry is swapped, and the orphaned one disposed, since the
    // renderer keys GPU buffers on geometry identity and an undisposed one both
    // leaks them and re-arms the recycled-pointer staleness SplineOverlay's
    // writeSamples() was fixed for.
    void syncPart(Object3D& tree, const TreeConfig& config, TreeConfig::Part part,
                  const std::shared_ptr<BufferGeometry>& geometry) {

        if (!geometry) return;

        if (auto* existing = TreeConfig::derivedPart(tree, part)) {
            if (auto* mesh = existing->as<Mesh>()) {
                const auto old = mesh->geometry();
                mesh->setGeometry(geometry);
                if (old && old != geometry) old->dispose();
                return;
            }
            // Tagged but not a mesh: a hand-edited document. Replaced rather
            // than worked around.
            existing->removeFromParent();
        }

        auto material = part == TreeConfig::Part::Trunk ? config.makeBarkMaterial()
                                                        : config.makeLeafMaterial();
        auto mesh = Mesh::create(geometry, material);
        mesh->name = TreeConfig::label(part);
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        TreeConfig::markDerived(*mesh, part);
        tree.add(mesh);
    }

    Material* partMaterial(const Object3D& tree, TreeConfig::Part part) {

        auto* node = TreeConfig::derivedPart(tree, part);
        auto* mesh = node ? node->as<Mesh>() : nullptr;
        return mesh ? mesh->material().get() : nullptr;
    }

}// namespace


void EditorApp::syncTreeOverlays() {

    // --- who is a tree this frame ------------------------------------------
    static thread_local std::vector<Object3D*> owners;
    owners.clear();
    document_.scene().traverse([this](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        if (TreeConfig::isTree(object)) owners.push_back(&object);
    });

    // --- retire records whose tree is gone ---------------------------------
    for (auto it = treeOverlays_.begin(); it != treeOverlays_.end();) {
        if (std::find(owners.begin(), owners.end(), it->owner) == owners.end()) {
            it = treeOverlays_.erase(it);
        } else {
            ++it;
        }
    }

    if (owners.empty()) return;

    // A widget is mid-drag somewhere in the UI. Geometry does not care; the
    // atlases wait, because redrawing one costs more than the frame does.
    const bool settled = !ImGui::IsAnyItemActive();

    for (auto* owner : owners) {

        auto it = std::find_if(treeOverlays_.begin(), treeOverlays_.end(),
                               [owner](const TreeOverlay& o) { return o.owner == owner; });

        // The raw entry rather than a decoded config: this runs every frame for
        // every tree in the scene, and decoding forty-odd fields to discover
        // that nothing moved is the expensive way to learn it.
        const auto encoded = codec::readString(*owner, TreeConfig::userDataKey);

        const auto complete = [owner] {
            return TreeConfig::derivedPart(*owner, TreeConfig::Part::Trunk) &&
                   TreeConfig::derivedPart(*owner, TreeConfig::Part::Leaves);
        };

        if (it == treeOverlays_.end()) {
            treeOverlays_.push_back(TreeOverlay{owner});
            it = std::prev(treeOverlays_.end());

            // Adoption: a document (or the factory) that already carries both
            // halves carries ones generated from the config it also carries.
            // Trust that, and this frame costs nothing.
            if (complete()) {
                it->encoded = encoded;
                it->textureKey = TreeConfig::decode(encoded).value_or(TreeConfig{}).textureKey();
                it->wantedTextureKey = it->textureKey;
                continue;
            }
        }

        // Decoded only where it is about to be used — every frame for every
        // tree, discovering through forty-odd fields that nothing moved is the
        // expensive way to learn it.
        const bool changed = encoded != it->encoded;
        if (changed || !complete()) {
            const auto config = TreeConfig::decode(encoded).value_or(TreeConfig{});
            if (changed) {
                it->encoded = encoded;
                it->wantedTextureKey = config.textureKey();
            }
            pruneDuplicateParts(*owner);
            const auto geometries = config.build();
            syncPart(*owner, config, TreeConfig::Part::Trunk, geometries.trunk);
            syncPart(*owner, config, TreeConfig::Part::Leaves, geometries.leaves);
        }

        // The slow clock. Tracked apart from the entry above so a geometry
        // rebuild landing mid-drag does not mark the atlases done.
        if (settled && it->wantedTextureKey != it->textureKey) {
            TreeConfig::applyTextures(partMaterial(*owner, TreeConfig::Part::Trunk),
                                      partMaterial(*owner, TreeConfig::Part::Leaves),
                                      TreeConfig::decode(it->encoded).value_or(TreeConfig{}));
            it->textureKey = it->wantedTextureKey;
        }
    }
}

void EditorApp::clearTreeOverlays() {

    // No editor-side nodes to retire — the records are pure bookkeeping over
    // document content. Dropped so no owner pointer outlives the graph.
    treeOverlays_.clear();
}
