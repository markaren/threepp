// The baked scan surface, drawn over the scan while it is authored.
//
// "Collide and sense" is authored blind otherwise: the mesh only exists during
// Play, and a bake that captured the wrong thing (voxels too coarse, poses too
// few, half the scan carved away) looks exactly like a bake that captured the
// right thing until a robot falls through it. This is the picture.
//
// EDITOR CHROME, not document content. The previews hang off overlay_, which the
// document registers as editor-only: SceneDocument::isEditorOnly walks parents,
// so picking skips them (EditorApp::pickAt), export detaches them, and
// sensors_->setHiddenDuringScan(overlay_) keeps them out of every sensor scan.
// The toggle is a plain bool like physicsDebug_ — a view state, nothing the
// document or the undo stack knows about.
//
// THE MESH IS NEVER BAKED HERE. This runs every frame; a bake blocks for ~0.4 s
// and flashes the viewport through its pose loop. The sync only ASKS the memo
// (SplatSurfaceCache::find), so a key the memo no longer holds — the node moved,
// a knob changed — draws nothing at all rather than a surface that is no longer
// the one this scan would produce. Toggling the preview on is what bakes, once,
// through the same memo "Bake now" and Play use.
//
// Hidden for the duration of a Play: SplatSurfacePlaySession adds its own
// sensor-only twin of these very triangles, and two copies of one surface is not
// a preview.
//
// VULKAN ONLY, for the reason the bake is (the depth AOV it fuses is a Vulkan
// G-buffer attachment) — on any other backend the memo is always empty and the
// inspector says why.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/SplatSurfaceCache.hpp"
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"// kSplatUnoccludedOverlayLayer
#endif

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

using namespace threepp;
using namespace threepp::editor;


void EditorApp::syncSplatSurfacePreviews() {

    if (!splatSurfacePreview_ || isPlaying() || !splatSurfaces_ || !overlay_) {
        clearSplatSurfacePreviews();
        return;
    }

    static thread_local std::vector<std::string> live;
    live.clear();

    document_.scene().traverse([&](Object3D& object) {
        auto* cloud = object.as<SplatCloud>();
        if (!cloud) return;
        const auto config = SplatSurfaceConfig::read(object);
        if (!config || !config->enabled) return;

        const auto* mesh = splatSurfaces_->find(*cloud, *config);
        if (!mesh) return;

        // The memo's own key, so a re-bake under new knobs replaces the picture
        // instead of leaving the old triangles standing under a new config.
        const auto key = SplatSurfaceCache::keyFor(*cloud, *config);
        auto& preview = splatSurfacePreviews_[object.uuid];
        if (!preview.mesh || preview.key != key) {

            if (preview.mesh) {
                preview.mesh->removeFromParent();
                if (const auto geometry = preview.mesh->geometry()) geometry->dispose();
            }

            // LINE SEGMENTS, not a wireframe-flagged Mesh, and the reason is
            // measured rather than stylistic: a Mesh under overlay_ does not
            // draw on the Vulkan path (the node is visible, overlay-parented
            // and correctly placed, and an A/B of the same frame with and
            // without it differs by 435 pixels of ordinary temporal noise).
            // Every overlay this app already has — the grid, the PhysX debug
            // lines, the spline curves, the spawn slab — is LineSegments, and
            // that is the shape that renders.
            //
            // Edges are deduplicated: a closed marching-cubes surface shares
            // nearly every edge between two triangles, so the naive three per
            // triangle is twice the geometry for the same picture.
            std::vector<float> positions;
            positions.reserve(mesh->indices.size() * 3);
            std::unordered_set<std::uint64_t> seen;
            seen.reserve(mesh->indices.size());
            const auto vertexCount = mesh->positions.size() / 3;
            const auto edge = [&](std::uint32_t a, std::uint32_t b) {
                if (a >= vertexCount || b >= vertexCount) return;
                const std::uint64_t key = a < b
                                                  ? (static_cast<std::uint64_t>(a) << 32) | b
                                                  : (static_cast<std::uint64_t>(b) << 32) | a;
                if (!seen.insert(key).second) return;
                positions.insert(positions.end(),
                                 {mesh->positions[a * 3], mesh->positions[a * 3 + 1],
                                  mesh->positions[a * 3 + 2], mesh->positions[b * 3],
                                  mesh->positions[b * 3 + 1], mesh->positions[b * 3 + 2]});
            };
            for (std::size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
                const auto i0 = mesh->indices[i], i1 = mesh->indices[i + 1], i2 = mesh->indices[i + 2];
                edge(i0, i1);
                edge(i1, i2);
                edge(i2, i0);
            }

            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

            const auto tint = theme::accent();
            auto material = LineBasicMaterial::create();
            material->color.setRGB(tint.x, tint.y, tint.z);
            material->toneMapped = false;
            // Depth-tested, unlike the spawn slab: a wall in front of the scan
            // must hide this preview exactly as it hides the grid.
            material->depthTest = true;

            preview.triangles = mesh->triangleCount();
            preview.mesh = LineSegments::create(geometry, material);
            preview.mesh->name = "__editor_splat_surface_preview";
            // ...but never depth-tested against the CLOUD it is a picture of.
            // This shell hugs the cloud's front within a voxel, and the splat
            // overlay-depth stamp is re-expressed from the jittered raster's
            // AOV — at a grazing view from a distance that comparison flickers
            // per frame however it is biased (measured, and the numbers are in
            // SplatOverlayFlicker_probe). The layer draws these lines before
            // the stamp: occluded by geometry, never by splats.
            //
            // The layer is a Vulkan one. A GL session reaches here only for a
            // point-route cloud (the fusion bake declines earlier), and draws
            // the shell depth-tested against the splats like any other overlay
            // line — the flicker above is the price, and a visible preview is
            // worth more than none.
#ifdef THREEPP_WITH_VULKAN
            preview.mesh->layers.enable(VulkanRenderer::kSplatUnoccludedOverlayLayer);
#endif
            // splats::bakeSurface emits WORLD-space vertices (the play session
            // parents its twin at the scene root for this reason), so the
            // preview must carry no transform of its own.
            preview.mesh->matrixAutoUpdate = false;
            preview.key = key;
            overlay_->add(preview.mesh);
        }
        // applyAuthoringVisibility hides the authoring layer for a screenshot
        // pass, and nothing else would turn this back on.
        preview.mesh->visible = true;
        live.push_back(object.uuid);
    });

    // Deleted node, erased config, an undone Add: whatever the reason, nothing
    // in the document describes this picture any more.
    for (auto it = splatSurfacePreviews_.begin(); it != splatSurfacePreviews_.end();) {
        if (std::find(live.begin(), live.end(), it->first) != live.end()) {
            ++it;
            continue;
        }
        if (it->second.mesh) {
            it->second.mesh->removeFromParent();
            if (const auto geometry = it->second.mesh->geometry()) geometry->dispose();
        }
        it = splatSurfacePreviews_.erase(it);
    }
}

void EditorApp::clearSplatSurfacePreviews() {

    for (auto& [uuid, preview] : splatSurfacePreviews_) {
        if (!preview.mesh) continue;
        preview.mesh->removeFromParent();
        // The renderer keys GPU buffers on geometry identity; an orphan that is
        // never disposed leaks them and re-arms the recycled-pointer staleness
        // (SplineOverlay::writeSamples).
        if (const auto geometry = preview.mesh->geometry()) geometry->dispose();
    }
    splatSurfacePreviews_.clear();
}
