// The memo behind the surface bake: one baked mesh per splat node, rebuilt only
// when something it depends on changes.
//
// A bake renders the cloud from 26 poses and fuses them on the CPU — 0.4 s on a
// synthetic cloud, and it scales with allocated blocks x poses. Play must not
// pay that on every press, and the inspector's "Bake now" must warm exactly what
// Play will use, so the memo lives here (owned by the editor app) rather than
// inside the play session the way PhysicsPlaySession's V-HACD cache does. That
// cache is per-session because a decomposition has no authoring UI; this one has
// a button.
//
// THE KEY IS THE WHOLE INVALIDATION STORY: the cloud's identity and splat count,
// the config, and the node's WORLD MATRIX — bakeSurface emits world-space
// vertices, so moving the scan is a different surface. Anything the key does not
// name cannot change the mesh, because the bake is deterministic in exactly
// these inputs (splats::SplatSurface.hpp). One entry per node: a re-bake under a
// new key REPLACES the old one, so dragging the voxel slider does not accumulate
// a megabyte of stale meshes.
//
// Vulkan only, and it says so rather than pretending: the depth AOV the bake
// reads is a Vulkan G-buffer attachment. On any other backend bake() declines
// with a reason and the caller reports it.

#ifndef THREEPP_EDITOR_SPLATSURFACECACHE_HPP
#define THREEPP_EDITOR_SPLATSURFACECACHE_HPP

#include "threepp/extras/editor/SplatSurfaceConfig.hpp"
#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/Renderer.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/splats/SplatSurface.hpp"
#endif

#include <cstddef>
#include <string>
#include <unordered_map>

namespace threepp::editor {

#ifdef THREEPP_WITH_VULKAN

    class SplatSurfaceCache {

    public:
        // Which renderer can bake at all. Null reads as "no" — a headless or GL
        // session authors the config and declines to realize it.
        [[nodiscard]] static bool available(const Renderer* renderer) {
            return dynamic_cast<const VulkanRenderer*>(renderer) != nullptr;
        }

        // The baked mesh for this node under this config, or nullptr on a miss.
        // Non-const because the key includes the world matrix, which has to be
        // up to date to mean anything.
        [[nodiscard]] const splats::SurfaceMesh* find(SplatCloud& cloud,
                                                      const SplatSurfaceConfig& config) {

            const auto it = entries_.find(cloud.uuid);
            if (it == entries_.end()) return nullptr;
            return it->second.key == keyFor(cloud, config) ? &it->second.mesh : nullptr;
        }

        // find(), or bake it now. `problem` (optional) gets the reason on a
        // decline. The bake reparents the cloud into a private stage and renders
        // 26 frames through the primary view: the viewport visibly flashes, so
        // every caller runs it as a discrete user-visible action, never per
        // frame.
        const splats::SurfaceMesh* bake(Renderer* renderer, SplatCloud& cloud,
                                        const SplatSurfaceConfig& config,
                                        std::string* problem = nullptr) {

            if (const auto* hit = find(cloud, config)) return hit;

            auto* vulkan = dynamic_cast<VulkanRenderer*>(renderer);
            if (!vulkan) {
                if (problem) *problem = "the surface bake needs the Vulkan backend";
                return nullptr;
            }
            if (cloud.splatCount() == 0) {
                if (problem) *problem = "the cloud has no splats";
                return nullptr;
            }

            splats::SurfaceBakeOptions options;
            options.voxelSize = config.voxelSize;
            if (config.minComponentVoxels > 0) options.minComponentVoxels = config.minComponentVoxels;
            if (config.poseCount > 0) options.poseCount = config.poseCount;

            auto mesh = splats::bakeSurface(*vulkan, cloud, options);
            ++bakeCount_;
            if (mesh.empty()) {
                if (problem) *problem = "the bake found no surface";
                // Not cached: an empty result is usually a scan the poses could
                // not see, and the next press should try again rather than
                // return the same nothing instantly.
                return nullptr;
            }

            auto& entry = entries_[cloud.uuid];
            entry.key = keyFor(cloud, config);
            entry.mesh = std::move(mesh);
            return &entry.mesh;
        }

        // Bakes actually run since construction — a cache-hit test reads this,
        // exactly as PhysicsPlaySession::decompositionCookCount() is read.
        [[nodiscard]] std::size_t bakeCount() const { return bakeCount_; }

        [[nodiscard]] std::size_t size() const { return entries_.size(); }

        // A new document invalidates everything: uuids are unique per node, but
        // holding meshes for a scene that is gone is a leak with a plausible
        // excuse.
        void clear() { entries_.clear(); }

        // The key find() matches on, for a consumer that CACHES what a hit gave
        // it: a find() that succeeds says the mesh is current, not that it is
        // the same mesh as last frame (a re-bake under new knobs replaces the
        // entry in place). The editor's viewport preview keys its geometry on
        // this string for exactly that reason.
        [[nodiscard]] static std::string keyFor(SplatCloud& cloud, const SplatSurfaceConfig& config) {

            cloud.updateWorldMatrix(true, false);
            std::string key = std::to_string(cloud.splatCount()) + "|" + config.encode() + "|m=";
            for (const float e : cloud.matrixWorld->elements) {
                key += codec::number(e);
                key += ',';
            }
            return key;
        }

    private:
        struct Entry {
            std::string key;
            splats::SurfaceMesh mesh;
        };

        std::unordered_map<std::string, Entry> entries_;// by cloud uuid
        std::size_t bakeCount_ = 0;
    };

#else

    // Same shape, no Vulkan: the config authors and the callers report one
    // reason instead of failing to link against a bake that was not compiled.
    class SplatSurfaceCache {

    public:
        [[nodiscard]] static bool available(const Renderer*) { return false; }

        [[nodiscard]] const void* find(SplatCloud&, const SplatSurfaceConfig&) { return nullptr; }

        const void* bake(Renderer*, SplatCloud&, const SplatSurfaceConfig&,
                         std::string* problem = nullptr) {
            if (problem) *problem = "this build has no Vulkan backend to bake with";
            return nullptr;
        }

        [[nodiscard]] std::size_t bakeCount() const { return 0; }
        [[nodiscard]] std::size_t size() const { return 0; }
        void clear() {}
    };

#endif

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLATSURFACECACHE_HPP
