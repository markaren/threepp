// The PlaySession that makes an authored scan SOLID and SENSABLE: one baked
// triangle surface per SplatCloud carrying an enabled SplatSurfaceConfig, cooked
// into the PhysX world as a static collider and added to the scene as a
// sensor-only mesh.
//
// Header-only and PhysX-dependent, exactly like ConveyorPlaySession and
// GranularPlaySession — included only by builds that found the SDK. The world is
// BORROWED with the same contract: this session is registered after the physics
// one, so PlayController's reverse-order stop tears these down while that world
// is still alive.
//
// THE BAKE IS NOT DONE HERE. SplatSurfaceCache owns it (see that header for the
// key and the invalidation rule); this session only asks. A warm cache — the
// inspector's "Bake now" — makes Play instant; a cold one pays ~0.4 s of render
// and fusion on the first press, with the viewport flashing through the pose
// loop, which is why the button exists.
//
// The baked vertices are WORLD SPACE (splats::bakeSurface's contract), so the
// collider is cooked at identity and the sensor mesh goes to the SCENE ROOT —
// the granular visuals' rule, for the granular visuals' reason: a node with a
// transform would apply it twice. Both are scene/world content created after the
// play snapshot was taken, so Stop restores a document that never saw them.
//
// The sensor half is a SCENE-WIDE master (VulkanRenderer::setSensorOnlySurfaces)
// plus a per-mesh layer, so one enabled scan grants perception to every baked
// surface in the scene; turning it off is stop()'s job and not the snapshot's,
// because it is renderer state and the snapshot only restores the graph.

#ifndef THREEPP_EDITOR_SPLATSURFACEPLAYSESSION_HPP
#define THREEPP_EDITOR_SPLATSURFACEPLAYSESSION_HPP

#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SplatSurfaceCache.hpp"
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// The bake and the sensor master are both Vulkan constructs, so a build without
// the backend has no session to compile — the registration site carries the same
// guard. (SplatSurfaceConfig and SplatSurfaceCache still author and decline
// everywhere, which is what keeps a GL editor's documents readable.)
#ifdef THREEPP_WITH_VULKAN

namespace threepp::editor {

    class SplatSurfacePlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Splat surfaces"; }

        ~SplatSurfacePlaySession() override { clearSensorSurfaces(); }

        // --- wiring (set once, before the first Play) ------------------------

        // Where the colliders are cooked. Borrowed; null means the scan is
        // still sensable but nothing stands on it.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        // The renderer that bakes (Vulkan) and that owns the sensor-surface
        // master. Borrowed.
        void setRenderer(Renderer* renderer) { renderer_ = renderer; }

        // The editor's memo. Borrowed and shared with the inspector's "Bake
        // now" — that is the whole point of it not living here.
        void setCache(SplatSurfaceCache* cache) { cache_ = cache; }

        void setLogger(std::function<void(const std::string&)> logger) { logger_ = std::move(logger); }

        // --- readouts --------------------------------------------------------

        // Clouds carrying an enabled config that the last start() saw, whether
        // or not any of them baked.
        [[nodiscard]] std::size_t surfaceNodeCount() const { return surfaceNodeCount_; }

        // Surfaces actually realized this play.
        [[nodiscard]] std::size_t surfaceCount() const { return surfaces_.size(); }

        // Of those, how many cooked into a static collider.
        [[nodiscard]] std::size_t colliderCount() const { return colliderCount_; }

        // Whether the scene-wide sensor master is on because of this session.
        [[nodiscard]] bool sensorSurfaces() const { return masterOn_; }

        void start(Scene& scene) override {

            scene_ = &scene;
            surfaceNodeCount_ = 0;
            colliderCount_ = 0;
            surfaces_.clear();

            scene.updateMatrixWorld(true);

            // Collect first, realize second: the sensor meshes are added to the
            // same scene, and a traversal that grows under its own walk is a
            // question nobody should have to answer.
            std::vector<SplatCloud*> clouds;
            scene.traverse([&](Object3D& object) {
                if (auto* cloud = object.as<SplatCloud>()) {
                    const auto config = SplatSurfaceConfig::read(object);
                    if (config && config->enabled) clouds.push_back(cloud);
                }
            });
            surfaceNodeCount_ = clouds.size();
            if (clouds.empty()) return;

            if (!cache_) {
                log("splat surface: no bake cache is wired up - no scan is solid this play");
                return;
            }

            for (auto* cloud : clouds) {
                const auto config = SplatSurfaceConfig::read(*cloud);
                if (!config) continue;

                std::string problem;
                const auto* mesh = cache_->bake(renderer_, *cloud, *config, &problem);
                if (!mesh) {
                    log("splat surface: \"" + cloud->name + "\" - " + problem);
                    continue;
                }

                // World-space triangles, so the geometry is cooked and drawn
                // where it already is: identity pose, no scale.
                auto geometry = BufferGeometry::create();
                geometry->setAttribute("position",
                                       FloatBufferAttribute::create(mesh->positions, 3));
                geometry->setIndex(std::vector<unsigned int>(mesh->indices.begin(),
                                                            mesh->indices.end()));

                if (physics_ && physics_->world()) {
                    if (physics_->world()->addStaticTrimesh(*geometry)) {
                        ++colliderCount_;
                    } else {
                        log("splat surface: \"" + cloud->name +
                            "\" baked but did not cook into a collider");
                    }
                }

                if (auto surface = splats::makeSensorMesh(*mesh)) {
                    surface->name = cloud->name.empty() ? "Splat Surface"
                                                        : cloud->name + " Surface";
                    scene.add(surface);
                    surfaces_.push_back(std::move(surface));
                }

                char line[192];
                std::snprintf(line, sizeof(line),
                              "splat surface: \"%s\" %zu triangles at %.3f m voxels",
                              cloud->name.c_str(), mesh->triangleCount(),
                              static_cast<double>(mesh->stats.voxelSize));
                log(line);
            }

            if (!surfaces_.empty()) setSensorSurfaces(true);
        }

        void update(float) override {}

        void stop() override {

            clearSensorSurfaces();
            // The colliders are left to the world's own teardown: they are
            // static actors it owns, and PhysicsPlaySession::stop() — which runs
            // after this one, by reverse registration order — releases the whole
            // world a moment later.
            colliderCount_ = 0;
            scene_ = nullptr;
        }

    private:
        void log(const std::string& message) {

            if (logger_) logger_(message);
        }

        void setSensorSurfaces(bool enabled) {

            if (auto* vulkan = dynamic_cast<VulkanRenderer*>(renderer_)) {
                vulkan->setSensorOnlySurfaces(enabled);
                masterOn_ = enabled;
            }
        }

        // Both halves of the sensor state, in the order that leaves nothing
        // dangling: the meshes out of the scene, then the master off. Also the
        // destructor's path, for an app torn down mid-Play.
        void clearSensorSurfaces() {

            if (scene_) {
                for (auto& surface : surfaces_) scene_->remove(*surface);
            }
            surfaces_.clear();
            if (masterOn_) setSensorSurfaces(false);
            masterOn_ = false;
        }

        PhysicsPlaySession* physics_ = nullptr;
        Renderer* renderer_ = nullptr;
        SplatSurfaceCache* cache_ = nullptr;
        std::function<void(const std::string&)> logger_;

        Scene* scene_ = nullptr;
        std::vector<std::shared_ptr<Mesh>> surfaces_;
        std::size_t surfaceNodeCount_ = 0;
        std::size_t colliderCount_ = 0;
        bool masterOn_ = false;
    };

}// namespace threepp::editor

#endif//THREEPP_WITH_VULKAN

#endif//THREEPP_EDITOR_SPLATSURFACEPLAYSESSION_HPP
