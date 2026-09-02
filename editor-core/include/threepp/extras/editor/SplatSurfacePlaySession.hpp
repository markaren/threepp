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
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#ifdef THREEPP_EDITOR_WITH_VHACD
#include "threepp/extras/physx/ConvexDecomposition.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

        // Of those, how many cooked into a collider (static, dynamic or
        // kinematic alike).
        [[nodiscard]] std::size_t colliderCount() const { return colliderCount_; }

        // Of those, how many are moving bodies (SplatSurfaceConfig::Dynamic or
        // Kinematic): a compound of convex hulls at the cloud's pose.
        [[nodiscard]] std::size_t movingCount() const { return moving_.size(); }

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

                const bool movingBody = config->body != SplatSurfaceConfig::Static;

                if (!movingBody) {

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
                    continue;
                }

                // ── a moving body ───────────────────────────────────────────
                // PhysX will not simulate a triangle mesh, so the surface is
                // split into convex hulls and welded into one compound actor at
                // the cloud's pose. Everything below is expressed in that
                // actor's frame — the hulls, and the sensor mesh, which becomes
                // a CHILD of the cloud so the scene graph carries it along —
                // because the bake's vertices are world space and the body is
                // about to leave that place.
                Vector3 worldPos, worldScale;
                Quaternion worldRot;
                cloud->matrixWorld->decompose(worldPos, worldRot, worldScale);
                Matrix4 frame;// the actor's frame: pose without scale
                frame.compose(worldPos, worldRot, Vector3{1.f, 1.f, 1.f});
                Matrix4 toFrame;
                toFrame.copy(frame).invert();

                std::vector<float> local(mesh->positions.size());
                for (std::size_t i = 0; i < mesh->positions.size(); i += 3) {
                    Vector3 p{mesh->positions[i], mesh->positions[i + 1], mesh->positions[i + 2]};
                    p.applyMatrix4(toFrame);
                    local[i] = p.x;
                    local[i + 1] = p.y;
                    local[i + 2] = p.z;
                }

                std::size_t hullCount = 0;
                if (physics_ && physics_->world()) {

                    const auto hulls = decomposeHulls(*cloud, *config, local, mesh->indices);
                    auto* world = physics_->world();
                    std::vector<PhysxWorld::ConvexPart> parts;
                    std::vector<::physx::PxConvexMesh*> cooked;
                    for (const auto& hull : hulls) {
                        if (hull.size() < 12) continue;// < 4 points
                        auto* convex = world->cookConvexHull(hull.data(), hull.size() / 3,
                                                             static_cast<unsigned>(64));
                        if (!convex) continue;
                        cooked.push_back(convex);
                        parts.push_back({convex, ::physx::PxTransform(::physx::PxIdentity),
                                         Vector3{1.f, 1.f, 1.f}});
                    }

                    const ::physx::PxTransform pose(
                            ::physx::PxVec3(worldPos.x, worldPos.y, worldPos.z),
                            ::physx::PxQuat(worldRot.x, worldRot.y, worldRot.z, worldRot.w));
                    auto* actor = world->addCompound(pose, parts, /*dynamic=*/true, 1000.f);
                    for (auto* convex : cooked) convex->release();

                    if (actor) {
                        auto* body = static_cast<::physx::PxRigidDynamic*>(actor);
                        if (config->body == SplatSurfaceConfig::Kinematic) {
                            body->setRigidBodyFlag(::physx::PxRigidBodyFlag::eKINEMATIC, true);
                        } else {
                            ::physx::PxRigidBodyExt::setMassAndUpdateInertia(*body,
                                                                             std::max(config->mass, 1e-3f));
                            // The cloud follows the simulation, parents and all
                            // (PhysxWorld::syncRigidBindings converts to the
                            // parent's frame).
                            world->bind(*cloud, *actor);
                        }
                        moving_.push_back({cloud, body, config->body == SplatSurfaceConfig::Kinematic});
                        hullCount = parts.size();
                        ++colliderCount_;
                    } else {
                        log("splat surface: \"" + cloud->name +
                            "\" baked but none of its hulls cooked - no moving collider");
                    }
                }

                if (auto surface = splats::makeSensorMesh(*mesh)) {
                    // Into the CLOUD's local frame — scale included, since the
                    // scene graph will apply the cloud's full matrix — and under
                    // the cloud, so it rides along.
                    Matrix4 toLocal;
                    toLocal.copy(*cloud->matrixWorld).invert();
                    auto* position = surface->geometry()->getAttribute<float>("position");
                    auto& arr = position->array();
                    for (std::size_t i = 0; i < arr.size(); i += 3) {
                        Vector3 p{arr[i], arr[i + 1], arr[i + 2]};
                        p.applyMatrix4(toLocal);
                        arr[i] = p.x;
                        arr[i + 1] = p.y;
                        arr[i + 2] = p.z;
                    }
                    position->needsUpdate();
                    surface->geometry()->computeVertexNormals();
                    surface->geometry()->computeBoundingSphere();
                    surface->name = cloud->name.empty() ? "Splat Surface"
                                                        : cloud->name + " Surface";
                    cloud->add(surface);
                    surfaces_.push_back(std::move(surface));
                }

                char line[224];
                std::snprintf(line, sizeof(line),
                              "splat surface: \"%s\" %zu triangles at %.3f m voxels, %s body of "
                              "%zu convex hull(s)%s",
                              cloud->name.c_str(), mesh->triangleCount(),
                              static_cast<double>(mesh->stats.voxelSize),
                              config->body == SplatSurfaceConfig::Kinematic ? "kinematic" : "dynamic",
                              hullCount, hullCount ? "" : " (none cooked)");
                log(line);
            }

            if (!surfaces_.empty()) setSensorSurfaces(true);
        }

        // A kinematic scan is driven by its node: whatever moved the cloud this
        // frame (a script, an animation) becomes the actor's target, and the
        // dynamics it meets are pushed. A dynamic one is written the other way
        // by the world's own bindings and needs nothing here.
        void update(float) override {

            for (auto& m : moving_) {
                if (!m.kinematic || !m.actor || !m.cloud) continue;
                m.cloud->updateWorldMatrix(true, false);
                Vector3 p, s;
                Quaternion q;
                m.cloud->matrixWorld->decompose(p, q, s);
                m.actor->setKinematicTarget(::physx::PxTransform(
                        ::physx::PxVec3(p.x, p.y, p.z), ::physx::PxQuat(q.x, q.y, q.z, q.w)));
            }
        }

        void stop() override {

            moving_.clear();
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
                // A static surface sits at the scene root, a moving one under
                // its cloud; removeFromParent covers both.
                for (auto& surface : surfaces_) surface->removeFromParent();
            }
            surfaces_.clear();
            if (masterOn_) setSensorSurfaces(false);
            masterOn_ = false;
        }

        // The hulls a moving body is made of: V-HACD over the surface when it
        // was linked in, one hull of every vertex otherwise (the body still
        // simulates, without its concavities, and the log says so).
        std::vector<std::vector<float>> decomposeHulls(const SplatCloud& cloud,
                                                       const SplatSurfaceConfig& config,
                                                       const std::vector<float>& positions,
                                                       const std::vector<uint32_t>& indices) {

            std::vector<std::vector<float>> hulls;
#ifdef THREEPP_EDITOR_WITH_VHACD
            const std::size_t tris = indices.size() / 3;
            if (tris > 200000) {
                log("splat surface: decomposing a " + std::to_string(tris) +
                    "-triangle surface - this may take a while");
            }
            ConvexDecompositionParams p;
            p.maxHulls = static_cast<std::uint32_t>(std::clamp(config.hulls, 1, 256));
            p.maxVertsPerHull = 64;
            p.voxelResolution = 100000;
            hulls = decomposeConvex(positions.data(), positions.size() / 3,
                                    indices.data(), indices.size(), p);
#else
            (void) indices;
            log("splat surface: \"" + cloud.name +
                "\" - this build has no V-HACD, so the moving body is one convex hull");
#endif
            (void) cloud;
            (void) config;
            if (hulls.empty()) hulls.emplace_back(positions.begin(), positions.end());
            return hulls;
        }

        struct MovingBody {
            SplatCloud* cloud = nullptr;
            ::physx::PxRigidDynamic* actor = nullptr;
            bool kinematic = false;
        };
        std::vector<MovingBody> moving_;

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
