
#ifndef THREEPP_PHYSX_WORLD_HPP
#define THREEPP_PHYSX_WORLD_HPP

#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContext.h>
#include <cudamanager/PxCudaContextManager.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    struct SoftBodyTetBind {
        ::physx::PxU32 i0, i1, i2, i3;
        float w0, w1, w2, w3;
    };

    class SoftBody;// defined in PhysxSoftBody.hpp, included at the bottom of this file

    inline ::physx::PxVec3 toPxVec3(const Vector3& v) {
        return {v.x, v.y, v.z};
    }

    inline ::physx::PxQuat toPxQuat(const Quaternion& q) {
        return {q.x, q.y, q.z, q.w};
    }

    inline ::physx::PxTransform toPxTransform(const Vector3& pos, const Quaternion& rot = Quaternion()) {
        return ::physx::PxTransform(toPxVec3(pos), toPxQuat(rot));
    }

    inline Vector3 fromPxVec3(const ::physx::PxVec3& v) {
        return {v.x, v.y, v.z};
    }

    inline Quaternion fromPxQuat(const ::physx::PxQuat& q) {
        return Quaternion(q.x, q.y, q.z, q.w);
    }

    // Owns PhysX foundation/physics/scene. Single-scene wrapper aimed at scene-graph
    // integration; advanced use (multiple scenes, custom filter shaders, GPU dynamics)
    // can drop down to physics() / scene() / foundation() and ignore the helpers.
    //
    // Vehicle-ready: PxVehicle2 can register its update via onPreSubstep, then
    // raycast / sweep through scene() and write torques on the vehicle's wheels.
    class PhysxWorld {

    public:
        struct Settings {
            Vector3 gravity{0, -9.81f, 0};
            float fixedTimestep = 1.f / 60.f;
            int maxSubSteps = 4;
            unsigned numThreads = 2;
            // Enable GPU dynamics. Required for soft bodies (PxDeformableVolume),
            // particle systems, and GPU broadphase. Switches the scene to the TGS
            // solver. Needs a CUDA-capable GPU and the omniverse-physx GPU library
            // (gpu-library is copied next to the example by AddExample.cmake).
            bool enableGpuDynamics = false;
        };

        PhysxWorld() : PhysxWorld(Settings{}) {}

        explicit PhysxWorld(Settings s)
            : settings_(s) {

            using namespace ::physx;

            foundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errorCallback_);
            if (!foundation_) throw std::runtime_error("PxCreateFoundation failed");

            // trackOutstandingAllocations=true matches the GPU samples; harmless when off.
            physics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, PxTolerancesScale(),
                                       settings_.enableGpuDynamics);
            if (!physics_) throw std::runtime_error("PxCreatePhysics failed");

            if (!PxInitExtensions(*physics_, nullptr)) {
                throw std::runtime_error("PxInitExtensions failed");
            }

            if (settings_.enableGpuDynamics) {
                PxCudaContextManagerDesc cudaDesc;
                cuda_ = PxCreateCudaContextManager(*foundation_, cudaDesc, PxGetProfilerCallback());
                if (cuda_ && !cuda_->contextIsValid()) {
                    cuda_->release();
                    cuda_ = nullptr;
                }
                if (!cuda_) throw std::runtime_error("PhysxWorld: failed to create CUDA context (no CUDA GPU?)");
                {
                    PxScopedCudaLock _lock(*cuda_);
                    cuda_->getCudaContext()->streamCreate(&cudaCopyStream_, 0);
                }
            }

            dispatcher_ = PxDefaultCpuDispatcherCreate(settings_.numThreads);
            if (!dispatcher_) throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");

            PxSceneDesc desc(physics_->getTolerancesScale());
            desc.gravity = toPxVec3(settings_.gravity);
            desc.cpuDispatcher = dispatcher_;
            desc.filterShader = PxDefaultSimulationFilterShader;
            if (settings_.enableGpuDynamics) {
                desc.cudaContextManager = cuda_;
                desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
                desc.flags |= PxSceneFlag::eENABLE_PCM;
                desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
                desc.broadPhaseType = PxBroadPhaseType::eGPU;
                desc.gpuMaxNumPartitions = 8;
                desc.solverType = PxSolverType::eTGS;
            }
            scene_ = physics_->createScene(desc);
            if (!scene_) throw std::runtime_error("createScene failed");

            defaultMat_ = physics_->createMaterial(0.5f, 0.5f, 0.2f);
        }

        ~PhysxWorld() {
            using namespace ::physx;
            // Soft bodies must be released BEFORE the scene/physics/cuda context;
            // their destructor releases the PxDeformableVolume actor and frees pinned
            // host memory through the CUDA context.
            softBodies_.clear();
            for (auto& [_, entry] : cookCache_) {
                if (entry.mesh) entry.mesh->release();
            }
            cookCache_.clear();
            if (scene_) {
                scene_->release();
                scene_ = nullptr;
            }
            if (dispatcher_) {
                dispatcher_->release();
                dispatcher_ = nullptr;
            }
            if (cuda_ && cudaCopyStream_) {
                PxScopedCudaLock _lock(*cuda_);
                cuda_->getCudaContext()->streamDestroy(cudaCopyStream_);
                cudaCopyStream_ = nullptr;
            }
            if (physics_) {
                PxCloseExtensions();
                physics_->release();
                physics_ = nullptr;
            }
            if (cuda_) {
                cuda_->release();
                cuda_ = nullptr;
            }
            if (foundation_) {
                foundation_->release();
                foundation_ = nullptr;
            }
        }

        PhysxWorld(const PhysxWorld&) = delete;
        PhysxWorld& operator=(const PhysxWorld&) = delete;

        // Variable-rate caller, fixed-rate physics. Pre/post substep hooks fire
        // around each fetchResults boundary; bindings re-sync after the final substep.
        void step(float dt) {
            accumulator_ += dt;
            int steps = 0;
            while (accumulator_ >= settings_.fixedTimestep && steps < settings_.maxSubSteps) {
                substep(settings_.fixedTimestep);
                accumulator_ -= settings_.fixedTimestep;
                ++steps;
            }
            if (steps >= settings_.maxSubSteps) {
                accumulator_ = 0;// avoid spiral of death on hitches
            }
            if (steps > 0) {
                syncBindings();
            }
        }

        void onPreSubstep(std::function<void(float)> cb) { preSubstep_.push_back(std::move(cb)); }
        void onPostSubstep(std::function<void(float)> cb) { postSubstep_.push_back(std::move(cb)); }

        // After each step, copy actor's world pose into Object3D.position/quaternion.
        void bind(Object3D& obj, ::physx::PxRigidActor& actor) {
            objBindings_.push_back({&obj, &actor});
        }

        // Each instance i mirrors the pose of actors[i]. Per-instance scale is preserved
        // by decomposing the existing matrix; rotation/translation come from the actor.
        void bind(InstancedMesh& mesh, std::vector<::physx::PxRigidActor*> actors) {
            instBindings_.push_back({&mesh, std::move(actors)});
        }

        void unbind(Object3D& obj) {
            objBindings_.erase(
                    std::remove_if(objBindings_.begin(), objBindings_.end(),
                                   [&](const ObjBinding& b) { return b.obj == &obj; }),
                    objBindings_.end());
        }

        void unbind(InstancedMesh& mesh) {
            instBindings_.erase(
                    std::remove_if(instBindings_.begin(), instBindings_.end(),
                                   [&](const InstBinding& b) { return b.mesh == &mesh; }),
                    instBindings_.end());
        }

        // Low-level: explicit geometry + transform. Use when shape can't be inferred
        // (custom geometry, plane, trimesh) or when shape != mesh visuals.
        ::physx::PxRigidStatic* addStatic(const ::physx::PxGeometry& geom,
                                          const ::physx::PxTransform& tr,
                                          ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            if (!mat) mat = defaultMat_;
            PxRigidStatic* body = physics_->createRigidStatic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            scene_->addActor(*body);
            return body;
        }

        ::physx::PxRigidDynamic* addDynamic(const ::physx::PxGeometry& geom,
                                            const ::physx::PxTransform& tr,
                                            float density,
                                            ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            if (!mat) mat = defaultMat_;
            PxRigidDynamic* body = physics_->createRigidDynamic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            return body;
        }

        // High-level: shape inferred from mesh.geometry(), pose from mesh.matrixWorld,
        // automatic transform sync. Supported geometries: Box, Sphere, Capsule.
        // For unsupported geometries, fall back to addDynamic / addStatic.
        ::physx::PxRigidDynamic* add(Mesh& mesh, float density, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::add: mesh has no geometry");
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::add: unsupported geometry");
            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            PxRigidDynamic* body = physics_->createRigidDynamic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
            PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
            shape->setLocalPose(inferred.localPose);
            body->attachShape(*shape);
            shape->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            bind(mesh, *body);
            return body;
        }

        ::physx::PxRigidStatic* addStatic(Mesh& mesh, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addStatic: mesh has no geometry");
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::addStatic: unsupported geometry");
            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            PxRigidStatic* body = physics_->createRigidStatic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
            PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
            shape->setLocalPose(inferred.localPose);
            body->attachShape(*shape);
            shape->release();
            scene_->addActor(*body);
            return body;
        }

        // Cook a triangle mesh from arbitrary indexed geometry and add it as a static
        // collider. Returns nullptr if the geometry has no position attribute, fewer
        // than 3 vertices, or a non-triangle index buffer. Trimesh colliders are valid
        // only for static / kinematic actors — dynamics need a convex decomposition.
        ::physx::PxRigidStatic* addStaticTrimesh(
                const BufferGeometry& geometry,
                const ::physx::PxTransform& tr = ::physx::PxTransform(::physx::PxIdentity),
                const Vector3& scale = Vector3(1.f, 1.f, 1.f),
                ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            const auto* posAttr = geometry.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const auto& positions = posAttr->array();
            const PxU32 vertCount = static_cast<PxU32>(posAttr->count());
            if (vertCount < 3) return nullptr;

            // Build a u32 index buffer (copied from geometry index, or 0..N-1 for unindexed).
            std::vector<PxU32> indices;
            const auto* idxAttr = geometry.getIndex();
            if (idxAttr) {
                const auto& src = idxAttr->array();
                indices.assign(src.begin(), src.end());
            } else {
                indices.resize(vertCount);
                for (PxU32 i = 0; i < vertCount; ++i) indices[i] = i;
            }
            if (indices.size() < 3 || indices.size() % 3 != 0) return nullptr;

            PxTriangleMeshDesc desc;
            desc.points.count = vertCount;
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = positions.data();
            desc.triangles.count = static_cast<PxU32>(indices.size() / 3);
            desc.triangles.stride = sizeof(PxU32) * 3;
            desc.triangles.data = indices.data();

            PxCookingParams params(physics_->getTolerancesScale());
            PxTriangleMesh* triMesh = PxCreateTriangleMesh(params, desc);
            if (!triMesh) return nullptr;

            if (!mat) mat = defaultMat_;
            PxTriangleMeshGeometry geom(triMesh, PxMeshScale(toPxVec3(scale)));
            PxRigidStatic* body = physics_->createRigidStatic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            // Shape retains a reference; release our local one so the mesh is freed
            // when the shape (and therefore actor) goes away.
            triMesh->release();
            scene_->addActor(*body);
            return body;
        }

        // Build a static trimesh from mesh.geometry(); pose and scale come from
        // mesh.matrixWorld (decomposed). Convenience wrapper around the geometry overload.
        ::physx::PxRigidStatic* addStaticTrimesh(Mesh& mesh, ::physx::PxMaterial* mat = nullptr) {
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addStaticTrimesh: mesh has no geometry");
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            return addStaticTrimesh(
                    *g,
                    ::physx::PxTransform(toPxVec3(pos), toPxQuat(rot)),
                    scale,
                    mat);
        }

        // Walk the subtree and add every Mesh as its own static trimesh. Useful for
        // imported scenes (glTF tracks, environments) where the visual hierarchy is
        // also the collision geometry. Returns the created actors.
        // `filter` lets the caller skip meshes (e.g. dynamic obstacles); pass nullptr
        // to accept all.
        std::vector<::physx::PxRigidStatic*> addStaticTrimeshTree(
                Object3D& root,
                const std::function<bool(const Mesh&)>& filter = nullptr,
                ::physx::PxMaterial* mat = nullptr) {
            std::vector<::physx::PxRigidStatic*> out;
            root.updateMatrixWorld();
            root.traverseType<Mesh>([&](Mesh& m) {
                if (filter && !filter(m)) return;
                if (auto* body = addStaticTrimesh(m, mat)) out.push_back(body);
            });
            return out;
        }

        // Cook a convex hull from the mesh vertices and add it as a dynamic actor.
        // PhysX dynamics require convex shapes — for non-convex visuals (a traffic
        // cone, an L-bracket) the hull is the smallest convex envelope. Pose and
        // scale come from mesh.matrixWorld; the mesh is bound to the new actor so
        // it follows the simulation.
        ::physx::PxRigidDynamic* addDynamicConvex(
                Mesh& mesh, float density, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addDynamicConvex: mesh has no geometry");
            const auto* posAttr = g->getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const auto& positions = posAttr->array();
            const PxU32 vertCount = static_cast<PxU32>(posAttr->count());
            if (vertCount < 4) return nullptr;

            PxConvexMeshDesc desc;
            desc.points.count = vertCount;
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = positions.data();
            desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;

            PxCookingParams params(physics_->getTolerancesScale());
            PxConvexMesh* convex = PxCreateConvexMesh(params, desc);
            if (!convex) return nullptr;

            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);

            PxConvexMeshGeometry geom(convex, PxMeshScale(toPxVec3(scale)));
            PxRigidDynamic* body = physics_->createRigidDynamic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            convex->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            bind(mesh, *body);
            return body;
        }

        // One PxRigidDynamic per instance. Initial pose taken from each instance matrix.
        std::vector<::physx::PxRigidActor*> add(InstancedMesh& mesh,
                                                float density,
                                                ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::add(InstancedMesh): no geometry");
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::add(InstancedMesh): unsupported geometry");
            if (!mat) mat = defaultMat_;
            std::vector<PxRigidActor*> actors;
            actors.reserve(mesh.count());
            Matrix4 m;
            Vector3 pos, scale;
            Quaternion rot;
            for (size_t i = 0; i < mesh.count(); ++i) {
                mesh.getMatrixAt(i, m);
                m.decompose(pos, rot, scale);
                PxRigidDynamic* body = physics_->createRigidDynamic(
                        PxTransform(toPxVec3(pos), toPxQuat(rot)));
                PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
                shape->setLocalPose(inferred.localPose);
                body->attachShape(*shape);
                shape->release();
                PxRigidBodyExt::updateMassAndInertia(*body, density);
                scene_->addActor(*body);
                actors.push_back(body);
            }
            bind(mesh, actors);
            return actors;
        }

        ::physx::PxPhysics& physics() { return *physics_; }
        ::physx::PxScene& scene() { return *scene_; }
        ::physx::PxFoundation& foundation() { return *foundation_; }
        ::physx::PxMaterial& defaultMaterial() { return *defaultMat_; }
        ::physx::PxCpuDispatcher& dispatcher() { return *dispatcher_; }

        // CUDA context — non-null only when Settings::enableGpuDynamics was set.
        // Required for soft bodies, particle systems, and GPU broadphase.
        ::physx::PxCudaContextManager* cudaContextManager() const { return cuda_; }

        void setGravity(const Vector3& g) {
            settings_.gravity = g;
            scene_->setGravity(toPxVec3(g));
        }

        // --- Soft body API (requires Settings::enableGpuDynamics). Implementations
        // live in PhysxSoftBody.hpp because they depend on the full SoftBody type.

        // Create a deformable-volume material. Owned by PxPhysics; no manual release.
        ::physx::PxDeformableVolumeMaterial* createSoftBodyMaterial(
                float youngsModulus = 1e6f, float poissonsRatio = 0.45f,
                float dynamicFriction = 0.5f);

        // Cook the supplied geometry into a deformable volume and add it to the
        // scene. The geometry's position attribute is mutated each frame to match
        // the deformed simulation. The geometry's positions are taken as-is (world
        // space). voxelResolution sets the simulation mesh detail (~10 default;
        // higher = finer simulation + more solver work).
        SoftBody* addSoftBody(
                const std::shared_ptr<BufferGeometry>& visualGeometry,
                ::physx::PxDeformableVolumeMaterial* material = nullptr,
                int voxelResolution = 10,
                unsigned solverIterations = 20,
                bool selfCollision = false);

        // Convenience: bake mesh.matrixWorld into the geometry positions, reset
        // the mesh's local transform to identity, then add as a soft body. Useful
        // for typical scene-graph workflows (`mesh->position.set(...)` then add).
        // When cacheKey is non-empty, the expensive tet mesh cooking and per-vertex
        // binding computation are cached and reused for subsequent calls with the
        // same key (e.g. fish species ID). Only the first spawn of each key pays
        // the full cook cost.
        SoftBody* addSoftBody(
                Mesh& mesh,
                ::physx::PxDeformableVolumeMaterial* material = nullptr,
                int voxelResolution = 10,
                unsigned solverIterations = 20,
                bool selfCollision = false,
                const std::string& cacheKey = "");

        // Destroy a soft body. Releases the PhysX actor + GPU/pinned resources and
        // — when the body was created via the Mesh& overload — also removes the
        // visual Mesh from its parent in the scene graph (single-call cleanup).
        // Soft bodies created from a bare BufferGeometry leave the scene graph alone.
        void removeSoftBody(SoftBody* softBody);

    private:
        struct ObjBinding {
            Object3D* obj;
            ::physx::PxRigidActor* actor;
        };
        struct InstBinding {
            InstancedMesh* mesh;
            std::vector<::physx::PxRigidActor*> actors;
        };

        struct InferredShape {
            ::physx::PxGeometryHolder geom;
            ::physx::PxTransform localPose{::physx::PxIdentity};
            bool valid = true;
        };

        InferredShape inferShape(const BufferGeometry& geometry) const {
            using namespace ::physx;
            InferredShape out;
            if (auto box = dynamic_cast<const BoxGeometry*>(&geometry)) {
                out.geom = PxBoxGeometry(box->width * 0.5f, box->height * 0.5f, box->depth * 0.5f);
                return out;
            }
            if (auto sph = dynamic_cast<const SphereGeometry*>(&geometry)) {
                out.geom = PxSphereGeometry(sph->radius);
                return out;
            }
            if (auto cap = dynamic_cast<const CapsuleGeometry*>(&geometry)) {
                out.geom = PxCapsuleGeometry(cap->radius, cap->length * 0.5f);
                // PhysX capsule axis is X; threepp capsule axis is Y. Rotate -PI/2 about Z.
                out.localPose = PxTransform(PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
                return out;
            }
            out.valid = false;
            return out;
        }

        void substep(float dt) {
            for (auto& cb : preSubstep_) cb(dt);
            scene_->simulate(dt);
            scene_->fetchResults(true);
            for (auto& cb : postSubstep_) cb(dt);
        }

        // Pull deformed positions GPU->CPU for every soft body, then write them into
        // the bound BufferGeometry's position attribute. Defined in PhysxSoftBody.hpp.
        void syncSoftBodies();

        ::physx::PxDeformableVolumeMaterial* defaultSoftBodyMaterial();

        void syncBindings() {
            syncSoftBodies();
            for (auto& b : objBindings_) {
                auto t = b.actor->getGlobalPose();
                const Vector3 worldPos = fromPxVec3(t.p);
                const Quaternion worldRot = fromPxQuat(t.q);
                if (auto* parent = b.obj->parent) {
                    // Convert world pose into parent-local pose so the visual lands
                    // correctly when the bound object is nested under transformed groups.
                    parent->updateMatrixWorld();
                    Matrix4 worldMat;
                    worldMat.compose(worldPos, worldRot, Vector3(1.f, 1.f, 1.f));
                    Matrix4 inverseParent;
                    inverseParent.copy(*parent->matrixWorld).invert();
                    Matrix4 localMat;
                    localMat.multiplyMatrices(inverseParent, worldMat);
                    Vector3 unusedScale;
                    localMat.decompose(b.obj->position, b.obj->quaternion, unusedScale);
                } else {
                    b.obj->position.copy(worldPos);
                    b.obj->quaternion.copy(worldRot);
                }
            }
            for (auto& b : instBindings_) {
                Matrix4 m;
                Vector3 pos, scale;
                Quaternion rot;
                for (size_t i = 0; i < b.actors.size(); ++i) {
                    b.mesh->getMatrixAt(i, m);
                    m.decompose(pos, rot, scale);
                    auto t = b.actors[i]->getGlobalPose();
                    m.compose(fromPxVec3(t.p), fromPxQuat(t.q), scale);
                    b.mesh->setMatrixAt(i, m);
                }
                b.mesh->instanceMatrix()->needsUpdate();
                b.mesh->computeBoundingSphere();
            }
        }

        Settings settings_;
        ::physx::PxDefaultAllocator allocator_;
        ::physx::PxDefaultErrorCallback errorCallback_;
        ::physx::PxFoundation* foundation_ = nullptr;
        ::physx::PxPhysics* physics_ = nullptr;
        ::physx::PxDefaultCpuDispatcher* dispatcher_ = nullptr;
        ::physx::PxScene* scene_ = nullptr;
        ::physx::PxMaterial* defaultMat_ = nullptr;
        ::physx::PxCudaContextManager* cuda_ = nullptr;
        CUstream cudaCopyStream_ = nullptr;
        ::physx::PxDeformableVolumeMaterial* defaultSoftBodyMat_ = nullptr;
        struct CookCacheEntry {
            ::physx::PxDeformableVolumeMesh* mesh = nullptr;
            std::vector<SoftBodyTetBind> bindings;
            bool hasBindings = false;
        };
        std::unordered_map<std::string, CookCacheEntry> cookCache_;
        std::vector<std::unique_ptr<SoftBody>> softBodies_;
        float accumulator_ = 0.f;

        std::vector<ObjBinding> objBindings_;
        std::vector<InstBinding> instBindings_;
        std::vector<std::function<void(float)>> preSubstep_;
        std::vector<std::function<void(float)>> postSubstep_;
    };

}// namespace threepp

#include "threepp/extras/physx/PhysxSoftBody.hpp"

#endif//THREEPP_PHYSX_WORLD_HPP
