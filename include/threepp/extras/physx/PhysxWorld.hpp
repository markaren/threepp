
#ifndef THREEPP_PHYSX_WORLD_HPP
#define THREEPP_PHYSX_WORLD_HPP

#include <PxPhysicsAPI.h>

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
#include <vector>

namespace threepp {

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
        };

        explicit PhysxWorld(Settings s = {})
            : settings_(s) {

            using namespace ::physx;

            foundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errorCallback_);
            if (!foundation_) throw std::runtime_error("PxCreateFoundation failed");

            physics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, PxTolerancesScale());
            if (!physics_) throw std::runtime_error("PxCreatePhysics failed");

            if (!PxInitExtensions(*physics_, nullptr)) {
                throw std::runtime_error("PxInitExtensions failed");
            }

            dispatcher_ = PxDefaultCpuDispatcherCreate(settings_.numThreads);
            if (!dispatcher_) throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");

            PxSceneDesc desc(physics_->getTolerancesScale());
            desc.gravity = toPxVec3(settings_.gravity);
            desc.cpuDispatcher = dispatcher_;
            desc.filterShader = PxDefaultSimulationFilterShader;
            scene_ = physics_->createScene(desc);
            if (!scene_) throw std::runtime_error("createScene failed");

            defaultMat_ = physics_->createMaterial(0.5f, 0.5f, 0.2f);
        }

        ~PhysxWorld() {
            using namespace ::physx;
            if (scene_) {
                scene_->release();
                scene_ = nullptr;
            }
            if (dispatcher_) {
                dispatcher_->release();
                dispatcher_ = nullptr;
            }
            if (physics_) {
                PxCloseExtensions();
                physics_->release();
                physics_ = nullptr;
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

        void setGravity(const Vector3& g) {
            settings_.gravity = g;
            scene_->setGravity(toPxVec3(g));
        }

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

        void syncBindings() {
            for (auto& b : objBindings_) {
                auto t = b.actor->getGlobalPose();
                b.obj->position.copy(fromPxVec3(t.p));
                b.obj->quaternion.copy(fromPxQuat(t.q));
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
        float accumulator_ = 0.f;

        std::vector<ObjBinding> objBindings_;
        std::vector<InstBinding> instBindings_;
        std::vector<std::function<void(float)>> preSubstep_;
        std::vector<std::function<void(float)>> postSubstep_;
    };

}// namespace threepp

#endif//THREEPP_PHYSX_WORLD_HPP
