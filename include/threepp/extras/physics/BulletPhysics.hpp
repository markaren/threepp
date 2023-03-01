
#ifndef THREEPP_BULLETPHYSICS_HPP
#define THREEPP_BULLETPHYSICS_HPP

#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>

#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/geometries/geometries.hpp"

#include <map>
#include <vector>

namespace threepp {

    namespace detail {

        void compose(const btVector3& position, const btQuaternion& quaternion, std::vector<float>& array, size_t index) {

            auto x = quaternion.x(), y = quaternion.y(), z = quaternion.z(), w = quaternion.w();
            auto x2 = x + x, y2 = y + y, z2 = z + z;
            auto xx = x * x2, xy = x * y2, xz = x * z2;
            auto yy = y * y2, yz = y * z2, zz = z * z2;
            auto wx = w * x2, wy = w * y2, wz = w * z2;

            array[index + 0] = (1 - (yy + zz));
            array[index + 1] = (xy + wz);
            array[index + 2] = (xz - wy);
            array[index + 3] = 0;

            array[index + 4] = (xy - wz);
            array[index + 5] = (1 - (xx + zz));
            array[index + 6] = (yz + wx);
            array[index + 7] = 0;

            array[index + 8] = (xz + wy);
            array[index + 9] = (yz - wx);
            array[index + 10] = (1 - (xx + yy));
            array[index + 11] = 0;

            array[index + 12] = position.x();
            array[index + 13] = position.y();
            array[index + 14] = position.z();
            array[index + 15] = 1;
        }

    }// namespace detail

    inline btVector3 tobtVector(const Vector3& v) {

        return {v.x, v.y, v.z};
    }

    inline btQuaternion tobtQuaternion(const Quaternion& q) {

        return {q.x(), q.y(), q.z(), q.w()};
    }


    struct RigidBodyConstructionInfo {
        std::unique_ptr<btCollisionShape> shape;
        std::unique_ptr<btMotionState> motionState;
        std::unique_ptr<btRigidBody> body;

        RigidBodyConstructionInfo(std::unique_ptr<btCollisionShape> shape, std::unique_ptr<btMotionState> state, std::unique_ptr<btRigidBody> body)
            : shape(std::move(shape)), motionState(std::move(state)), body(std::move(body)) {}
    };

    class BulletPhysics {

    public:
        explicit BulletPhysics(const Vector3& gravity = {0, -9.81f, 0})
            : dispatcher(&collisionConfiguration),
              world(&dispatcher, &broadphase, &solver, &collisionConfiguration),
              onObjectRemovedListener(std::make_shared<ObjectRemovedListener>(this)) {

            world.setGravity(tobtVector(gravity));
        }

        void step(float dt) {

            world.stepSimulation(dt, 10);

            for (auto& [mesh, rbInfo] : meshMap) {

                if (rbInfo->body->getMass() == 0) continue;

                if (mesh->is<InstancedMesh>()) {

                    // TODO

                } else if (mesh->is<Mesh>()) {

                    auto& motionState = rbInfo->motionState;
                    motionState->getWorldTransform(worldTransform);

                    auto& position = worldTransform.getOrigin();
                    auto quaternion = worldTransform.getRotation();
                    mesh->position.set(position.x(), position.y(), position.z());
                    mesh->quaternion.set(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
                }
            }
        }

        RigidBodyConstructionInfo* addMesh(Mesh& mesh, float mass = 0, bool disableDeactivation = false) {

            auto shape = getShape(mesh.geometry());

            if (shape) {

                if (mesh.is<InstancedMesh>()) {

                    // TODO

                } else if (mesh.is<Mesh>()) {

                    return handleMesh(mesh, mass, std::move(shape), disableDeactivation);
                }
            }

            return nullptr;
        }

        RigidBodyConstructionInfo* handleMesh(Mesh& mesh, float mass, std::unique_ptr<btCollisionShape> shape, bool disableDeactivation) {

            const auto& position = mesh.position;
            const auto& quaternion = mesh.quaternion;

            auto transform = btTransform();
            transform.setIdentity();
            transform.setOrigin(tobtVector(position));
            transform.setRotation(tobtQuaternion(quaternion));

            auto motionState = std::make_unique<btDefaultMotionState>(transform);
            btVector3 localInertia(0, 0, 0);
            shape->calculateLocalInertia(mass, localInertia);

            btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, motionState.get(), shape.get(), localInertia);

            auto body = std::make_unique<btRigidBody>(rbInfo);
            world.addRigidBody(body.get());

            if (disableDeactivation) {
                body->setActivationState(DISABLE_DEACTIVATION);
            }

            mesh.addEventListener("remove", onObjectRemovedListener);

            meshMap[&mesh] = std::make_unique<RigidBodyConstructionInfo>(std::move(shape), std::move(motionState), std::move(body));

            return meshMap.at(&mesh).get();
        }

        static std::unique_ptr<btCollisionShape> getShape(BufferGeometry* geometry) {

            if (geometry->type() == "BoxGeometry") {
                auto g = dynamic_cast<BoxGeometry*>(geometry);
                auto shape = std::make_unique<btBoxShape>(tobtVector({g->width, g->height, g->depth}) / 2);
                shape->setMargin(0.05f);

                return shape;

            } else if (geometry->type() == "SphereGeometry") {

                auto g = dynamic_cast<SphereGeometry*>(geometry);
                auto shape = std::make_unique<btSphereShape>(g->radius);
                shape->setMargin(0.05f);

                return shape;

            } else if (geometry->type() == "PlaneGeometry") {

                auto g = dynamic_cast<PlaneGeometry*>(geometry);
                auto shape = std::make_unique<btBoxShape>(btVector3(g->width, 0.2f, g->height) / 2);
                shape->setMargin(0.05f);

                return shape;

            } else if (geometry->type() == "CylinderGeometry") {

                auto g = dynamic_cast<CylinderGeometry*>(geometry);
                auto shape = std::make_unique<btCylinderShape>(btVector3{g->radiusTop, g->height, g->radiusBottom});
                shape->setMargin(0.05f);

                return shape;

            } else if (geometry->type() == "CapsuleGeometry") {

                auto g = dynamic_cast<CapsuleGeometry*>(geometry);
                auto shape = std::make_unique<btCapsuleShape>(g->radius, g->length);
                shape->setMargin(0.05f);

                return shape;

            } else if (geometry->type() == "ConeGeometry") {

                auto g = dynamic_cast<ConeGeometry*>(geometry);
                auto shape = std::make_unique<btConeShape>(g->radiusBottom, g->height);
                shape->setMargin(0.05f);

                return shape;
            }

            return nullptr;
        }

        void addConstraint(btTypedConstraint* c, bool disableCollisionsBetweenLinkedBodies = false) {

            world.addConstraint(c, disableCollisionsBetweenLinkedBodies);
        }

    private:
        btDefaultCollisionConfiguration collisionConfiguration;
        btCollisionDispatcher dispatcher;
        btDbvtBroadphase broadphase;
        btSequentialImpulseConstraintSolver solver;
        btDiscreteDynamicsWorld world;

        btTransform worldTransform;

        struct ObjectRemovedListener: EventListener {

            explicit ObjectRemovedListener(BulletPhysics* scope): scope(scope) {}

            void onEvent(Event& event) override {
                if (event.type == "remove") {
                    auto m = static_cast<Mesh*>(event.target);
                    if (scope->meshMap.count(m)) {
                        auto& rb = scope->meshMap.at(m);
                        scope->world.removeRigidBody(rb->body.get());
                        scope->meshMap.erase(m);
                    }
                }
            }

        private:
            BulletPhysics* scope;
        };

        std::shared_ptr<ObjectRemovedListener> onObjectRemovedListener;
        std::map<Mesh*, std::unique_ptr<RigidBodyConstructionInfo>> meshMap;
    };

}// namespace threepp

#endif//THREEPP_BULLETPHYSICS_HPP
