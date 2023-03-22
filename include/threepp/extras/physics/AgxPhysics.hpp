
#ifndef THREEPP_AGXPHYSICS_HPP
#define THREEPP_AGXPHYSICS_HPP


#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxSDK/Simulation.h>
#include <agxIO/ReaderWriter.h>

#include <optional>

#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/geometries/geometries.hpp"


namespace threepp {

    inline agx::Vec3 toVec3(const Vector3& v) {

        return {v.x, v.y, v.z};
    }

    inline agx::Quat toQuat(const Quaternion& q) {

        return {q.x(), q.y(), q.z(), q.w()};
    }

    namespace detail {

        agxCollide::Geometry* getGeometry(BufferGeometry& geometry) {

            if (geometry.type() == "SphereGeometry") {

                auto g = dynamic_cast<SphereGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Sphere(g->radius));

            } else if (geometry.type() == "BoxGeometry") {

                auto g = dynamic_cast<BoxGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Box(g->width/2, g->height/2, g->depth/2));

            } else if (geometry.type() == "PlaneGeometry") {

                auto g = dynamic_cast<PlaneGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Box(g->width/2, 0.05f, g->height/2));

            } else if (geometry.type() == "CapsuleGeometry") {

                auto g = dynamic_cast<CapsuleGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Capsule(g->radius, g->length));

            }

            return nullptr;
        }

        agx::RigidBodyRef createRigidBody(float mass, agxCollide::Geometry* geometry) {

            agx::RigidBodyRef rb = new agx::RigidBody();
            if (mass == 0) {
                rb->setMotionControl(agx::RigidBody::STATIC);
            } else {
                rb->getMassProperties()->setMass(mass);
            }
            rb->add(geometry->clone());

            return rb;
        }

        void compose(const agx::Vec3& position, const agx::Quat& quaternion, std::vector<float>& array, size_t index) {

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

    class AgxPhysics {

    public:
        AgxPhysics() {
            setGravity({0,-9.81,0});
        };

        explicit AgxPhysics(float timeStep) {

            sim->setTimeStep(timeStep);
            setGravity({0,-9.81,0});
        }

        void setGravity(const Vector3& g) {

            sim->setUniformGravity(toVec3(g));
        }

        void addMesh(Mesh& mesh, float mass = 0) {

            auto g = mesh.geometry();
            if (!g) return ;

            auto geometry = detail::getGeometry(*g);

            if (geometry) {

                if (mesh.is<InstancedMesh>()) {

                    handleInstancedMesh(mesh.as<InstancedMesh>(), mass, geometry);

                } else if (mesh.is<Mesh>()) {

                    handleMesh(&mesh, mass, geometry);
                }
            }
        }

        void setMeshPosition(Mesh& mesh, const Vector3& position, unsigned int index = 0) {

            if (!meshMap.count(&mesh)) return;

            auto& body = meshMap.at(&mesh);

            body->setAngularVelocity({0,0,0});
            body->setVelocity({0,0,0});

            body->setPosition(toVec3(position));
        }

        void setInstancedMeshPosition(InstancedMesh& mesh, const Vector3& position, unsigned int index = 0) {

            if (!instancedMeshMap.count(&mesh)) return;

            auto& bodies = instancedMeshMap.at(&mesh);
            auto& body = bodies[index];

            body->setAngularVelocity({0,0,0});
            body->setVelocity({0,0,0});

            body->setPosition(toVec3(position));
        }

        void step(float dt) {

            t += dt;
            while (sim->getTimeStamp() + sim->getTimeStep() < t) {

                sim->stepForward();

                for (auto& [mesh, rb] : meshMap) {

                    if (rb->getMassProperties()->getMass() == 0) continue;

                    auto position = rb->getPosition();
                    auto quaternion = rb->getRotation();

                    mesh->position.set(position.x(), position.y(), position.z());
                    mesh->quaternion.set(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
                }

                for (auto& [mesh, bodies] : instancedMeshMap) {

                    auto& array = mesh->instanceMatrix->array();

                    for (unsigned j = 0; j < bodies.size(); ++j) {

                        auto& body = bodies[j];

                        auto position = body->getPosition();
                        auto quaternion = body->getRotation();

                        detail::compose(position, quaternion, array, j * 16);
                    }

                    mesh->instanceMatrix->needsUpdate();
                }
            }
        }

        void saveScene(const std::string& name) {

            agxIO::writeFile(name + ".agx", sim);
        }

    private:
        float t = 0;
        agx::AutoInit init;
        agxSDK::SimulationRef sim = new agxSDK::Simulation();
        std::unordered_map<Mesh*, agx::RigidBodyRef> meshMap;
        std::unordered_map<InstancedMesh*, std::vector<agx::RigidBodyRef>> instancedMeshMap;

        void handleMesh(Mesh* mesh, float mass, agxCollide::Geometry* geometry) {

            auto rb = detail::createRigidBody(mass, geometry);
            const auto& position = mesh->position;
            const auto& quaternion = mesh->quaternion;

            rb->setPosition(toVec3(position));
            rb->setRotation(toQuat(quaternion));
            sim->add(rb);

            meshMap[mesh] = rb;
            mesh->addEventListener("remove", onMeshRemovedListener);
        }

        void handleInstancedMesh(InstancedMesh* mesh, float mass, agxCollide::Geometry* geometry) {

            auto& array = mesh->instanceMatrix->array();

            for (unsigned i = 0; i < mesh->count; i++) {

                unsigned index = i * 16;
                std::vector<double> slice{array.begin() + index, array.begin() + index + 16};

                auto rb = detail::createRigidBody(mass, geometry);
//                rb->setPosition({slice[13], slice[14], slice[15]});
                rb->setLocalTransform(agx::AffineMatrix4x4().set(slice.data()));
                sim->add(rb);

                instancedMeshMap[mesh].emplace_back(rb);
            }

            mesh->addEventListener("remove", onInstancedMeshRemovedListener);
        }

        struct MeshRemovedListener: EventListener {

            explicit MeshRemovedListener(AgxPhysics* scope): scope(scope) {}

            void onEvent(Event& event) override {
                if (event.type == "remove") {
                    auto m = static_cast<Mesh*>(event.target);
                    if (scope->meshMap.count(m)) {
                        auto rb = scope->meshMap.at(m);
                        scope->sim->remove(rb);
                        scope->meshMap.erase(m);
                    }
                }
            }

        private:
            AgxPhysics* scope;
        };

        struct InstancedMeshRemovedListener: EventListener {

            explicit InstancedMeshRemovedListener(AgxPhysics* scope): scope(scope) {}

            void onEvent(Event& event) override {
                if (event.type == "remove") {
                    auto m = static_cast<Mesh*>(event.target);
                    if (scope->meshMap.count(m)) {
                        auto rb = scope->meshMap.at(m);
                        scope->sim->remove(rb);
                        scope->meshMap.erase(m);
                    }
                }
            }

        private:
            AgxPhysics* scope;
        };

        std::shared_ptr<EventListener> onMeshRemovedListener = std::make_unique<MeshRemovedListener>(this);
        std::shared_ptr<EventListener> onInstancedMeshRemovedListener = std::make_unique<InstancedMeshRemovedListener>(this);
    };

}// namespace threepp

#endif//THREEPP_AGXPHYSICS_HPP
