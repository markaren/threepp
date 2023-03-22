
#ifndef THREEPP_AGXPHYSICS_HPP
#define THREEPP_AGXPHYSICS_HPP


#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxSDK/Simulation.h>

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
                return new agxCollide::Geometry(new agxCollide::Box(g->width, g->height, g->depth));

            } else if (geometry.type() == "PlaneGeometry") {

                auto g = dynamic_cast<PlaneGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Box(g->width, 0.1f, g->height));

            } else if (geometry.type() == "CapsuleGeometry") {

                auto g = dynamic_cast<CapsuleGeometry*>(&geometry);
                return new agxCollide::Geometry(new agxCollide::Capsule(g->radius, g->length));

            }

            return nullptr;
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

//                    handleInstancedMesh(mesh.as<InstancedMesh>(), mass, std::move(shape));

                } else if (mesh.is<Mesh>()) {

                    handleMesh(&mesh, mass, geometry);
                }
            }
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
            }
        }

    private:
        float t = 0;
        agx::AutoInit init;
        agxSDK::SimulationRef sim = new agxSDK::Simulation();

        std::unordered_map<Object3D*, agx::RigidBodyRef> meshMap;


        void handleMesh(Mesh* mesh, float mass, agxCollide::Geometry* geometry) {

            const auto& position = mesh->position;
            const auto& quaternion = mesh->quaternion;

            agx::RigidBodyRef rb = new agx::RigidBody();
            if (mass == 0) {
                rb->setMotionControl(agx::RigidBody::STATIC);
            } else {
                rb->getMassProperties()->setMass(mass);
            }
            rb->setPosition(toVec3(position));
            rb->setRotation(toQuat(quaternion));
            rb->add(geometry);
            sim->add(rb);

            meshMap[mesh] = rb;

            mesh->addEventListener("remove", onMeshRemovedListener);
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

        std::shared_ptr<MeshRemovedListener> onMeshRemovedListener = std::make_unique<MeshRemovedListener>(this);
    };

}// namespace threepp

#endif//THREEPP_AGXPHYSICS_HPP
