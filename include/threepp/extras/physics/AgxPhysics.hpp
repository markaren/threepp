
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

            }

            return nullptr;
        }


    }// namespace detail

    class AgxPhysics {

    public:
        AgxPhysics() = default;

        explicit AgxPhysics(float timeStep) {

            sim->setTimeStep(timeStep);
        }

        void addMesh(Mesh& mesh, float mass = 0) {

            auto g = mesh.geometry();
            if (!g) return ;

            auto geometry = detail::getGeometry(*g);

            if (geometry) {

                if (mesh.is<InstancedMesh>()) {

//                    handleInstancedMesh(mesh.as<InstancedMesh>(), mass, std::move(shape));

                } else if (mesh.is<Mesh>()) {

//                    handleMesh(&mesh, mass, std::move(geometry));
                }
            }
        }

        void step(float dt) {
            t += dt;
            while (sim->getTimeStamp() + sim->getTimeStep() < t) {
                sim->stepForward();
            }
        }

    private:
        float t = 0;
        agx::AutoInit init;
        agxSDK::SimulationRef sim = new agxSDK::Simulation();
    };

}// namespace threepp

#endif//THREEPP_AGXPHYSICS_HPP
