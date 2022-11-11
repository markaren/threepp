

#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP

#include <bullet/btBulletDynamicsCommon.h>

#include <threepp/objects/Mesh.hpp>

#include <memory>
#include <unordered_map>

namespace {
    btVector3 convert(const threepp::Vector3 &v) {
        return {v.x, v.y, v.z};
    }

    struct Rbinfo {
        std::unique_ptr<btCollisionShape> shape;
        std::unique_ptr<btMotionState> state;
        std::unique_ptr<btRigidBody> body;
    };
}

class BulletEngine {

public:
    BulletEngine()
        : dispatcher{&collisionConfiguration},
          world{&dispatcher, &broadphase, &solver, &collisionConfiguration} {
        world.setGravity(btVector3(0, -10, 0));
    }

    void register_mesh(std::shared_ptr<threepp::Mesh> m, btScalar mass) {

        auto geometry = m->geometry();

        if (std::dynamic_pointer_cast<const threepp::BoxGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<threepp::BoxGeometry>(geometry);

            auto rbInfo = std::make_unique<Rbinfo>();
            rbInfo->shape = std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);
            rbInfo->state = std::make_unique< btDefaultMotionState>();

            btTransform t{};
            t.setIdentity();
            t.setOrigin(convert(m->position));
            rbInfo->state->setWorldTransform(t);

            rbInfo->body = std::make_unique<btRigidBody>(mass, rbInfo->state.get(), rbInfo->shape.get());
            world.addRigidBody(rbInfo->body.get());

            bodies[m] = std::move(rbInfo);
        }
    }

    void step(float dt) {

        world.stepSimulation(dt);

        for (auto &[m, info] : bodies) {
            auto p = info->body->getWorldTransform().getOrigin();
            m->position.set(p.x(), p.y(), p.z());
        }
    }

private:

    std::unordered_map<std::shared_ptr<threepp::Mesh>, std::unique_ptr<Rbinfo>> bodies{};

    btDbvtBroadphase broadphase{};

    ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
    btSequentialImpulseConstraintSolver solver{};

    ///collision configuration contains default setup for memory, collision setup
    btDefaultCollisionConfiguration collisionConfiguration{};

    ///use the default collision dispatcher. For parallel processing you can use a different dispatcher (see Extras/BulletMultiThreaded)
    btCollisionDispatcher dispatcher;

    btDiscreteDynamicsWorld world;

};

#endif//THREEPP_BULLETENGINE_HPP
