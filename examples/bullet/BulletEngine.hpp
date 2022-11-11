

#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP

#include <bullet/btBulletDynamicsCommon.h>

#include <threepp/objects/Mesh.hpp>

#include <memory>
#include <unordered_map>

btVector3 convert(const threepp::Vector3 &v) {
    return {v.x, v.y, v.z};
}

struct Rbinfo {
    btCollisionShape* shape;
    btMotionState* state;
    btRigidBody* body;
};

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

            Rbinfo rbInfo;
            rbInfo.shape = new btBoxShape(btVector3(g->width, g->height, g->depth) / 2);
            rbInfo.state = new btDefaultMotionState();


            btTransform t;
            t.setOrigin(convert(m->position));
            rbInfo.state->setWorldTransform(t);

            rbInfo.body = new btRigidBody(mass, rbInfo.state, rbInfo.shape);
            world.addRigidBody(rbInfo.body);

            bodies[m] = rbInfo;
        }
    }

    void step(float dt) {

        world.stepSimulation(dt);

        for (auto &[m, rb] : bodies) {
            auto p = rb.body->getWorldTransform().getOrigin();
            m->position.set(p.x(), p.y(), std::isnan(p.z()) ? 0 : p.z());
        }
    }

private:
    btDiscreteDynamicsWorld world;

    btDbvtBroadphase broadphase = btDbvtBroadphase{};

    ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
    btSequentialImpulseConstraintSolver solver{};

    ///collision configuration contains default setup for memory, collision setup
    btDefaultCollisionConfiguration collisionConfiguration{};

    ///use the default collision dispatcher. For parallel processing you can use a different dispatcher (see Extras/BulletMultiThreaded)
    btCollisionDispatcher dispatcher;

    std::unordered_map<std::shared_ptr<threepp::Mesh>, Rbinfo> bodies{};
};

#endif//THREEPP_BULLETENGINE_HPP
