
#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP

#include <threepp/objects/Group.hpp>
#include <threepp/objects/Mesh.hpp>

#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>

#include <memory>


namespace {
    btVector3 convert(const threepp::Vector3 &v) {
        return {v.x, v.y, v.z};
    }

    btQuaternion convert(const threepp::Quaternion &q) {
        return {q.x(), q.y(), q.z(), q.w()};
    }

    btTransform convert(const threepp::Matrix4 &m) {
        return btTransform{
                convert(threepp::Quaternion().setFromRotationMatrix(m)),
                convert(threepp::Vector3().setFromMatrixPosition(m))};
    }
}

struct Rbinfo {
    std::unique_ptr<btCollisionShape> shape;
    std::unique_ptr<btMotionState> state;
    std::unique_ptr<btRigidBody> body;

    Rbinfo(std::unique_ptr<btCollisionShape> shape, float mass, const threepp::Matrix4 &origin)
        : shape(std::move(shape)),
          state(std::make_unique<btDefaultMotionState>()) {

        btVector3 inertia{};
        this->shape->calculateLocalInertia(mass, inertia);

        state->setWorldTransform(convert(origin));

        body = std::make_unique<btRigidBody>(mass, state.get(), this->shape.get(), inertia);
    }
};


class BulletEngine {

public:
    explicit BulletEngine(float gravity = -9.81f)
        : dispatcher{&collisionConfiguration},
          world{&dispatcher, &broadphase, &solver, &collisionConfiguration} {
        world.setGravity(btVector3(0, gravity, 0));
    }

    btRigidBody *registerMesh(const std::shared_ptr<threepp::Mesh> &m, float mass);
    btRigidBody *registerGroup(const std::shared_ptr<threepp::Group> &m, float mass);

    void addConstraint(btTypedConstraint* c);

    void step(float dt);


private:
    std::unordered_map<std::shared_ptr<threepp::Object3D>, std::unique_ptr<Rbinfo>> bodies{};

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
