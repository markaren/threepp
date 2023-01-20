
#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP

#include <threepp/geometries/geometries.hpp>
#include <threepp/objects/Group.hpp>
#include <threepp/objects/Mesh.hpp>

#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>

#include <memory>

namespace threepp {

    inline btVector3 convert(const threepp::Vector3 &v) {
        return {v.x, v.y, v.z};
    }

    inline btQuaternion convert(const threepp::Quaternion &q) {
        return {q.x(), q.y(), q.z(), q.w()};
    }

    inline btTransform convert(const threepp::Matrix4 &m) {
        return btTransform{
                convert(threepp::Quaternion().setFromRotationMatrix(m)),
                convert(threepp::Vector3().setFromMatrixPosition(m))};
    }

    std::unique_ptr<btCollisionShape> fromGeometry(const std::shared_ptr<BufferGeometry> &geometry) {
        if (std::dynamic_pointer_cast<const BoxGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const BoxGeometry>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);

        } else if (std::dynamic_pointer_cast<const PlaneGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const PlaneGeometry>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width / 2, 0.01f, g->height / 2));

        } else if (std::dynamic_pointer_cast<const SphereGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const SphereGeometry>(geometry);
            return std::make_unique<btSphereShape>(g->radius);

        } else if (std::dynamic_pointer_cast<const CylinderGeometry>(geometry)) {

            auto g = std::dynamic_pointer_cast<const CylinderGeometry>(geometry);
            return std::make_unique<btCylinderShape>(btVector3(g->radiusTop, g->height / 2, g->radiusTop));
        }

        return std::make_unique<btEmptyShape>();
    }


    struct RbWrapper {
        std::unique_ptr<btCollisionShape> shape;
        std::unique_ptr<btMotionState> state;
        std::unique_ptr<btRigidBody> body;

        RbWrapper(std::unique_ptr<btCollisionShape> shape, float mass, const threepp::Matrix4 &origin = Matrix4())
            : shape(std::move(shape)),
              state(std::make_unique<btDefaultMotionState>()) {

            btVector3 inertia;
            this->shape->calculateLocalInertia(mass, inertia);

            state->setWorldTransform(convert(origin));

            body = std::make_unique<btRigidBody>(mass, state.get(), this->shape.get(), inertia);
            body->setActivationState(DISABLE_DEACTIVATION);
        }

    };


    class BulletWrapper {

    public:
        explicit BulletWrapper(float gravity = -9.81f)
            : dispatcher{&collisionConfiguration},
              world{&dispatcher, &broadphase, &solver, &collisionConfiguration} {
            //            world.setGravity(btVector3(0, gravity, 0));
        }

        void step(float dt, int maxSubSteps = 1, btScalar fixedTimeStep = btScalar(1.) / btScalar(60.)) {
            world.stepSimulation(dt, maxSubSteps, fixedTimeStep);
        }

        BulletWrapper &setGravity(const Vector3 &g) {
            world.setGravity(convert(g));
            return *this;
        }

        BulletWrapper &addRigidbody(btRigidBody *rb) {
            world.addRigidBody(rb);
            return *this;
        }

        BulletWrapper &addRigidbody(btRigidBody *rb, int group, int mask) {
            world.addRigidBody(rb, group, mask);
            return *this;
        }

        BulletWrapper &addRigidbody(const RbWrapper &rb) {
            return addRigidbody(rb.body.get());
        }

        BulletWrapper &addRigidbody(const RbWrapper &rb, int group, int mask) {
            return addRigidbody(rb.body.get(), group, mask);
        }

        //        btRigidBody *registerMesh(const std::shared_ptr<threepp::Mesh> &m, float mass);
        //        btRigidBody *registerGroup(const std::shared_ptr<threepp::Group> &m, float mass);


    private:
        //        std::unordered_map<std::shared_ptr<threepp::Object3D>, std::unique_ptr<Rbinfo>> bodies{};

        btDbvtBroadphase broadphase{};

        ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
        btSequentialImpulseConstraintSolver solver{};

        ///collision configuration contains default setup for memory, collision setup
        btDefaultCollisionConfiguration collisionConfiguration{};

        ///use the default collision dispatcher. For parallel processing you can use a different dispatcher (see Extras/BulletMultiThreaded)
        btCollisionDispatcher dispatcher;

        btDiscreteDynamicsWorld world;
    };

}// namespace threepp

#endif//THREEPP_BULLETENGINE_HPP
