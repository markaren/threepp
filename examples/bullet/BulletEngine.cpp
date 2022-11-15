
#include "BulletEngine.hpp"

#include <bullet/btBulletDynamicsCommon.h>

#include <threepp/utils/InstanceOf.hpp>
#include <threepp/geometries/geometries.hpp>

#include <unordered_map>

using namespace threepp;

namespace {
    btVector3 convert(const threepp::Vector3 &v) {
        return {v.x, v.y, v.z};
    }

    btQuaternion convert(const threepp::Quaternion &q) {
        return {q.x(), q.y(), q.z(), q.w()};
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

            btTransform t{};
            t.setIdentity();
            t.setOrigin(convert(Vector3().setFromMatrixPosition(origin)));
            t.setRotation(convert(Quaternion().setFromRotationMatrix(origin)));
            state->setWorldTransform(t);

            body = std::make_unique<btRigidBody>(mass, state.get(), this->shape.get(), inertia);
        }
    };
}// namespace

struct BulletEngine::Impl {

    Impl()
        : dispatcher{&collisionConfiguration},
          world{&dispatcher, &broadphase, &solver, &collisionConfiguration} {
        world.setGravity(btVector3(0, -9.81, 0));
    }

    void register_mesh(std::shared_ptr<Mesh> m, float mass) {

        auto geometry = m->geometry();

        m->updateMatrixWorld();
        m->matrixAutoUpdate = false;

        if (instanceof <const BoxGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<BoxGeometry>(geometry);

            auto shape = std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);
            auto rbInfo = std::make_unique<Rbinfo>(std::move(shape), mass, *m->matrixWorld);

            world.addRigidBody(rbInfo->body.get());
            bodies[m] = std::move(rbInfo);

        } else if (instanceof <const SphereGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<threepp::SphereGeometry>(geometry);

            auto shape = std::make_unique<btSphereShape>(g->radius);
            auto rbInfo = std::make_unique<Rbinfo>(std::move(shape), mass, *m->matrixWorld);

            world.addRigidBody(rbInfo->body.get());
            bodies[m] = std::move(rbInfo);
        }
    }

    void step(float dt) {

        world.stepSimulation(dt);

        for (auto &[m, info] : bodies) {
            auto t = info->body->getWorldTransform();
            auto p = t.getOrigin();
            auto r = t.getRotation();

            m->matrix->makeRotationFromQuaternion(Quaternion(r.x(), r.y(), r.z(), r.w()));
            m->matrix->setPosition(Vector3(p.x(), p.y(), p.z()));

        }
    }

    ~Impl() = default;


private:
    std::unordered_map<std::shared_ptr<Mesh>, std::unique_ptr<Rbinfo>> bodies{};

    btDbvtBroadphase broadphase{};

    ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
    btSequentialImpulseConstraintSolver solver{};

    ///collision configuration contains default setup for memory, collision setup
    btDefaultCollisionConfiguration collisionConfiguration{};

    ///use the default collision dispatcher. For parallel processing you can use a different dispatcher (see Extras/BulletMultiThreaded)
    btCollisionDispatcher dispatcher;

    btDiscreteDynamicsWorld world;
};

BulletEngine::BulletEngine()
    : pimpl_(std::make_unique<Impl>()) {}

void BulletEngine::register_mesh(std::shared_ptr<Mesh> m, float mass) {
    pimpl_->register_mesh(std::move(m), mass);
}

void BulletEngine::step(float dt) {
    pimpl_->step(dt);
}

BulletEngine::~BulletEngine() = default;
