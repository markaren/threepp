
#include "BulletEngine.hpp"

#include <bullet/btBulletDynamicsCommon.h>

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

            btTransform t{convert(Quaternion().setFromRotationMatrix(origin)), convert(Vector3().setFromMatrixPosition(origin))};
            state->setWorldTransform(t);

            body = std::make_unique<btRigidBody>(mass, state.get(), this->shape.get(), inertia);
        }
    };

    std::unique_ptr<Rbinfo> createFromMesh(const Mesh &m, float mass) {
        auto geometry = m.geometry();
        if (std::dynamic_pointer_cast<const BoxGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<const BoxGeometry>(geometry);

            auto shape = std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);
            return std::make_unique<Rbinfo>(std::move(shape), mass, m.matrixWorld);

        } else if (std::dynamic_pointer_cast <const PlaneGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<const PlaneGeometry>(geometry);

            auto shape = std::make_unique<btBoxShape>(btVector3(g->width / 2, 0.01, g->height / 2));
            return std::make_unique<Rbinfo>(std::move(shape), mass, m.matrixWorld);
        } else if (std::dynamic_pointer_cast <const SphereGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<const SphereGeometry>(geometry);

            auto shape = std::make_unique<btSphereShape>(g->radius);
            return std::make_unique<Rbinfo>(std::move(shape), mass, m.matrixWorld);

        } else if (std::dynamic_pointer_cast <const CylinderGeometry>(geometry)) {
            auto g = std::dynamic_pointer_cast<const CylinderGeometry>(geometry);

            auto shape = std::make_unique<btCylinderShape>(btVector3(g->radiusTop, g->height/2, g->radiusTop));
            return std::make_unique<Rbinfo>(std::move(shape), mass, m.matrixWorld);
        }

        return nullptr;
    }

}// namespace

struct BulletEngine::Impl {

    explicit Impl(float gravity)
        : dispatcher{&collisionConfiguration},
          world{&dispatcher, &broadphase, &solver, &collisionConfiguration} {
        world.setGravity(btVector3(0, gravity, 0));
    }

    void register_mesh(const std::shared_ptr<Mesh> &m, float mass) {

        auto geometry = m->geometry();

        m->updateMatrixWorld();

        auto rb = createFromMesh(*m, mass);

        if (rb) {
            world.addRigidBody(rb->body.get());
            bodies[m] = std::move(rb);
        }
    }

    void step(float dt) {

        world.stepSimulation(dt);

        for (auto &[m, info] : bodies) {
            auto t = info->body->getWorldTransform();
            auto p = t.getOrigin();
            auto r = t.getRotation();

            m->quaternion.set(r.x(), r.y(), r.z(), r.w());
            m->position.set(p.x(), p.y(), p.z());
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

BulletEngine::BulletEngine(float gravity)
    : pimpl_(std::make_unique<Impl>(gravity)) {}

void BulletEngine::register_mesh(std::shared_ptr<Mesh> m, float mass) {
    pimpl_->register_mesh(std::move(m), mass);
}

void BulletEngine::step(float dt) {
    pimpl_->step(dt);
}

BulletEngine::~BulletEngine() = default;
