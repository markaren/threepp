
#ifndef THREEPP_BULLETENGINE_HPP
#define THREEPP_BULLETENGINE_HPP

#include <threepp/geometries/geometries.hpp>
#include <threepp/objects/Group.hpp>
#include <threepp/objects/Mesh.hpp>

#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>

#include <memory>

namespace threepp {

    inline btVector3 convert(const threepp::Vector3& v) {
        return {v.x, v.y, v.z};
    }

    inline btQuaternion convert(const threepp::Quaternion& q) {
        return {q.x(), q.y(), q.z(), q.w()};
    }

    inline btTransform convert(const threepp::Matrix4& m) {
        return btTransform{
                convert(threepp::Quaternion().setFromRotationMatrix(m)),
                convert(threepp::Vector3().setFromMatrixPosition(m))};
    }

    inline std::unique_ptr<btCollisionShape> fromGeometry(const BufferGeometry* geometry) {

        if (!geometry) {
            return std::make_unique<btEmptyShape>();
        }

        if (dynamic_cast<const BoxGeometry*>(geometry)) {

            auto g = dynamic_cast<const BoxGeometry*>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width, g->height, g->depth) / 2);

        } else if (dynamic_cast<const PlaneGeometry*>(geometry)) {

            auto g = dynamic_cast<const PlaneGeometry*>(geometry);
            return std::make_unique<btBoxShape>(btVector3(g->width / 2, 0.1f, g->height / 2));

        } else if (dynamic_cast<const SphereGeometry*>(geometry)) {

            auto g = dynamic_cast<const SphereGeometry*>(geometry);
            return std::make_unique<btSphereShape>(g->radius);

        } else if (dynamic_cast<const ConeGeometry*>(geometry)) {

            auto g = dynamic_cast<const ConeGeometry*>(geometry);
            return std::make_unique<btConeShape>(g->radiusBottom, g->height);

        } else if (dynamic_cast<const CylinderGeometry*>(geometry)) {

            auto g = dynamic_cast<const CylinderGeometry*>(geometry);
            return std::make_unique<btCylinderShape>(btVector3(g->radiusTop, g->height / 2, g->radiusTop));

        } else if (dynamic_cast<const CapsuleGeometry*>(geometry)) {

            auto g = dynamic_cast<const CapsuleGeometry*>(geometry);
            return std::make_unique<btCapsuleShape>(g->radius, g->length);

        } else {

            return std::make_unique<btEmptyShape>();
        }
    }


    struct RbWrapper {
        std::unique_ptr<btCollisionShape> shape;
        std::unique_ptr<btMotionState> state;
        std::unique_ptr<btRigidBody> body;

        static std::shared_ptr<RbWrapper> create(const BufferGeometry* shape, float mass = 0) {
            return std::shared_ptr<RbWrapper>(new RbWrapper(shape, mass));
        }

    private:
        RbWrapper(const BufferGeometry* shape, float mass)
            : shape(fromGeometry(shape)),
              state(std::make_unique<btDefaultMotionState>()) {

            btVector3 inertia{1, 1, 1};
            if (shape) {
                this->shape->calculateLocalInertia(mass, inertia);
            } else {
                inertia *= mass;
            }

            body = std::make_unique<btRigidBody>(mass, state.get(), this->shape.get(), inertia);
            body->setActivationState(DISABLE_DEACTIVATION);
        }
    };


    class BulletWrapper {

    public:
        explicit BulletWrapper(const Vector3& gravity = {0, -9.81f, 0})
            : dispatcher{&collisionConfiguration},
              world{&dispatcher, &broadphase, &solver, &collisionConfiguration},
              listener(std::make_shared<ObjectRemovedListener>(this)) {

            world.setGravity(convert(gravity));
        }

        void step(float dt, int maxSubSteps = 1, btScalar fixedTimeStep = btScalar(1.) / btScalar(60.)) {
            world.stepSimulation(dt, maxSubSteps, fixedTimeStep);

            for (auto& [m, info] : bodies) {
                auto& t = info->body->getWorldTransform();
                auto& p = t.getOrigin();
                auto r = t.getRotation();

                m->quaternion.set(r.x(), r.y(), r.z(), r.w());
                m->position.set(p.x(), p.y(), p.z());
            }
        }

        BulletWrapper& setGravity(const Vector3& g) {
            world.setGravity(convert(g));
            return *this;
        }

        BulletWrapper& addRigidbody(const std::shared_ptr<RbWrapper>& rb, Object3D& obj) {

            obj.updateMatrixWorld();
            auto t = convert(*obj.matrixWorld);
            rb->state->setWorldTransform(t);
            rb->body->setWorldTransform(t);

            world.addRigidBody(rb->body.get());
            bodies[&obj] = rb;

            obj.addEventListener("remove", listener);

            return *this;
        }

        void addConstraint(btTypedConstraint* c, bool disableCollisionsBetweenLinkedBodies = false) {
            world.addConstraint(c, disableCollisionsBetweenLinkedBodies);
        }

    private:
        struct ObjectRemovedListener: EventListener {

            explicit ObjectRemovedListener(BulletWrapper* scope): scope(scope) {}

            void onEvent(Event& event) override {
                if (event.type == "remove") {
                    auto o = static_cast<Object3D*>(event.target);
                    if (scope->bodies.count(o)) {
                        auto& rb = scope->bodies.at(o);
                        scope->world.removeRigidBody(rb->body.get());
                        scope->bodies.erase(o);
                    }
                }
            }

        private:
            BulletWrapper* scope;
        };

        std::unordered_map<threepp::Object3D*, std::shared_ptr<RbWrapper>> bodies{};

        btDbvtBroadphase broadphase{};

        ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
        btSequentialImpulseConstraintSolver solver{};

        ///collision configuration contains default setup for memory, collision setup
        btDefaultCollisionConfiguration collisionConfiguration{};

        ///use the default collision dispatcher. For parallel processing you can use a different dispatcher (see Extras/BulletMultiThreaded)
        btCollisionDispatcher dispatcher;

        btDiscreteDynamicsWorld world;

        std::shared_ptr<ObjectRemovedListener> listener;
    };

}// namespace threepp

#endif//THREEPP_BULLETENGINE_HPP
