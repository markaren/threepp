// The cathedral in PhysX: one rigid body per plank, at real size and real
// gravity, on a floor.
//
// A design is proven before it is shown: every plank is kept awake for a
// couple of seconds of simulated time, and the largest distance any plank has
// moved from where the design put it must stay under a millimetre. Then every
// plank is put to sleep, so that a standing cathedral costs next to nothing,
// and a blow wakes only the planks it reaches, by contact.
//
// Millimetre bodies want millimetre tolerances: PhysX's tolerance scale is set
// to centimetres, contacts are made within a millimetre of touching, friction
// anchors are merged only within 5 mm. What the tall stacks need above all is
// a small step and many solver iterations: at 240 Hz a ring of 60 courses
// pops planks out; at 960 Hz with 24 TGS position iterations and the solver's
// stabilization, rings of 150-200 courses and walls of 90 settle to well under
// a millimetre. Without stabilization a ring of 150 courses falls. A whole
// cathedral of 30-40k planks, towers of 250 courses, walls leaning in as roofs,
// creeps 2-4 mm at its tops in 2 s at 24 iterations (the solver's give, not a
// fall: nothing moves further with more time); the proof takes 48. Once proven,
// the scene runs at 24: woken parts that still stand hold for the seconds of a
// collapse. At 480 Hz and 12 iterations a woken cathedral sags down on its own.

#ifndef PLANKCATHEDRAL_STABILITY_HPP
#define PLANKCATHEDRAL_STABILITY_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "Cathedral.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace planks {

    struct SettleResult {
        float maxDisplacement = 0; // metres, the most any plank moved
        int worst = -1;            // which plank
        int movedOverMm = 0;       // how many moved more than a millimetre
        double seconds = 0;        // wall time
        double msPerStep = 0;
    };

    // switches for trying the physics out (main.cpp --pgs, --nospec, --nopcm, --nostab, --tol, --offset, --iters)
    struct PhysicsTuning {
        bool tgs = true;
        bool speculative = true;
        float toleranceLength = 0.01f;
        float contactOffset = 0.001f;
        int positionIterations = 48;// the proof; the scene runs at --scene-iters
        bool stabilization = true;
        bool pcm = true;
    };
    inline PhysicsTuning tuning;

    class Physics {
    public:
        static constexpr float density = 470.f;      // pine, kg/m3: about 10 g a plank
        static constexpr float sleepEnergy = 2e-4f;  // mass-normalised: asleep below ~2 cm/s

        float dt;
        float time = 0;
        std::vector<::physx::PxRigidDynamic*> bodies;
        ::physx::PxRigidDynamic* ball = nullptr;
        ::physx::PxDistanceJoint* rope = nullptr;

        Physics(const Design& design, float step, unsigned threads)
            : dt(step), world_(settings(step, threads)) {

            using namespace ::physx;
            auto& physics = world_.physics();

            // A scene of our own on the world's physics and threads, so that the
            // solver's options can be chosen one by one.
            PxSceneDesc desc(physics.getTolerancesScale());
            desc.gravity = PxVec3(0, -9.81f, 0);
            desc.cpuDispatcher = &world_.dispatcher();
            desc.filterShader = PxDefaultSimulationFilterShader;
            desc.solverType = tuning.tgs ? PxSolverType::eTGS : PxSolverType::ePGS;
            if (tuning.pcm) desc.flags |= PxSceneFlag::eENABLE_PCM;
            if (tuning.stabilization) desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
            desc.bounceThresholdVelocity = 0.5f;
            desc.frictionOffsetThreshold = 0.004f;
            desc.frictionCorrelationDistance = 0.005f;
            scene_ = physics.createScene(desc);
            auto& scene = *scene_;

            wood_ = physics.createMaterial(0.5f, 0.45f, 0.05f);

            PxShape* floorShape = physics.createShape(PxPlaneGeometry(), *wood_, true);
            floorShape->setContactOffset(0.002f);
            floorShape->setRestOffset(0.f);
            auto* floor = physics.createRigidStatic(PxTransform(PxVec3(0, 0, 0), PxQuat(PxHalfPi, PxVec3(0, 0, 1))));
            floor->attachShape(*floorShape);
            floorShape->release();
            scene.addActor(*floor);

            // one shared box shape for every plank
            shape_ = physics.createShape(PxBoxGeometry(LEN / 2, THICK / 2, WIDE / 2), *wood_, false);
            shape_->setContactOffset(tuning.contactOffset);
            shape_->setRestOffset(0.f);

            bodies.reserve(design.planks.size());
            for (const auto& p : design.planks) {
                const PxTransform pose(PxVec3(p.x, courseY(p.course), p.z), PxQuat(-p.angle, PxVec3(0, 1, 0)));
                PxRigidDynamic* body = physics.createRigidDynamic(pose);
                body->attachShape(*shape_);
                PxRigidBodyExt::updateMassAndInertia(*body, density);
                body->setSolverIterationCounts(tuning.positionIterations, 1);
                body->setMaxDepenetrationVelocity(0.2f);
                body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, tuning.speculative);
                body->setSleepThreshold(sleepEnergy);
                body->setStabilizationThreshold(5e-5f);
                body->setAngularDamping(0.05f);
                bodies.push_back(body);
            }
            std::vector<PxActor*> actors(bodies.begin(), bodies.end());
            scene.addActors(actors.data(), static_cast<PxU32>(actors.size()));

            start_.reserve(bodies.size());
            for (auto* b : bodies) start_.push_back(b->getGlobalPose().p);
        }

        ~Physics() {
            if (rope) rope->release();
            if (scene_) scene_->release();
            if (shape_) shape_->release();
        }

        void step() {
            scene_->simulate(dt);
            scene_->fetchResults(true);
            time += dt;
        }

        // The rate and solver iterations of the scene once the design is
        // proven: what the standing parts still need, not what the proof took.
        void setRate(float step, int positionIterations) {
            dt = step;
            for (auto* b : bodies) b->setSolverIterationCounts(positionIterations, 1);
        }

        // Every plank awake for `seconds`; how far did they move?
        SettleResult settle(float seconds) {
            for (auto* b : bodies) {
                b->setSleepThreshold(0.f);
                b->wakeUp();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const int steps = static_cast<int>(std::lround(seconds / dt));
            for (int i = 0; i < steps; ++i) {
                for (auto* b : bodies) b->setWakeCounter(1.f);
                step();
            }
            const auto t1 = std::chrono::steady_clock::now();

            SettleResult r;
            r.seconds = std::chrono::duration<double>(t1 - t0).count();
            r.msPerStep = 1000.0 * r.seconds / std::max(1, steps);
            for (size_t i = 0; i < bodies.size(); ++i) {
                const float moved = (bodies[i]->getGlobalPose().p - start_[i]).magnitude();
                if (moved > 0.001f) ++r.movedOverMm;
                if (moved > r.maxDisplacement) {
                    r.maxDisplacement = moved;
                    r.worst = static_cast<int>(i);
                }
            }
            return r;
        }

        // Displacement of every plank from where the design put it.
        [[nodiscard]] std::vector<float> displacements() const {
            std::vector<float> out(bodies.size());
            for (size_t i = 0; i < bodies.size(); ++i) out[i] = (bodies[i]->getGlobalPose().p - start_[i]).magnitude();
            return out;
        }

        void sleepAll() {
            for (auto* b : bodies) {
                b->setSleepThreshold(sleepEnergy);
                b->putToSleep();
            }
        }

        [[nodiscard]] int awake() const {
            int n = 0;
            for (auto* b : bodies) n += b->isSleeping() ? 0 : 1;
            return n;
        }

        // A steel wrecking ball on a rope: a sphere hung from a fixed point by a
        // distance joint as long as the rope, let go from `start` at rest. The
        // rope only pulls; nothing else drives the ball.
        void hangBall(const threepp::Vector3& pivot, const threepp::Vector3& start, float radius) {
            using namespace ::physx;
            auto& physics = world_.physics();
            PxShape* s = physics.createShape(PxSphereGeometry(radius), *wood_, true);
            s->setContactOffset(0.004f);
            s->setRestOffset(0.f);
            ball = physics.createRigidDynamic(PxTransform(PxVec3(start.x, start.y, start.z)));
            ball->attachShape(*s);
            s->release();
            PxRigidBodyExt::updateMassAndInertia(*ball, 7850.f);
            ball->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
            ball->setSolverIterationCounts(16, 2);
            ball->setLinearDamping(0.f);
            ball->setAngularDamping(0.05f);
            scene_->addActor(*ball);
            const PxVec3 p(pivot.x, pivot.y, pivot.z);
            rope = PxDistanceJointCreate(physics, nullptr, PxTransform(p), ball, PxTransform(PxIdentity));
            rope->setMaxDistance((PxVec3(start.x, start.y, start.z) - p).magnitude());
            rope->setMinDistance(0.f);
            rope->setTolerance(0.f);
            rope->setDistanceJointFlag(PxDistanceJointFlag::eMAX_DISTANCE_ENABLED, true);
            rope->setDistanceJointFlag(PxDistanceJointFlag::eSPRING_ENABLED, false);
            ball->wakeUp();
        }

        // How many planks have moved further than `by` from where the design put them.
        [[nodiscard]] int movedMoreThan(float by) const {
            int n = 0;
            for (size_t i = 0; i < bodies.size(); ++i) n += (bodies[i]->getGlobalPose().p - start_[i]).magnitude() > by ? 1 : 0;
            return n;
        }

        // Every plank's pose, one line each: index, position, rotation (x y z w).
        void dump(const std::string& path) const {
            std::FILE* f = std::fopen(path.c_str(), "w");
            if (!f) return;
            for (size_t i = 0; i < bodies.size(); ++i) {
                const auto t = bodies[i]->getGlobalPose();
                std::fprintf(f, "%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", i, t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w);
            }
            std::fclose(f);
        }

        [[nodiscard]] ::physx::PxTransform pose(size_t i) const { return bodies[i]->getGlobalPose(); }

    private:
        static threepp::PhysxWorld::Settings settings(float step, unsigned threads) {
            threepp::PhysxWorld::Settings s;
            s.fixedTimestep = step;
            s.maxSubSteps = 1;
            s.numThreads = threads;
            s.enableTgsPcm = tuning.tgs;
            // planks are centimetres, and fall at a few metres a second at most
            s.toleranceLength = tuning.toleranceLength;
            s.toleranceSpeed = 1.f;
            return s;
        }

        threepp::PhysxWorld world_;
        ::physx::PxScene* scene_ = nullptr;
        ::physx::PxMaterial* wood_ = nullptr;
        ::physx::PxShape* shape_ = nullptr;
        std::vector<::physx::PxVec3> start_;
    };

}// namespace planks

#endif//PLANKCATHEDRAL_STABILITY_HPP
