// Force/torque sensor truth.
//
// A locked joint holding a horizontal arm out against gravity is a statics
// problem with a known answer: the joint must transmit a force equal to the
// arm's weight and a torque equal to weight times lever arm. Those two numbers
// are asserted as MAGNITUDES, which are frame-independent — the wrench itself is
// reported in the joint's child frame, and pinning component signs would be
// testing PhysX's frame convention rather than the sensor.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/Joint.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    constexpr float kSubstep = 1.f / 240.f;
    constexpr float kG = 9.81f;

    PhysxWorld::Settings fixedStep() {
        PhysxWorld::Settings s;
        s.fixedTimestep = kSubstep;
        s.maxSubSteps = 1;
        return s;
    }

    std::shared_ptr<Mesh> box(float size, float x, float y, float z) {
        auto m = Mesh::create(BoxGeometry::create(size, size, size), MeshBasicMaterial::create());
        m->position.set(x, y, z);
        return m;
    }

    // A cantilever: fixed root at the origin, one arm link held out horizontally
    // at distance `arm` on a joint locked to zero travel. Statics then says the
    // joint carries the arm's full weight and weight*arm of torque.
    struct Cantilever {
        std::unique_ptr<Articulation> art;
        std::shared_ptr<Mesh> rootMesh, armMesh;
        ArticulationLink root{nullptr, nullptr, nullptr, ::physx::PxTransform(::physx::PxIdentity)};
        ArticulationLink link{nullptr, nullptr, nullptr, ::physx::PxTransform(::physx::PxIdentity)};
        float mass = 0.f;
        float arm = 0.f;
    };

    Cantilever makeCantilever(PhysxWorld& world, float armLen = 1.f, float size = 0.2f,
                              float density = 1000.f) {
        Cantilever c;
        c.arm = armLen;
        c.mass = size * size * size * density;

        c.art = std::make_unique<Articulation>(world, /*fixedBase*/ true,
                                               /*solverPositionIters*/ 32,
                                               /*disableSelfCollision*/ true);
        c.rootMesh = box(size, 0.f, 0.f, 0.f);
        c.armMesh = box(size, armLen, 0.f, 0.f);

        c.root = c.art->addLink(nullptr, *c.rootMesh, density, {0, 0, 1}, {0, 0, 0},
                                false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
        // limited with a zero-width range = locked: the joint cannot rotate, so
        // the whole static load goes through it.
        c.link = c.art->addLink(&c.root, *c.armMesh, density, {0, 0, 1}, {0, 0, 0},
                                /*limited*/ true, 0.f, 0.f,
                                /*stiffness*/ 0.f, /*damping*/ 0.f, /*maxForce*/ 0.f,
                                /*driveTarget*/ 0.f, "revolute", 0.f, nullptr);
        c.art->finalize();
        return c;
    }

    void stepFor(PhysxWorld& world, int substeps) {
        for (int i = 0; i < substeps; ++i) world.step(kSubstep);
    }

    // Build a cantilever in its own world, settle it, and return the measured
    // wrench. Comparison tests must run one world at a time: PhysX permits only
    // one PxFoundation per process, so two live PhysxWorlds fail to construct.
    WrenchSample measureCantilever(float armLen, float size = 0.2f, float density = 1000.f) {
        PhysxWorld world(fixedStep());
        auto c = makeCantilever(world, armLen, size, density);

        ForceTorqueSensor ft(*c.armMesh, *c.art, c.link);
        world.registerSensor(&ft);
        stepFor(world, 240);

        const auto s = ft.latest();
        world.unregisterSensor(&ft);
        REQUIRE(s.has_value());
        return *s;
    }

}// namespace


TEST_CASE("A locked joint carries the weight of what it holds") {

    PhysxWorld world(fixedStep());
    auto c = makeCantilever(world);

    ForceTorqueSensor ft(*c.armMesh, *c.art, c.link);
    world.registerSensor(&ft);

    stepFor(world, 240);// let the solver settle into the static solution

    const auto s = ft.latest();
    REQUIRE(s.has_value());

    const float weight = c.mass * kG;
    INFO("mass=" << c.mass << " kg, weight=" << weight << " N");
    INFO("|F|=" << s->force.length() << " N, |T|=" << s->torque.length() << " N*m");

    CHECK_THAT(s->force.length(), WithinRel(weight, 0.1f));
    CHECK_THAT(s->torque.length(), WithinRel(weight * c.arm, 0.1f));

    world.unregisterSensor(&ft);
}

TEST_CASE("Torque scales with the lever arm") {

    // Same weight, twice the reach: the force is unchanged and the torque
    // doubles. This is the assertion that catches a wrench read from the wrong
    // link, or a force/torque swap.
    const auto shortArm = measureCantilever(/*armLen*/ 1.f);
    const auto longArm = measureCantilever(/*armLen*/ 2.f);

    INFO("short |F|=" << shortArm.force.length() << " |T|=" << shortArm.torque.length());
    INFO("long  |F|=" << longArm.force.length() << " |T|=" << longArm.torque.length());

    CHECK_THAT(longArm.force.length(), WithinRel(shortArm.force.length(), 0.1f));
    CHECK_THAT(longArm.torque.length(), WithinRel(2.f * shortArm.torque.length(), 0.1f));
}

TEST_CASE("A heavier payload raises the measured load proportionally") {

    const auto light = measureCantilever(1.f, 0.2f, /*density*/ 1000.f);
    const auto heavy = measureCantilever(1.f, 0.2f, /*density*/ 3000.f);

    INFO("light |F|=" << light.force.length() << ", heavy |F|=" << heavy.force.length());
    CHECK_THAT(heavy.force.length(), WithinRel(3.f * light.force.length(), 0.1f));
}

TEST_CASE("A weightless world puts no load on the joint") {

    // The control: same rig, no gravity, so there is nothing to hold up. Without
    // this, an assertion like "|F| is about the weight" could be satisfied by a
    // sensor that reports some unrelated constant.
    PhysxWorld::Settings s = fixedStep();
    s.gravity.set(0.f, 0.f, 0.f);
    PhysxWorld world(s);

    auto c = makeCantilever(world);
    ForceTorqueSensor ft(*c.armMesh, *c.art, c.link);
    world.registerSensor(&ft);

    stepFor(world, 240);

    const auto sample = ft.latest();
    REQUIRE(sample.has_value());
    INFO("|F|=" << sample->force.length() << " |T|=" << sample->torque.length());
    CHECK_THAT(sample->force.length(), WithinAbs(0.f, 1.f));
    CHECK_THAT(sample->torque.length(), WithinAbs(0.f, 1.f));

    world.unregisterSensor(&ft);
}

TEST_CASE("Wrench readings are stamped and buffered like every other sensor") {

    PhysxWorld world(fixedStep());
    auto c = makeCantilever(world);

    ForceTorqueSensor ft(*c.armMesh, *c.art, c.link);
    world.registerSensor(&ft);

    stepFor(world, 24);

    std::vector<WrenchSample> samples;
    ft.drain(samples);
    REQUIRE(samples.size() == 24);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        INFO("sample " << i);
        REQUIRE_THAT(samples[i].t, WithinAbs(static_cast<double>(i + 1) * kSubstep, 1e-9));
    }
    CHECK(ft.available() == 0);

    world.unregisterSensor(&ft);
}

TEST_CASE("A rate-gated F/T sensor runs on its own cadence") {

    PhysxWorld world(fixedStep());// 240 Hz
    auto c = makeCantilever(world);

    ForceTorqueSensor ft(*c.armMesh, *c.art, c.link, /*rateHz*/ 40.0);
    world.registerSensor(&ft);

    stepFor(world, 240);// 1 second

    std::vector<WrenchSample> samples;
    ft.drain(samples);
    INFO("emitted " << samples.size() << " samples at 40 Hz");
    CHECK(samples.size() >= 39);
    CHECK(samples.size() <= 41);

    world.unregisterSensor(&ft);
}

TEST_CASE("F/T noise is reproducible for a fixed seed") {

    const auto run = [] {
        PhysxWorld world(fixedStep());
        auto c = makeCantilever(world);

        ForceTorqueSensor ft(*c.armMesh, *c.art, c.link);
        ft.forceNoise.whiteNoiseDensity.set(0.5f, 0.5f, 0.5f);
        ft.forceNoise.seed = 0xFEEDFACE;
        ft.torqueNoise.constantBias.set(0.1f, 0.f, 0.f);
        ft.reset();
        world.registerSensor(&ft);

        stepFor(world, 60);
        std::vector<WrenchSample> out;
        ft.drain(out);
        world.unregisterSensor(&ft);
        return out;
    };

    const auto a = run();
    const auto b = run();
    REQUIRE(a.size() == 60);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        INFO("sample " << i);
        REQUIRE(a[i].force.x == b[i].force.x);
        REQUIRE(a[i].force.y == b[i].force.y);
        REQUIRE(a[i].torque.x == b[i].torque.x);
    }
}

TEST_CASE("A load cell across a breaking weld records the failure load, then zero") {

    // A crate welded to the world with nothing under it, the weld armed to
    // fail under the crate's own weight. The load that snapped it is a known
    // number — the weight — so the break sample can be pinned against physics
    // rather than against the implementation.
    //
    // Deliberately NO break subscription here: this is the standalone path,
    // where the failure load is recovered by Joint's lazy latch (the first
    // post-break read of PxConstraint::getForce, whose buffer freezes on the
    // breaking step's value) rather than handed over by the world's callback.
    PhysxWorld world(fixedStep());
    auto crate = box(0.2f, 0.f, 3.f, 0.f);
    auto* body = world.add(*crate, /*density*/ 1000.f);// 8 kg -> 78.5 N of weight

    const float weight = 0.2f * 0.2f * 0.2f * 1000.f * kG;

    Joint::Params params;
    params.breakForce = weight * 0.5f;// gravity alone snaps it, first substep
    params.breakTorque = 1e9f;
    Joint weld(world, body, nullptr,
               ::physx::PxTransform(::physx::PxVec3(0.f, 3.f, 0.f)), params);

    ForceTorqueSensor ft(*crate, weld);
    world.registerSensor(&ft);

    stepFor(world, 24);
    REQUIRE(weld.broken());

    std::vector<WrenchSample> samples;
    ft.drain(samples);
    REQUIRE(samples.size() == 24);

    // The first sample is the breaking substep's, carrying the true failure
    // load: past the threshold that armed the break, equal to the weight the
    // weld was holding when it gave way.
    INFO("break sample |F|=" << samples[0].force.length() << " N, weight=" << weight << " N");
    CHECK(samples[0].force.length() >= params.breakForce);
    CHECK_THAT(samples[0].force.length(), WithinRel(weight, 0.1f));
    // Every sample after it reads zero: a broken joint transmits nothing.
    // (Without the latch-and-zero this stream would repeat the failure load
    // forever — getForce's buffer freezes, it does not clear.)
    for (std::size_t i = 1; i < samples.size(); ++i) {
        INFO("sample " << i);
        REQUIRE_THAT(samples[i].force.length(), WithinAbs(0.f, 1e-4f));
        REQUIRE_THAT(samples[i].torque.length(), WithinAbs(0.f, 1e-4f));
    }

    // The joint keeps both answers apart: breakWrench is the failure load for
    // the rest of its life, reactionForce is honestly zero.
    Vector3 force, torque;
    weld.breakWrench(force, torque);
    CHECK_THAT(force.length(), WithinRel(weight, 0.1f));
    weld.reactionForce(force, torque);
    CHECK_THAT(force.length(), WithinAbs(0.f, 1e-4f));

    world.unregisterSensor(&ft);
}

TEST_CASE("The break event carries the breaking step's wrench") {

    // The callback is the one point where the failure load is DEFINED (it
    // fires inside fetchResults of the breaking substep, solver results
    // current); everything downstream latches from here. Same rig as above,
    // this time listening.
    PhysxWorld world(fixedStep());
    auto crate = box(0.2f, 0.f, 3.f, 0.f);
    auto* body = world.add(*crate, /*density*/ 1000.f);

    const float weight = 0.2f * 0.2f * 0.2f * 1000.f * kG;

    Joint::Params params;
    params.breakForce = weight * 0.5f;
    params.breakTorque = 1e9f;
    Joint weld(world, body, nullptr,
               ::physx::PxTransform(::physx::PxVec3(0.f, 3.f, 0.f)), params);

    bool fired = false;
    Vector3 eventForce;
    const auto watch = world.watchConstraintBreaks([&](const ConstraintBreakEvent& event) {
        if (event.joint != weld.raw()) return;
        fired = true;
        eventForce.copy(event.force);
    });

    stepFor(world, 24);
    REQUIRE(fired);

    INFO("event |F|=" << eventForce.length() << " N, weight=" << weight << " N");
    CHECK(eventForce.length() >= params.breakForce);
    CHECK_THAT(eventForce.length(), WithinRel(weight, 0.1f));

    world.unwatchConstraintBreaks(watch);
}

TEST_CASE("Attaching an F/T sensor to the root link throws") {

    PhysxWorld world(fixedStep());
    auto c = makeCantilever(world);

    // The root has no incoming joint, so PhysX reports a zero wrench for it —
    // silently useless. Reject it at construction instead.
    CHECK_THROWS_AS(ForceTorqueSensor(*c.rootMesh, *c.art, c.root), std::invalid_argument);
}

TEST_CASE("Registering before finalize throws") {

    PhysxWorld world(fixedStep());

    Articulation art(world, true, 8, true);
    auto rootMesh = box(0.2f, 0.f, 0.f, 0.f);
    auto armMesh = box(0.2f, 1.f, 0.f, 0.f);
    auto root = art.addLink(nullptr, *rootMesh, 1000.f, {0, 0, 1}, {0, 0, 0},
                            false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
    auto link = art.addLink(&root, *armMesh, 1000.f, {0, 0, 1}, {0, 0, 0},
                            false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
    // deliberately NOT finalized

    ForceTorqueSensor ft(*armMesh, art, link);
    CHECK_THROWS_AS(world.registerSensor(&ft), std::runtime_error);
}
