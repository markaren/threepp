// Contact sensor truth.
//
// The assertions worth making are the ones that catch a wrong sign or a wrong
// lifetime: the normal must point INTO the sensor's body whichever order PhysX
// happened to put the pair in, the force must roughly balance the weight it is
// holding up, the latch must survive the pair falling asleep, and none of it may
// dangle when the other body is removed.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/ContactSensor.hpp"
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

    // A large static box as ground, top surface at y = 0.
    ::physx::PxRigidStatic* addGround(PhysxWorld& world) {
        using namespace ::physx;
        return world.addStatic(PxBoxGeometry(50.f, 1.f, 50.f),
                               PxTransform(PxVec3(0.f, -1.f, 0.f)));
    }

    void stepFor(PhysxWorld& world, int substeps) {
        for (int i = 0; i < substeps; ++i) world.step(kSubstep);
    }

    // Step until the sensor latches a touch, or give up. Returns substeps taken.
    int stepUntilContact(PhysxWorld& world, const ContactSensor& s, int maxSubsteps = 600) {
        for (int i = 0; i < maxSubsteps; ++i) {
            world.step(kSubstep);
            if (s.inContact()) return i + 1;
        }
        return -1;
    }

}// namespace


TEST_CASE("A resting body reports contact with an upward normal") {

    PhysxWorld world(fixedStep());
    addGround(world);

    constexpr float size = 0.4f;
    auto mesh = box(size, 0.f, 0.6f, 0.f);// starts just above the ground
    world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);
    REQUIRE(sensor.attached());
    REQUIRE_FALSE(sensor.inContact());

    const int steps = stepUntilContact(world, sensor);
    INFO("latched after " << steps << " substeps");
    REQUIRE(steps > 0);

    // Find a sample that actually carries manifold points (the landing substeps
    // do; once it settles and sleeps, PhysX stops re-reporting).
    std::vector<ContactSample> samples;
    stepFor(world, 30);
    sensor.drain(samples);

    const ContactSample* withPoints = nullptr;
    for (const auto& s: samples) {
        if (s.pointCount > 0) withPoints = &s;
    }
    REQUIRE(withPoints != nullptr);

    // The ground pushes the box UP, so every normal must point +Y regardless of
    // which slot PhysX put this body in.
    for (std::uint32_t i = 0; i < withPoints->pointCount; ++i) {
        const auto& p = withPoints->points[i];
        INFO("point " << i << " normal = " << p.normal.x << ", " << p.normal.y
                      << ", " << p.normal.z);
        REQUIRE(p.normal.y > 0.9f);
        REQUIRE_THAT(p.normal.length(), WithinAbs(1.f, 1e-3f));
        // And it sits on the ground plane, not somewhere random.
        REQUIRE_THAT(p.position.y, WithinAbs(0.f, 0.05f));
    }

    world.unregisterSensor(&sensor);
}

TEST_CASE("Contact force balances the weight it holds up") {

    // A box at rest on the ground is supported by exactly its own weight, so
    // the summed normal impulse over an interval divided by that interval must
    // come back to m*g. This is the assertion that catches an impulse read with
    // the wrong sign or divided by the wrong dt.
    PhysxWorld world(fixedStep());
    addGround(world);

    constexpr float size = 0.4f;
    constexpr float density = 1000.f;
    const float mass = size * size * size * density;// 64 kg

    auto mesh = box(size, 0.f, 0.6f, 0.f);
    auto* body = world.add(*mesh, density);
    CHECK_THAT(body->getMass(), WithinRel(mass, 0.01f));

    // Sleeping would stop the impulse stream; keep it awake for the measurement.
    body->setSleepThreshold(0.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);
    stepFor(world, 120);// let the landing transient settle
    std::vector<ContactSample> discard;
    sensor.drain(discard);

    stepFor(world, 60);
    std::vector<ContactSample> samples;
    sensor.drain(samples);
    REQUIRE(samples.size() == 60);

    // Average over the window: per-substep solver output is noisy, the mean is
    // what has to equal the weight.
    double sumFy = 0.0;
    int counted = 0;
    for (const auto& s: samples) {
        sumFy += s.force.y;
        ++counted;
    }
    const double meanFy = sumFy / counted;
    const double weight = static_cast<double>(mass) * kG;

    INFO("mean Fy = " << meanFy << " N, weight = " << weight << " N");
    CHECK(meanFy > 0.0);// pushing up, not down
    CHECK_THAT(meanFy, WithinRel(weight, 0.15));

    world.unregisterSensor(&sensor);
}

TEST_CASE("The latch survives the contact pair falling asleep") {

    // PhysX stops issuing TOUCH_PERSISTS once a resting pair sleeps, without a
    // TOUCH_LOST. A sensor that inferred "touching" from "got points this
    // substep" would report the box floating away while it sits there.
    PhysxWorld world(fixedStep());
    addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    auto* body = world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);

    // Run long enough for the body to settle and sleep.
    stepFor(world, 1200);// 5 seconds
    REQUIRE(body->isSleeping());

    CHECK(sensor.inContact());// latch held
    const auto s = sensor.latest();
    REQUIRE(s.has_value());
    CHECK(s->inContact);
    CHECK(s->pointCount == 0);// and the observation channel is honest about it

    world.unregisterSensor(&sensor);
}

TEST_CASE("A free-falling body reports no contact") {

    PhysxWorld world(fixedStep());// no ground at all

    auto mesh = box(0.4f, 0.f, 5.f, 0.f);
    world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    stepFor(world, 120);

    CHECK_FALSE(sensor.inContact());
    std::vector<ContactSample> samples;
    sensor.drain(samples);
    REQUIRE(samples.size() == 120);
    for (const auto& s: samples) {
        REQUIRE_FALSE(s.inContact);
        REQUIRE(s.pointCount == 0);
        REQUIRE_THAT(s.force.length(), WithinAbs(0.f, 1e-6f));
    }

    world.unregisterSensor(&sensor);
}

TEST_CASE("Touch begin and end are reported as edges") {

    PhysxWorld world(fixedStep());
    auto* ground = addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);

    std::vector<ContactSample> samples;
    sensor.drain(samples);
    int begins = 0;
    for (const auto& s: samples) {
        if (s.touchBegan) ++begins;
    }
    CHECK(begins >= 1);

    // Take the ground away: the touch must be reported lost.
    world.removeActor(ground);
    stepFor(world, 30);

    CHECK_FALSE(sensor.inContact());
    sensor.drain(samples);
    int ends = 0;
    for (const auto& s: samples) {
        if (s.touchEnded) ++ends;
    }
    CHECK(ends >= 1);

    world.unregisterSensor(&sensor);
}

TEST_CASE("The identity of the touched body is reported") {

    PhysxWorld world(fixedStep());
    auto* ground = addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);
    stepFor(world, 20);

    std::vector<ContactSample> samples;
    sensor.drain(samples);

    bool sawGround = false;
    for (const auto& s: samples) {
        for (std::uint32_t i = 0; i < s.pointCount; ++i) {
            REQUIRE(s.points[i].other != nullptr);
            if (s.points[i].other == ground) sawGround = true;
        }
    }
    CHECK(sawGround);

    world.unregisterSensor(&sensor);
}

TEST_CASE("Touching two bodies and losing one stays in contact") {

    // The case a single boolean latch gets wrong in both directions: a box with
    // the ground beneath it and another box stacked on top is touching two
    // things, so losing the top one must NOT read as "no longer touching
    // anything" — it is still very much standing on the ground.
    PhysxWorld world(fixedStep());
    auto* ground = addGround(world);

    auto middle = box(0.4f, 0.f, 0.6f, 0.f);
    world.add(*middle, 1000.f);

    auto top = box(0.4f, 0.f, 1.6f, 0.f);
    auto* topBody = world.add(*top, 1000.f);

    ContactSensor sensor(*middle);
    world.registerSensor(&sensor);

    // Let both land and settle into a stack.
    stepFor(world, 400);
    INFO("touching " << sensor.touchCount() << " bodies");
    REQUIRE(sensor.touchCount() == 2);// ground below, box above
    REQUIRE(sensor.inContact());

    // Launch the top box away. This produces a genuine TOUCH_LOST for the
    // middle/top pair while the middle/ground pair is untouched — the exact
    // sequence that made a single boolean latch report "not touching anything"
    // for a box that never left the floor.
    topBody->setLinearVelocity(::physx::PxVec3(0.f, 6.f, 0.f));
    stepFor(world, 60);

    CHECK(sensor.inContact());// still standing on the ground
    CHECK(sensor.touchCount() == 1);
    // It really did leave: it was resting at ~0.6 (on top of the middle box),
    // so anything up here is unambiguously separated.
    INFO("top box y = " << top->position.y);
    CHECK(top->position.y > 1.2f);

    // And now take the ground away too: nothing left to touch.
    world.removeActor(ground);
    CHECK_FALSE(sensor.inContact());
    CHECK(sensor.touchCount() == 0);

    world.unregisterSensor(&sensor);
}

TEST_CASE("One body touching through two shapes needs both to separate") {

    // Compound colliders are normal (a robot foot, a chassis), and PhysX reports
    // one pair PER SHAPE pair. Tracking touches per ACTOR alone is not enough:
    // when one shape of a two-shape foot lifts, that pair reports TOUCH_LOST
    // while the other is still planted, and erasing the actor on the first LOST
    // would call the foot airborne while it is still bearing load.
    PhysxWorld world(fixedStep());
    addGround(world);

    using namespace ::physx;
    constexpr float half = 0.2f;
    constexpr float arm = 0.6f;

    // A dumbbell: one rigid body, two box shapes offset along X.
    auto mesh = Mesh::create(BoxGeometry::create(0.1f, 0.1f, 0.1f), MeshBasicMaterial::create());
    auto* body = world.physics().createRigidDynamic(PxTransform(PxVec3(0.f, 0.5f, 0.f)));
    for (const float dx: {-arm, arm}) {
        auto* s = world.physics().createShape(PxBoxGeometry(half, half, half),
                                              world.defaultMaterial(), true);
        s->setLocalPose(PxTransform(PxVec3(dx, 0.f, 0.f)));
        body->attachShape(*s);
        s->release();
    }
    PxRigidBodyExt::updateMassAndInertia(*body, 1000.f);
    body->setSleepThreshold(0.f);
    world.scene().addActor(*body);
    world.bind(*mesh, *body);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);
    stepFor(world, 240);// settle flat, both feet down
    REQUIRE(sensor.inContact());
    REQUIRE(sensor.touchCount() == 1);// one BODY, touched through two shapes

    // Tip it geometrically rather than by hitting it — solver dynamics on a body
    // this stable are not worth guessing at. Rotating by `tilt` about Z lifts the
    // +X shape by arm*sin(tilt) and drops the -X shape by the same, so raising
    // the body by exactly that leaves -X where it was (planted, still touching)
    // and puts +X twice that height clear of the floor (separated).
    const float restY = body->getGlobalPose().p.y;
    constexpr float tilt = 0.12f;
    const float lift = arm * std::sin(tilt);
    body->setGlobalPose(PxTransform(PxVec3(0.f, restY + lift, 0.f), PxQuat(tilt, PxVec3(0, 0, 1))));

    // Geometry check: this really is one shape down and one shape up.
    INFO("restY=" << restY << " lift=" << lift);
    REQUIRE(lift > half * 0.25f);// +X clears by 2*lift, well beyond contact slop

    bool everLost = false;
    for (int i = 0; i < 20; ++i) {
        world.step(kSubstep);
        if (!sensor.inContact()) everLost = true;
    }

    // The -X shape never left the floor, so the sensor must never have gone
    // quiet — even though the +X shape's pair reported TOUCH_LOST.
    CHECK_FALSE(everLost);
    CHECK(sensor.inContact());
    CHECK(sensor.touchCount() == 1);

    world.unregisterSensor(&sensor);
}

TEST_CASE("Removing the sensor's own body does not dangle") {

    PhysxWorld world(fixedStep());
    addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    auto* body = world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);
    REQUIRE(stepUntilContact(world, sensor) > 0);
    REQUIRE(sensor.attached());

    world.removeActor(body);
    CHECK_FALSE(sensor.attached());

    stepFor(world, 30);// must not touch the released actor or its watcher
    CHECK_FALSE(sensor.inContact());

    world.unregisterSensor(&sensor);
}

TEST_CASE("Removing the touched body does not dangle") {

    // The other half: the sensor's body survives, the thing it was touching does
    // not. PhysX reports the pair with a REMOVED_ACTOR flag, which must not be
    // turned into a contact point against freed memory.
    PhysxWorld world(fixedStep());
    auto* ground = addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    world.add(*mesh, 1000.f);

    ContactSensor sensor(*mesh);
    world.registerSensor(&sensor);
    REQUIRE(stepUntilContact(world, sensor) > 0);

    world.removeActor(ground);
    stepFor(world, 60);

    CHECK(sensor.attached());// our own body is untouched by that removal
    CHECK_FALSE(sensor.inContact());

    world.unregisterSensor(&sensor);
}

TEST_CASE("Contact reporting is off unless asked for") {

    // The filter shader change must be inert for anyone not using it: a world
    // with no watcher gets no notifications, and collisions still happen.
    PhysxWorld world(fixedStep());
    addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    auto* body = world.add(*mesh, 1000.f);

    CHECK(world.contactDispatcher().empty());

    stepFor(world, 300);

    // It landed and stayed up — collision filtering is unchanged.
    const float y = body->getGlobalPose().p.y;
    INFO("resting y = " << y);
    CHECK(y > 0.f);
    CHECK(y < 0.5f);
    CHECK(world.contactDispatcher().empty());
}

TEST_CASE("A rate-gated contact sensor aggregates the interval") {

    PhysxWorld world(fixedStep());// 240 Hz
    addGround(world);

    auto mesh = box(0.4f, 0.f, 0.6f, 0.f);
    auto* body = world.add(*mesh, 1000.f);
    body->setSleepThreshold(0.f);

    ContactSensor sensor(*mesh, /*rateHz*/ 30.0);
    world.registerSensor(&sensor);

    REQUIRE(stepUntilContact(world, sensor) > 0);

    // Discard everything from the fall — the sensor has been emitting since
    // registration, so only the window below is 1 second long.
    std::vector<ContactSample> samples;
    sensor.drain(samples);

    stepFor(world, 240);// 1 second
    sensor.drain(samples);
    INFO("emitted " << samples.size() << " samples at 30 Hz");
    CHECK(samples.size() >= 29);
    CHECK(samples.size() <= 32);

    // Eight substeps of manifold folded into each reading, so the cap should be
    // hit and reported honestly rather than silently truncating.
    bool sawCappedReport = false;
    for (const auto& s: samples) {
        REQUIRE(s.pointCount <= ContactSample::maxPoints);
        REQUIRE(s.observedPoints >= s.pointCount);
        if (s.observedPoints > s.pointCount) sawCappedReport = true;
    }
    CHECK(sawCappedReport);

    world.unregisterSensor(&sensor);
}

TEST_CASE("Registering a contact sensor with no rigid body throws") {

    PhysxWorld world(fixedStep());
    auto orphan = Object3D::create();
    ContactSensor sensor(*orphan);

    CHECK_THROWS_AS(world.registerSensor(&sensor), std::invalid_argument);
    CHECK_FALSE(sensor.attached());
}
