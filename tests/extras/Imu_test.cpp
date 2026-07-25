// IMU physics truth.
//
// The IMU reports SPECIFIC FORCE, not acceleration, so the assertions here are
// the ones a real accelerometer has to satisfy: at rest it reads +g on its up
// axis, in free fall it reads ~0, and offset from the centre of mass it picks up
// the lever-arm terms. These are cheap to state and impossible to eyeball from a
// running scene, which is exactly why they belong in a test.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found (see
// tests/extras/CMakeLists.txt). The PhysX-free half of the sensor contract —
// PRNG, noise model, ring buffer, rate gate — is covered by Sensor_test.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    constexpr float kSubstep = 1.f / 240.f;
    constexpr float kG = 9.81f;

    // A cube: isotropic inertia, so a free spin about any axis stays about that
    // axis (no precession to confuse the gyro assertions).
    std::shared_ptr<Mesh> cube(float size = 1.f) {
        return Mesh::create(BoxGeometry::create(size, size, size), MeshBasicMaterial::create());
    }

    // World whose substep is exactly the dt we hand step(), so every step()
    // advances exactly one substep and the sample stream is deterministic.
    PhysxWorld::Settings fixedStep() {
        PhysxWorld::Settings s;
        s.fixedTimestep = kSubstep;
        s.maxSubSteps = 1;
        return s;
    }

    // A perfect sensor — the physics-truth assertions want the clean value.
    void makeNoiseless(Imu& imu) {
        imu.gyroNoise = NoiseModel{};
        imu.accelNoise = NoiseModel{};
        imu.reset();
    }

    void stepFor(PhysxWorld& world, int substeps) {
        for (int i = 0; i < substeps; ++i) world.step(kSubstep);
    }

}// namespace


TEST_CASE("An IMU at rest reads +g on its up axis") {

    // "At rest" in the accelerometer sense: proper acceleration zero while world
    // gravity is unchanged — a body sitting on a table, where the normal force
    // cancels gravity. Disabling gravity on the body models that without needing
    // a ground plane and contact to settle.
    PhysxWorld world(fixedStep());

    auto mesh = cube();
    auto* body = world.add(*mesh, 1000.f);
    body->setActorFlag(::physx::PxActorFlag::eDISABLE_GRAVITY, true);

    Imu imu(*mesh);
    makeNoiseless(imu);
    world.registerSensor(&imu);
    REQUIRE(imu.attached());

    stepFor(world, 10);

    const auto s = imu.latest();
    REQUIRE(s.has_value());
    CHECK_THAT(s->linearAcceleration.x, WithinAbs(0.f, 1e-3));
    CHECK_THAT(s->linearAcceleration.y, WithinAbs(kG, 1e-2));// threepp is Y-up
    CHECK_THAT(s->linearAcceleration.z, WithinAbs(0.f, 1e-3));

    CHECK_THAT(s->angularVelocity.length(), WithinAbs(0.f, 1e-4));

    world.unregisterSensor(&imu);
}

TEST_CASE("An IMU in free fall reads approximately zero") {

    PhysxWorld world(fixedStep());

    auto mesh = cube();
    auto* body = world.add(*mesh, 1000.f);
    body->setLinearDamping(0.f);

    Imu imu(*mesh);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 20);

    std::vector<ImuSample> samples;
    imu.drain(samples);
    REQUIRE(samples.size() == 20);

    // The first sample deliberately reports zero acceleration difference (prev
    // velocity := current, so there is no start-up impulse), which in free fall
    // reads as a full +g rather than 0. Documented behaviour — pin it.
    CHECK_THAT(samples.front().linearAcceleration.y, WithinAbs(kG, 1e-2));

    // Every later sample: proper acceleration IS g, so specific force is ~0.
    for (std::size_t i = 1; i < samples.size(); ++i) {
        INFO("sample " << i << " ay=" << samples[i].linearAcceleration.y);
        REQUIRE_THAT(samples[i].linearAcceleration.length(), WithinAbs(0.f, 1e-2));
    }

    // The body really was falling — otherwise the above would pass trivially.
    CHECK(body->getLinearVelocity().y < -0.5f);

    world.unregisterSensor(&imu);
}

TEST_CASE("An IMU measures the body's angular velocity") {

    PhysxWorld world(fixedStep());

    auto mesh = cube();
    auto* body = world.add(*mesh, 1000.f);
    body->setActorFlag(::physx::PxActorFlag::eDISABLE_GRAVITY, true);
    body->setAngularDamping(0.f);

    constexpr float omega = 3.f;// rad/s about Y
    body->setAngularVelocity(::physx::PxVec3(0.f, omega, 0.f));

    Imu imu(*mesh);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 30);

    const auto s = imu.latest();
    REQUIRE(s.has_value());
    CHECK_THAT(s->angularVelocity.x, WithinAbs(0.f, 1e-3));
    CHECK_THAT(s->angularVelocity.y, WithinRel(omega, 0.01f));
    CHECK_THAT(s->angularVelocity.z, WithinAbs(0.f, 1e-3));

    // Constant spin, sensor at the CoM: no angular acceleration and no lever
    // arm, so the proper acceleration is zero and the reading is pure -g. Note
    // that eDISABLE_GRAVITY makes the BODY unaccelerated, it does not change
    // world gravity — which is exactly the "supported on a table" case, so the
    // accelerometer still reads +g upward.
    CHECK_THAT(s->linearAcceleration.x, WithinAbs(0.f, 2e-2));
    CHECK_THAT(s->linearAcceleration.y, WithinAbs(kG, 2e-2));
    CHECK_THAT(s->linearAcceleration.z, WithinAbs(0.f, 2e-2));

    world.unregisterSensor(&imu);
}

TEST_CASE("An offset IMU picks up centripetal acceleration") {

    // The lever-arm term omega x (omega x r): a sensor r metres off the spin axis
    // reads omega^2*r pointing INWARD (its own -x, since the node's +x points
    // radially outward and stays there as the body turns).
    PhysxWorld world(fixedStep());

    auto mesh = cube();
    auto* body = world.add(*mesh, 1000.f);
    body->setActorFlag(::physx::PxActorFlag::eDISABLE_GRAVITY, true);
    body->setAngularDamping(0.f);

    constexpr float omega = 4.f;
    constexpr float r = 2.f;
    body->setAngularVelocity(::physx::PxVec3(0.f, omega, 0.f));

    // The sensor node rides the body, offset along its local +x.
    auto node = Object3D::create();
    node->position.set(r, 0.f, 0.f);
    mesh->add(node);
    mesh->updateMatrixWorld();

    Imu imu(*node);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 30);

    const auto s = imu.latest();
    REQUIRE(s.has_value());

    const float expected = omega * omega * r;// 32 m/s^2
    INFO("accel = " << s->linearAcceleration.x << ", " << s->linearAcceleration.y
                    << ", " << s->linearAcceleration.z);
    CHECK_THAT(s->linearAcceleration.x, WithinRel(-expected, 0.02f));
    // y still carries -g: the body is unaccelerated (gravity off) but world
    // gravity is unchanged, so the sensor reads it as "supported", as above.
    CHECK_THAT(s->linearAcceleration.y, WithinAbs(kG, 0.1f));
    CHECK_THAT(s->linearAcceleration.z, WithinAbs(0.f, 0.1f));

    // The gyro is unaffected by the offset — angular velocity is a property of
    // the whole rigid body, not of the measurement point.
    CHECK_THAT(s->angularVelocity.y, WithinRel(omega, 0.01f));

    world.unregisterSensor(&imu);
}

TEST_CASE("A noisy IMU is bit-reproducible for a fixed seed") {

    // The whole point of the seeded SplitMix64 stream: two identical runs must
    // produce identical measurements, or a recorded dataset cannot be replayed.
    const auto run = [] {
        PhysxWorld world(fixedStep());
        auto mesh = cube();
        auto* body = world.add(*mesh, 1000.f);
        body->setAngularDamping(0.f);
        body->setAngularVelocity(::physx::PxVec3(0.5f, 2.f, -0.25f));

        Imu imu(*mesh);// default MEMS-class noise, default seeds
        world.registerSensor(&imu);
        stepFor(world, 50);

        std::vector<ImuSample> out;
        imu.drain(out);
        world.unregisterSensor(&imu);
        return out;
    };

    const auto a = run();
    const auto b = run();

    REQUIRE(a.size() == 50);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        INFO("sample " << i);
        REQUIRE(a[i].t == b[i].t);
        REQUIRE(a[i].angularVelocity.x == b[i].angularVelocity.x);
        REQUIRE(a[i].angularVelocity.y == b[i].angularVelocity.y);
        REQUIRE(a[i].linearAcceleration.x == b[i].linearAcceleration.x);
        REQUIRE(a[i].linearAcceleration.y == b[i].linearAcceleration.y);
    }

    // And the noise actually did something — otherwise "reproducible" is vacuous.
    bool varies = false;
    for (std::size_t i = 1; i < a.size(); ++i) {
        if (a[i].angularVelocity.x != a[0].angularVelocity.x) varies = true;
    }
    CHECK(varies);
}

TEST_CASE("Samples are stamped with the accumulated sim clock") {

    PhysxWorld world(fixedStep());
    auto mesh = cube();
    world.add(*mesh, 1000.f);

    Imu imu(*mesh);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 12);

    std::vector<ImuSample> samples;
    imu.drain(samples);
    REQUIRE(samples.size() == 12);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        INFO("sample " << i);
        REQUIRE_THAT(samples[i].t, WithinAbs(static_cast<double>(i + 1) * kSubstep, 1e-9));
    }
    CHECK_THAT(world.simTime(), WithinAbs(12.0 * kSubstep, 1e-9));

    world.unregisterSensor(&imu);
}

TEST_CASE("A rate-gated IMU runs slower than the physics loop") {

    PhysxWorld world(fixedStep());// 240 Hz
    auto mesh = cube();
    world.add(*mesh, 1000.f);

    Imu imu(*mesh, /*rateHz*/ 60.0);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 240);// 1 second

    std::vector<ImuSample> samples;
    imu.drain(samples);
    INFO("emitted " << samples.size() << " samples at 60 Hz over 1 s of 240 Hz physics");
    CHECK(samples.size() >= 59);
    CHECK(samples.size() <= 61);

    world.unregisterSensor(&imu);
}

TEST_CASE("Registering an IMU with no rigid body throws") {

    // Better here than as silent zeros three hours into a data-collection run.
    PhysxWorld world(fixedStep());

    auto orphan = Object3D::create();
    Imu imu(*orphan);

    CHECK_THROWS_AS(world.registerSensor(&imu), std::invalid_argument);
    CHECK_FALSE(imu.attached());
}

TEST_CASE("An unregistered IMU can be ticked without crashing") {

    // tick() is public, and Sensor has no way to refuse: sample() has to cope
    // with never having resolved a body.
    auto mesh = cube();
    Imu imu(*mesh);

    CHECK_FALSE(imu.attached());
    imu.tick(kSubstep, kSubstep);
    CHECK_FALSE(imu.latest().has_value());
}

TEST_CASE("Removing the body under a registered IMU does not dangle") {

    // removeActor() releases the PxRigidActor. An IMU resolves and caches that
    // actor at registration, so without a notification the next substep would
    // call getGlobalPose() on freed memory.
    PhysxWorld world(fixedStep());

    auto mesh = cube();
    auto* body = world.add(*mesh, 1000.f);

    Imu imu(*mesh);
    makeNoiseless(imu);
    world.registerSensor(&imu);

    stepFor(world, 5);
    REQUIRE(imu.available() == 5);
    REQUIRE(imu.attached());

    world.removeActor(body);
    CHECK_FALSE(imu.attached());

    // Still registered, now inert: stepping must be safe and produce no data.
    stepFor(world, 5);
    CHECK(imu.available() == 5);

    world.unregisterSensor(&imu);
}
