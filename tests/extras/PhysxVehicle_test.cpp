// PhysxVehicle at rest and coasting: what holds a stopped car where it is, and
// what slows one that is let run.
//
// PhysX swaps the slip-based tire force for a "sticky tire" constraint once a
// wheel is nearly at rest, and that constraint is a damper, not a lock: against
// gravity on a grade the car settles at g * sin / damping. At PhysX's stock
// damping a fully braked car backed down a 10 % grade at 0.7 km/h (1.8 m in
// 10 s) and slid sideways parked across it. PhysxVehicleBaseSettings has the
// account, and the two dampings these cases hold to their jobs: the stiff one
// that holds a braked car, and the soft one that lets an unbraked car roll.
//
// Coasting, direct drive has no engine to hold the car back: the engine brake
// (PhysxVehicleSettings::engineBrakeTorque) is that, and off unless asked for.
//
// Physics alone on a tilted slab, a fixed 1/60 s clock, no renderer.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace threepp;

namespace {

    constexpr float kDt = 1.f / 60.f;
    // The chassis' middle over the road at rest, for the tuning below.
    constexpr float kRide = 0.996f;

    // The Range Rover Evoque of examples/projects/Vehicle and of the Python
    // binding's defaults, on evoque_rig.py's tires (mu 1.1).
    PhysxVehicle::Settings evoque() {

        PhysxVehicle::Settings s;
        s.chassisWidth = 1.95f;
        s.chassisHeight = 1.4f;
        s.chassisLength = 4.4f;
        s.wheelbase = 2.66f;
        s.trackWidth = 1.65f;
        s.drivenWheels = {true, true, true, true};
        s.maxThrottleTorque = 1500.f;
        s.wheelDampingRate = 1.5f;
        s.tireFriction = 1.1f;
        s.longitudinalStiffness = 100'000.f;
        return s;
    }

    // A slab that climbs `grade` (rise over run) towards +Z, with a car on it
    // facing `heading` radians round the slab's normal from straight uphill.
    struct Hill {
        PhysxWorld world;
        Quaternion tilt;
        Vector3 up;
        std::unique_ptr<PhysxVehicle> car;
        Vector3 start;

        explicit Hill(float grade, float heading = 0.f, float engineBrake = 0.f) {

            const float a = std::atan(grade);
            tilt.setFromAxisAngle({1, 0, 0}, -a);
            up.set(0, 1, 0).applyQuaternion(tilt);

            world.addStatic(::physx::PxBoxGeometry(100.f, 0.5f, 400.f),
                            toPxTransform(up.clone().multiplyScalar(-0.5f), tilt));

            auto s = evoque();
            s.engineBrakeTorque = engineBrake;
            s.spawnPosition = up.clone().multiplyScalar(kRide);
            s.spawnRotation = tilt.clone().multiply(Quaternion().setFromAxisAngle({0, 1, 0}, heading));
            car = std::make_unique<PhysxVehicle>(world, s);
            start = position();
        }

        Vector3 position() const {
            const auto p = car->chassisPose().p;
            return {p.x, p.y, p.z};
        }

        // How far the chassis has moved ALONG the slab since the spawn (m). The
        // suspension settling into the slab is not the car going anywhere.
        float moved() const {
            Vector3 d = position().sub(start);
            return d.addScaledVector(up, -d.dot(up)).length();
        }

        void run(float seconds, float throttle, float brake) {
            car->setThrottle(throttle);
            car->setBrake(brake);
            const int n = static_cast<int>(std::lround(seconds / kDt));
            for (int i = 0; i < n; ++i) world.step(kDt);
        }
    };

}// namespace


TEST_CASE("A fully braked vehicle stands still on a grade", "[physx][vehicle]") {

    struct Case {
        float grade;
        float heading;
        const char* what;
    };
    const Case cases[] = {
            {0.10f, 0.f, "10 %, nose uphill"},
            {0.10f, math::PI, "10 %, nose downhill"},
            {0.10f, math::PI / 2, "10 %, across the grade"},
            {0.13f, 0.f, "13 %, nose uphill"},
            {0.13f, math::PI, "13 %, nose downhill"},
            {0.13f, math::PI / 2, "13 %, across the grade"},
            {0.13f, math::PI / 4, "13 %, askew"},
            {0.25f, 0.f, "25 %, nose uphill"},
    };

    for (const auto& c : cases) {
        CAPTURE(c.what);
        Hill hill(c.grade, c.heading);
        hill.run(10.f, 0.f, 1.f);
        CHECK(hill.moved() < 0.01f);
        CHECK(std::abs(hill.car->forwardSpeed()) < 1e-3f);
    }
}

TEST_CASE("Any brake at all holds it, as the handbrake does a parked car", "[physx][vehicle]") {

    Hill hill(0.13f);
    hill.run(10.f, 0.f, 0.2f);
    CHECK(hill.moved() < 0.01f);
}

TEST_CASE("Let go on a grade, the vehicle rolls away", "[physx][vehicle]") {

    // The hold is the brake's, not the tires': with the brakes off the wheels
    // are free and nothing may keep the car on the hill.
    SECTION("from a braked stop") {
        Hill hill(0.10f);
        hill.run(1.f, 0.f, 1.f);
        REQUIRE(hill.moved() < 0.01f);
        hill.run(5.f, 0.f, 0.f);
        CHECK(hill.car->forwardSpeed() < -3.f);// 13 km/h backwards, measured
        CHECK(hill.moved() > 8.f);
    }

    SECTION("never braked") {
        Hill hill(0.10f);
        hill.run(5.f, 0.f, 0.f);
        CHECK(hill.car->forwardSpeed() < -3.f);
    }

    // Sideways is the tires' own: parked across the grade there is nothing to
    // roll on, brake or no brake.
    SECTION("but not sideways") {
        Hill hill(0.13f, math::PI / 2);
        hill.run(10.f, 0.f, 0.f);
        CHECK(hill.moved() < 0.01f);
    }
}

TEST_CASE("The hold lets go for the throttle", "[physx][vehicle]") {

    Hill hill(0.10f);
    hill.run(2.f, 0.f, 1.f);
    REQUIRE(hill.moved() < 0.01f);
    hill.run(6.f, 0.5f, 0.f);
    CHECK(hill.car->forwardSpeed() > 15.f);// 59 km/h up the grade, measured
}

TEST_CASE("A vehicle braked to a stop on the flat stays there", "[physx][vehicle]") {

    Hill flat(0.f);
    flat.run(4.f, 1.f, 0.f);
    REQUIRE(flat.car->forwardSpeed() > 25.f);// 110 km/h, measured
    flat.run(6.f, 0.f, 1.f);
    CHECK(std::abs(flat.car->forwardSpeed()) < 1e-3f);
    const float at = flat.moved();
    flat.run(4.f, 0.f, 1.f);
    CHECK(std::abs(flat.moved() - at) < 1e-3f);
}

TEST_CASE("On ground that moves, the hold does not nail the vehicle to the world", "[physx][vehicle]") {

    // The sticky constraint pulls the contact patch towards rest in the WORLD,
    // so on a deck under way the stiff damping would leave the car behind:
    // there both directions stay soft, as they were (see the settings). Soft
    // still drags (the car slips back at some 0.2 m/s, as it always has), and
    // that is the bound here: with the stiff damping it was 5.3 m and 2.1 m.
    const ::physx::PxVec3 ways[] = {{0, 0, 1}, {1, 0, 0}};// ahead, sideways
    const float bounds[] = {2.5f, 1.f};

    for (int k = 0; k < 2; ++k) {
        CAPTURE(k);
        PhysxWorld world;
        auto* deck = world.physics().createRigidDynamic(::physx::PxTransform(::physx::PxVec3(0, -0.5f, 0)));
        auto* shape = world.physics().createShape(::physx::PxBoxGeometry(50.f, 0.5f, 50.f), world.defaultMaterial(), true);
        deck->attachShape(*shape);
        shape->release();
        deck->setRigidBodyFlag(::physx::PxRigidBodyFlag::eKINEMATIC, true);
        world.scene().addActor(*deck);

        auto s = evoque();
        s.spawnPosition = {0, kRide, 0};
        PhysxVehicle car(world, s);
        car.setBrake(1.f);
        // A vehicle asleep is left behind whatever its tires do (PhysX wakes
        // it for the throttle and the steering only): not what is asked here.
        car.chassisActor()->setSleepThreshold(0.f);

        // At rest for a second, up to 1 m/s over two, and on to ten.
        float travelled = 0.f;
        for (int i = 1; i <= 600; ++i) {
            const float v = i <= 60 ? 0.f : std::min(1.f, static_cast<float>(i - 60) / 120.f);
            travelled += v * kDt;
            deck->setKinematicTarget(::physx::PxTransform(ways[k] * travelled + ::physx::PxVec3(0, -0.5f, 0)));
            world.step(kDt);
        }
        REQUIRE(travelled > 8.f);
        CHECK(travelled - car.chassisPose().p.dot(ways[k]) < bounds[k]);
    }
}

TEST_CASE("Engine braking holds a coasting vehicle back", "[physx][vehicle]") {

    constexpr float kTorque = 75.f;// N*m a wheel: 750 N on these four

    // Up to speed for 2 s on the flat, then 4 s with the throttle released:
    // the speed at each end. (One PhysxWorld at a time: PhysX's foundation.)
    struct Coast {
        float before, after;
    };
    const auto coast = [](float torque, bool neutral = false) {
        Hill flat(0.f, 0.f, torque);
        flat.run(2.f, 1.f, 0.f);
        const float before = flat.car->forwardSpeed();
        if (neutral) flat.car->setGear(PhysxVehicle::Gear::Neutral);
        flat.run(4.f, 0.f, 0.f);
        return Coast{before, flat.car->forwardSpeed()};
    };

    SECTION("on the flat") {
        const Coast free = coast(0.f), geared = coast(kTorque);
        // Under the throttle there is none of it: the same car to the bit.
        REQUIRE(free.before > 15.f);
        REQUIRE(geared.before == free.before);
        // 750 N on 1500 kg (and the wheels) over 4 s: some 1.8 m/s the fewer.
        CHECK(free.after - geared.after > 1.2f);
        CHECK(free.after - geared.after < 2.4f);
    }

    SECTION("in neutral there is none") {
        CHECK(coast(kTorque, true).after == coast(0.f).after);
    }

    SECTION("down a grade it settles where the free car is still gathering speed") {
        // Nose downhill on 10 %, let go and never braked. Gravity's 1470 N has
        // only the chassis' damping against it in the free car (150 N s/m: 35
        // km/h in the end); the engine's 750 N leave it some 17.
        {
            Hill free(0.10f, math::PI);
            free.run(30.f, 0.f, 0.f);
            CHECK(free.car->forwardSpeed() > 7.f);// 27 km/h after 30 s, measured
        }
        Hill geared(0.10f, math::PI, kTorque);
        geared.run(30.f, 0.f, 0.f);
        CHECK(geared.car->forwardSpeed() > 3.5f);// it does roll away: nothing of it under the idle speed
        CHECK(geared.car->forwardSpeed() < 6.f);

        // A lower gear, under way.
        geared.car->setEngineBrakeTorque(2.f * kTorque);
        geared.run(20.f, 0.f, 0.f);
        CHECK(geared.car->forwardSpeed() > 1.5f);// the idle speed: under it the engine lets go
        CHECK(geared.car->forwardSpeed() < 3.5f);
    }
}
