// VehicleConfig round trip + derived geometry.
//
// Same contract as PhysicsConfig/JointConfig: the flat key=value string rides
// through userData, unknown keys survive a newer-editor document. The vehicle
// particulars get their own cases — presence (not an enabled= key) is what
// makes a node a vehicle, and the four wheel NAMES live on plain keys of their
// own because a user-typed name may contain the flat format's delimiters. The
// derivation cases build a primitives car and read the geometry off it, which
// is the feature's soul: an imported car drives with no number typed.
// PhysX-free, so it runs everywhere.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/VehicleConfig.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // A primitives car the derivation can measure: box body over four
    // cylinder wheels (axle along X), hubs at (±0.8, 0.4, ±1.4), radius 0.4,
    // width 0.3 — so wheelbase 2.8, track 1.6 are the expected answers.
    std::shared_ptr<Group> makeCar() {

        auto car = Group::create();
        car->name = "Car";

        auto body = Mesh::create(BoxGeometry::create(1.6f, 0.8f, 4.2f));
        body->name = "Body";
        body->position.set(0.f, 1.f, 0.f);
        car->add(body);

        const char* names[4] = {"FR", "FL", "RR", "RL"};
        const Vector3 hubs[4] = {{0.8f, 0.4f, 1.4f},
                                 {-0.8f, 0.4f, 1.4f},
                                 {0.8f, 0.4f, -1.4f},
                                 {-0.8f, 0.4f, -1.4f}};
        for (int i = 0; i < 4; ++i) {
            auto wheel = Mesh::create(CylinderGeometry::create(0.4f, 0.4f, 0.3f, 24));
            wheel->name = names[i];
            wheel->rotation.z = math::PI / 2;// cylinder height (Y) onto the axle (X)
            wheel->position.copy(hubs[i]);
            car->add(wheel);
        }
        return car;
    }

    VehicleConfig carConfig() {

        VehicleConfig config;
        config.wheels = {"FR", "FL", "RR", "RL"};
        return config;
    }

}// namespace


TEST_CASE("VehicleConfig encodes and decodes every field") {

    VehicleConfig config;
    config.drive = VehicleConfig::Drive::Engine;
    config.driven = VehicleConfig::Driven::Front;
    config.autoGeometry = false;
    config.chassisWidth = 1.95f;
    config.chassisHeight = 1.4f;
    config.chassisLength = 4.4f;
    config.wheelRadius = 0.33f;
    config.wheelWidth = 0.25f;
    config.trackWidth = 1.65f;
    config.wheelbase = 2.66f;
    config.suspensionY = -0.25f;
    config.mass = 1750.f;
    config.suspensionTravel = 0.25f;
    config.suspensionStiffness = 40000.f;
    config.suspensionDamping = 5000.f;
    config.tireFriction = 1.75f;
    config.maxBrakeTorque = 6000.f;
    config.maxSteerAngle = 0.5f;
    config.throttleTorque = 2000.f;

    const auto decoded = VehicleConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(decoded->drive == VehicleConfig::Drive::Engine);
    CHECK(decoded->driven == VehicleConfig::Driven::Front);
    CHECK_FALSE(decoded->autoGeometry);
    CHECK_THAT(decoded->chassisWidth, WithinAbs(1.95f, 1e-4f));
    CHECK_THAT(decoded->chassisHeight, WithinAbs(1.4f, 1e-4f));
    CHECK_THAT(decoded->chassisLength, WithinAbs(4.4f, 1e-4f));
    CHECK_THAT(decoded->wheelRadius, WithinAbs(0.33f, 1e-4f));
    CHECK_THAT(decoded->wheelWidth, WithinAbs(0.25f, 1e-4f));
    CHECK_THAT(decoded->trackWidth, WithinAbs(1.65f, 1e-4f));
    CHECK_THAT(decoded->wheelbase, WithinAbs(2.66f, 1e-4f));
    CHECK_THAT(decoded->suspensionY, WithinAbs(-0.25f, 1e-4f));
    CHECK_THAT(decoded->mass, WithinAbs(1750.f, 1e-2f));
    CHECK_THAT(decoded->suspensionTravel, WithinAbs(0.25f, 1e-4f));
    CHECK_THAT(decoded->suspensionStiffness, WithinAbs(40000.f, 1.f));
    CHECK_THAT(decoded->suspensionDamping, WithinAbs(5000.f, 1e-1f));
    CHECK_THAT(decoded->tireFriction, WithinAbs(1.75f, 1e-4f));
    CHECK_THAT(decoded->maxBrakeTorque, WithinAbs(6000.f, 1e-1f));
    CHECK_THAT(decoded->maxSteerAngle, WithinAbs(0.5f, 1e-4f));
    CHECK_THAT(decoded->throttleTorque, WithinAbs(2000.f, 1e-1f));
    // Value equality, the whole struct at once — the wheels are empty on both
    // sides (they ride their own userData keys, not the flat string).
    CHECK(*decoded == config);
}

TEST_CASE("VehicleConfig defaults are an auto-geometry AWD direct drive") {

    const VehicleConfig config;
    CHECK(config.drive == VehicleConfig::Drive::Direct);
    CHECK(config.driven == VehicleConfig::Driven::All);
    CHECK(config.autoGeometry);
    for (const auto& wheel : config.wheels) CHECK(wheel.empty());
    // Nonzero defaults everywhere a widget drags — no log-drag-from-zero trap.
    CHECK(config.mass > 0.f);
    CHECK(config.suspensionStiffness > 0.f);
    CHECK(config.throttleTorque > 0.f);
}

TEST_CASE("VehicleConfig ignores unknown keys and keeps defaults for missing ones") {

    const auto decoded = VehicleConfig::decode(
            "drive=engine;futurekey=whatever;mass=990");
    REQUIRE(decoded.has_value());
    CHECK(decoded->drive == VehicleConfig::Drive::Engine);
    CHECK_THAT(decoded->mass, WithinAbs(990.f, 1e-2f));
    const VehicleConfig d;
    CHECK(decoded->driven == d.driven);
    CHECK(decoded->autoGeometry == d.autoGeometry);
    CHECK_THAT(decoded->wheelbase, WithinAbs(d.wheelbase, 1e-4f));
    CHECK_THAT(decoded->tireFriction, WithinAbs(d.tireFriction, 1e-4f));
}

TEST_CASE("An empty string is not a vehicle") {

    CHECK_FALSE(VehicleConfig::decode("").has_value());
}

TEST_CASE("presence of the entry is what makes a node a vehicle") {

    auto object = Group::create();
    CHECK_FALSE(VehicleConfig::isVehicle(*object));
    CHECK_FALSE(VehicleConfig::read(*object).has_value());

    // write() always writes, defaults included: the entry IS the node's
    // identity, so a vehicle at defaults must still be a vehicle after a save.
    VehicleConfig{}.write(*object);
    CHECK(VehicleConfig::isVehicle(*object));
    REQUIRE(VehicleConfig::read(*object).has_value());

    VehicleConfig::erase(*object);
    CHECK_FALSE(VehicleConfig::isVehicle(*object));
    CHECK_FALSE(object->userData.contains(VehicleConfig::userDataKey));
    for (const auto* key : VehicleConfig::wheelKeys) {
        CHECK_FALSE(object->userData.contains(key));
    }
}

TEST_CASE("wheel names ride their own keys and survive delimiters") {

    auto object = Group::create();

    VehicleConfig config;
    // Names a user is free to type, carrying both flat-format delimiters —
    // exactly what would corrupt the string if they rode inside it.
    config.wheels = {"Wheel; FR", "mesh=FL", "RR;x=1", "RL"};
    config.write(*object);

    const auto read = VehicleConfig::read(*object);
    REQUIRE(read.has_value());
    CHECK(read->wheels[0] == "Wheel; FR");
    CHECK(read->wheels[1] == "mesh=FL");
    CHECK(read->wheels[2] == "RR;x=1");
    CHECK(read->wheels[3] == "RL");
    // And the flat string itself stayed parseable: drive is still the default.
    CHECK(read->drive == VehicleConfig::Drive::Direct);

    // Clearing a wheel erases its key, so a half-picked vehicle leaves the
    // smaller document.
    config.wheels[3].clear();
    config.write(*object);
    CHECK_FALSE(object->userData.contains(VehicleConfig::wheelKeys[3]));
    REQUIRE(VehicleConfig::read(*object).has_value());
    CHECK(VehicleConfig::read(*object)->wheels[3].empty());
}

TEST_CASE("geometry is derived from the four picked wheels") {

    auto car = makeCar();
    const auto config = carConfig();

    const auto geo = config.derived(*car);
    REQUIRE(geo.valid);
    for (const auto* wheel : geo.wheels) CHECK(wheel != nullptr);

    CHECK_THAT(geo.wheelbase, WithinAbs(2.8f, 0.01f));
    CHECK_THAT(geo.trackWidth, WithinAbs(1.6f, 0.01f));
    CHECK_THAT(geo.wheelRadius, WithinAbs(0.4f, 0.01f));
    CHECK_THAT(geo.wheelWidth, WithinAbs(0.3f, 0.01f));
    // Model bounds: 1.6 wide (the body), wheels at ±0.95 -> 1.9; the body is
    // 4.2 long; height spans wheel bottom 0.0 to body top 1.4.
    CHECK_THAT(geo.chassisLength, WithinAbs(4.2f, 0.01f));
    CHECK_THAT(geo.chassisHeight, WithinAbs(1.4f, 0.01f));
    CHECK_THAT(geo.chassisWidth, WithinAbs(1.9f, 0.01f));

    // The chassis centre: axle midpoint in X/Z, model centre in Y.
    CHECK_THAT(geo.position.x, WithinAbs(0.f, 0.01f));
    CHECK_THAT(geo.position.y, WithinAbs(0.7f, 0.01f));
    CHECK_THAT(geo.position.z, WithinAbs(0.f, 0.01f));

    // Suspension attachment: hub height relative to the chassis centre
    // (0.4 - 0.7), lifted by travel minus the static jounce
    // (0.3 - 0.25*1500*9.81/35000) so the car RESTS at the authored height.
    CHECK_THAT(geo.suspensionY, WithinAbs(-0.3f + 0.3f - 0.10511f, 0.005f));

    // Hubs land where they were authored, relative to the centre.
    CHECK_THAT(geo.hubs[0].x, WithinAbs(0.8f, 0.01f));
    CHECK_THAT(geo.hubs[0].z, WithinAbs(1.4f, 0.01f));
    CHECK_THAT(geo.hubs[3].x, WithinAbs(-0.8f, 0.01f));
    CHECK_THAT(geo.hubs[3].z, WithinAbs(-1.4f, 0.01f));
}

TEST_CASE("the derived frame follows the picks, not a facing convention") {

    auto car = makeCar();
    const auto config = carConfig();

    // Park the car at an angle: the measurements must not change.
    car->position.set(3.f, 0.f, -2.f);
    car->rotation.y = math::degToRad(35.f);

    const auto geo = config.derived(*car);
    REQUIRE(geo.valid);
    CHECK_THAT(geo.wheelbase, WithinAbs(2.8f, 0.01f));
    CHECK_THAT(geo.trackWidth, WithinAbs(1.6f, 0.01f));
    CHECK_THAT(geo.wheelRadius, WithinAbs(0.4f, 0.01f));

    // The frame's +Z points from the rear axle to the front one, whatever
    // the model's own axes say: swap the front/rear picks and the derived
    // forward flips with them.
    VehicleConfig flipped = config;
    flipped.wheels = {"RR", "RL", "FR", "FL"};
    const auto geoFlipped = flipped.derived(*car);
    REQUIRE(geoFlipped.valid);
    Vector3 fwd(0.f, 0.f, 1.f);
    fwd.applyQuaternion(geo.rotation);
    Vector3 fwdFlipped(0.f, 0.f, 1.f);
    fwdFlipped.applyQuaternion(geoFlipped.rotation);
    CHECK_THAT(fwd.dot(fwdFlipped), WithinAbs(-1.f, 1e-3f));
}

TEST_CASE("a wheel pick may be a group, measured as its assembly") {

    // Wheels authored the way imports actually arrive: an assembly group per
    // corner holding a tire mesh and a hub block. Picking the GROUP measures
    // the union — the tire's radius wins, the hub adds nothing.
    auto car = Group::create();
    car->name = "Car";
    auto body = Mesh::create(BoxGeometry::create(1.6f, 0.8f, 4.2f));
    body->name = "Body";
    body->position.set(0.f, 1.f, 0.f);
    car->add(body);

    const char* names[4] = {"AssemblyFR", "AssemblyFL", "AssemblyRR", "AssemblyRL"};
    const Vector3 hubs[4] = {{0.8f, 0.4f, 1.4f},
                             {-0.8f, 0.4f, 1.4f},
                             {0.8f, 0.4f, -1.4f},
                             {-0.8f, 0.4f, -1.4f}};
    for (int i = 0; i < 4; ++i) {
        auto assembly = Group::create();
        assembly->name = names[i];
        assembly->position.copy(hubs[i]);
        auto tire = Mesh::create(CylinderGeometry::create(0.4f, 0.4f, 0.3f, 24));
        tire->name = "Tire";
        tire->rotation.z = math::PI / 2;
        assembly->add(tire);
        auto hub = Mesh::create(BoxGeometry::create(0.15f, 0.15f, 0.15f));
        hub->name = "Hub";
        assembly->add(hub);
        car->add(assembly);
    }

    VehicleConfig config;
    config.wheels = {"AssemblyFR", "AssemblyFL", "AssemblyRR", "AssemblyRL"};
    const auto geo = config.derived(*car);
    REQUIRE(geo.valid);
    CHECK_THAT(geo.wheelRadius, WithinAbs(0.4f, 0.01f));
    CHECK_THAT(geo.wheelWidth, WithinAbs(0.3f, 0.01f));
    CHECK_THAT(geo.wheelbase, WithinAbs(2.8f, 0.01f));
    CHECK_THAT(geo.trackWidth, WithinAbs(1.6f, 0.01f));
}

TEST_CASE("duplicate names are told apart by ordinal references") {

    // Four wheels ALL named "Wheel" — the normal shape of an imported asset,
    // not the odd one. "Wheel" is the first in document order and "Wheel#N"
    // the N-th, which is exactly what the inspector's combos offer.
    auto car = makeCar();
    for (const char* name : {"FR", "FL", "RR", "RL"}) {
        car->getObjectByName(name)->name = "Wheel";
    }

    VehicleConfig config;
    config.wheels = {"Wheel", "Wheel#2", "Wheel#3", "Wheel#4"};
    const auto geo = config.derived(*car);
    REQUIRE(geo.valid);
    // Document order is the order makeCar added them: FR, FL, RR, RL — so
    // each slot landed on its own corner.
    CHECK(geo.hubs[0].x > 0.f);
    CHECK(geo.hubs[0].z > 0.f);
    CHECK(geo.hubs[1].x < 0.f);
    CHECK(geo.hubs[1].z > 0.f);
    CHECK(geo.hubs[2].x > 0.f);
    CHECK(geo.hubs[2].z < 0.f);
    CHECK(geo.hubs[3].x < 0.f);
    CHECK(geo.hubs[3].z < 0.f);
    CHECK_THAT(geo.wheelbase, WithinAbs(2.8f, 0.01f));
    CHECK_THAT(geo.trackWidth, WithinAbs(1.6f, 0.01f));

    // An ordinal past the duplicates is a plain unresolvable pick.
    VehicleConfig overshoot = config;
    overshoot.wheels[3] = "Wheel#5";
    CHECK_FALSE(overshoot.derived(*car).valid);
}

TEST_CASE("an unresolvable pick says which wheel is the problem") {

    auto car = makeCar();

    VehicleConfig config = carConfig();
    config.wheels[2] = "Nowhere";
    const auto geo = config.derived(*car);
    CHECK_FALSE(geo.valid);
    CHECK(geo.problem.find("Nowhere") != std::string::npos);
    CHECK(geo.problem.find("Rear Right") != std::string::npos);

    VehicleConfig unpicked = carConfig();
    unpicked.wheels[1].clear();
    const auto geoUnpicked = unpicked.derived(*car);
    CHECK_FALSE(geoUnpicked.valid);
    CHECK(geoUnpicked.problem.find("Front Left") != std::string::npos);

    VehicleConfig doubled = carConfig();
    doubled.wheels[1] = "FR";
    const auto geoDoubled = doubled.derived(*car);
    CHECK_FALSE(geoDoubled.valid);
    CHECK(geoDoubled.problem.find("FR") != std::string::npos);
}
