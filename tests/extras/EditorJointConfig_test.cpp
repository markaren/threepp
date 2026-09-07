// JointConfig round trip.
//
// Same contract as PhysicsConfig/SensorConfig: the flat key=value string rides
// through userData, unknown keys survive a newer-editor document. Two joint
// particulars get their own cases — presence (not an enabled= key) is what
// makes a node a joint, and the other body's NAME lives on a plain key of its
// own because a user-typed name may contain the flat format's delimiters.
// PhysX-free, so it runs everywhere.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/JointConfig.hpp"

#include "threepp/objects/Group.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

TEST_CASE("JointConfig encodes and decodes every field") {

    JointConfig config;
    config.type = JointConfig::Type::Prismatic;
    config.limited = true;
    config.lower = -0.25f;
    config.upper = 0.75f;
    config.coneY = 0.5f;
    config.coneZ = 0.25f;
    config.stiffness = 321.5f;
    config.damping = 12.25f;
    config.maxForce = 5e5f;
    config.target = 0.4f;
    config.velocity = -1.5f;
    config.breakForce = 2000.f;
    config.breakTorque = 500.f;
    config.collide = true;

    const auto decoded = JointConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(decoded->type == JointConfig::Type::Prismatic);
    CHECK(decoded->limited);
    CHECK_THAT(decoded->lower, WithinAbs(-0.25f, 1e-4f));
    CHECK_THAT(decoded->upper, WithinAbs(0.75f, 1e-4f));
    CHECK_THAT(decoded->coneY, WithinAbs(0.5f, 1e-4f));
    CHECK_THAT(decoded->coneZ, WithinAbs(0.25f, 1e-4f));
    CHECK_THAT(decoded->stiffness, WithinAbs(321.5f, 1e-3f));
    CHECK_THAT(decoded->damping, WithinAbs(12.25f, 1e-3f));
    CHECK_THAT(decoded->maxForce, WithinAbs(5e5f, 1.f));
    CHECK_THAT(decoded->target, WithinAbs(0.4f, 1e-4f));
    CHECK_THAT(decoded->velocity, WithinAbs(-1.5f, 1e-4f));
    CHECK_THAT(decoded->breakForce, WithinAbs(2000.f, 1e-2f));
    CHECK_THAT(decoded->breakTorque, WithinAbs(500.f, 1e-2f));
    CHECK(decoded->collide);
    // Value equality, the whole struct at once — body is empty on both sides
    // (it rides its own userData key, not the flat string).
    CHECK(*decoded == config);
}

TEST_CASE("JointConfig defaults are a passive unlimited hinge") {

    const JointConfig config;
    CHECK(config.type == JointConfig::Type::Revolute);
    CHECK_FALSE(config.limited);
    CHECK(config.body.empty());
    // Passive until told otherwise: no drive, unbreakable, no self-contact.
    CHECK(config.stiffness == 0.f);
    CHECK(config.damping == 0.f);
    CHECK(config.breakForce == 0.f);
    CHECK_FALSE(config.collide);
}

TEST_CASE("JointConfig ignores unknown keys and keeps defaults for missing ones") {

    const auto decoded = JointConfig::decode(
            "type=spherical;futurekey=whatever;stiffness=200");
    REQUIRE(decoded.has_value());
    CHECK(decoded->type == JointConfig::Type::Spherical);
    CHECK_THAT(decoded->stiffness, WithinAbs(200.f, 1e-3f));
    const JointConfig d;
    CHECK(decoded->limited == d.limited);
    CHECK_THAT(decoded->lower, WithinAbs(d.lower, 1e-4f));
    CHECK_THAT(decoded->maxForce, WithinAbs(d.maxForce, 1.f));
}

TEST_CASE("An empty string is not a joint") {

    CHECK_FALSE(JointConfig::decode("").has_value());
}

TEST_CASE("presence of the entry is what makes a node a joint") {

    auto object = Group::create();
    CHECK_FALSE(JointConfig::isJoint(*object));
    CHECK_FALSE(JointConfig::read(*object).has_value());

    // write() always writes, defaults included: the entry IS the node's
    // identity, so a joint at defaults must still be a joint after a save.
    JointConfig{}.write(*object);
    CHECK(JointConfig::isJoint(*object));
    REQUIRE(JointConfig::read(*object).has_value());

    JointConfig::erase(*object);
    CHECK_FALSE(JointConfig::isJoint(*object));
    CHECK_FALSE(object->userData.contains(JointConfig::userDataKey));
    CHECK_FALSE(object->userData.contains(JointConfig::bodyKey));
}

TEST_CASE("the body name rides its own key and survives delimiters") {

    auto object = Group::create();

    JointConfig config;
    // A name a user is free to type, carrying both flat-format delimiters —
    // exactly what would corrupt the string if it rode inside it.
    config.body = "Crate; mk=2";
    config.write(*object);

    const auto read = JointConfig::read(*object);
    REQUIRE(read.has_value());
    CHECK(read->body == "Crate; mk=2");
    // And the flat string itself stayed parseable: type is still the default.
    CHECK(read->type == JointConfig::Type::Revolute);

    // Clearing the body erases its key, so "jointed to the world" leaves the
    // smaller document.
    config.body.clear();
    config.write(*object);
    CHECK_FALSE(object->userData.contains(JointConfig::bodyKey));
    REQUIRE(JointConfig::read(*object).has_value());
    CHECK(JointConfig::read(*object)->body.empty());
}
