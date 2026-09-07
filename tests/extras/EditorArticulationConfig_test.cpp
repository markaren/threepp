// ArticulationConfig round trip.
//
// Same contract as PhysicsConfig/SensorConfig, and the same reasons: the flat
// key=value string is what rides through userData, unknown keys survive a
// newer-editor document, and a disabled entry leaves no trace. PhysX-free, so it
// runs everywhere.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ArticulationConfig.hpp"

#include "threepp/objects/Group.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

TEST_CASE("ArticulationConfig encodes and decodes every field") {

    ArticulationConfig config;
    config.enabled = true;
    config.fixedBase = false;
    config.stiffness = 321.5f;
    config.damping = 12.25f;
    config.maxForce = 5e5f;
    config.selfCollision = true;
    config.iterations = 20;
    config.density = 750.f;

    const auto decoded = ArticulationConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(decoded->enabled);
    CHECK(decoded->fixedBase == false);
    CHECK_THAT(decoded->stiffness, WithinAbs(321.5f, 1e-3f));
    CHECK_THAT(decoded->damping, WithinAbs(12.25f, 1e-3f));
    CHECK_THAT(decoded->maxForce, WithinAbs(5e5f, 1.f));
    CHECK(decoded->selfCollision);
    CHECK(decoded->iterations == 20);
    CHECK_THAT(decoded->density, WithinAbs(750.f, 1e-3f));
    // Value equality, the whole struct at once.
    CHECK(*decoded == config);
}

TEST_CASE("ArticulationConfig defaults simulate a fixed arm that holds its pose") {

    // The arm case is the common one: a fixed base and a drive stiff enough that
    // a horizontal link does not sag the instant Play starts.
    const ArticulationConfig config;
    CHECK(config.fixedBase);
    CHECK(config.stiffness > 0.f);
    CHECK(config.damping > 0.f);
}

TEST_CASE("ArticulationConfig ignores unknown keys and keeps defaults for missing ones") {

    // A document from a newer editor carries a key this build has never heard of;
    // it must load, ignoring the stranger and defaulting what it lacks.
    const auto decoded = ArticulationConfig::decode(
            "fixedbase=0;stiffness=200;futurekey=whatever;damping=5");
    REQUIRE(decoded.has_value());
    CHECK(decoded->fixedBase == false);
    CHECK_THAT(decoded->stiffness, WithinAbs(200.f, 1e-3f));
    CHECK_THAT(decoded->damping, WithinAbs(5.f, 1e-3f));
    // maxForce/iterations/density were absent, so they keep their defaults.
    const ArticulationConfig d;
    CHECK_THAT(decoded->maxForce, WithinAbs(d.maxForce, 1.f));
    CHECK(decoded->iterations == d.iterations);
    CHECK_THAT(decoded->density, WithinAbs(d.density, 1e-3f));
}

TEST_CASE("An empty string is not an articulation") {

    CHECK_FALSE(ArticulationConfig::decode("").has_value());
}

TEST_CASE("write with enabled == false erases the entry entirely") {

    auto object = Group::create();

    ArticulationConfig on;
    on.enabled = true;
    on.write(*object);
    REQUIRE(ArticulationConfig::read(*object).has_value());
    // Presence of the entry is what "simulate" means.
    CHECK(object->userData.contains(ArticulationConfig::userDataKey));

    ArticulationConfig off;
    off.enabled = false;
    off.write(*object);
    CHECK_FALSE(ArticulationConfig::read(*object).has_value());
    CHECK_FALSE(object->userData.contains(ArticulationConfig::userDataKey));
}

TEST_CASE("read returns an enabled config whenever the entry is present") {

    auto object = Group::create();
    // A hand-written entry with only some keys still reads as enabled: presence,
    // not an enabled= key, is the signal.
    object->userData[ArticulationConfig::userDataKey] = std::string("stiffness=100");

    const auto read = ArticulationConfig::read(*object);
    REQUIRE(read.has_value());
    CHECK(read->enabled);
    CHECK_THAT(read->stiffness, WithinAbs(100.f, 1e-3f));
}
