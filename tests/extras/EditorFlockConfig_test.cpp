// EditorFlockConfig_test — the flock's editor wiring, renderer-free.
//
// Three seams: the userData round trip (what a saved document carries), the
// factory (what "Add ▸ Flock" creates), and the play session (birds exist
// exactly while playing, and actually fly). The Flock itself is covered by
// Flock_test; here only the editor plumbing around it is on trial.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/FlockPlaySession.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>

using namespace threepp;
using namespace threepp::editor;

TEST_CASE("FlockConfig: userData round trip preserves every field") {

    FlockConfig config;
    config.seed = 7;
    config.birdCount = 31;
    config.massKg = 0.12f;
    config.roamRadius = 25.f;
    config.cruiseAltitude = 9.f;
    config.altitudeSpread = 0.2f;
    config.cruiseSpeed = 11.f;
    config.perching = false;
    config.maxPerchedFraction = 0.4f;
    config.castShadow = true;
    config.windX = -1.25f;
    config.windZ = 0.5f;

    auto node = Group::create();
    REQUIRE_FALSE(FlockConfig::isFlock(*node));

    config.write(*node);
    REQUIRE(FlockConfig::isFlock(*node));
    REQUIRE(FlockConfig::read(*node) == config);

    FlockConfig::erase(*node);
    REQUIRE_FALSE(FlockConfig::isFlock(*node));
}

TEST_CASE("FlockConfig: makeParams carries the node's home and the config's knobs") {

    FlockConfig config;
    config.birdCount = 5;
    config.roamRadius = 20.f;

    const auto params = config.makeParams({3.f, 12.f, -4.f});
    REQUIRE(params.home == Vector3{3.f, 12.f, -4.f});
    REQUIRE(params.birdCount == 5);
    REQUIRE(params.roamRadius == 20.f);
}

TEST_CASE("FlockPlaySession: birds exist exactly while playing, and fly") {

    Scene scene;
    auto node = ObjectFactory::createFlock(scene);
    REQUIRE(FlockConfig::isFlock(*node));
    REQUIRE(node->name == "Flock");
    scene.add(node);

    const auto countFlocks = [&] {
        int n = 0;
        scene.traverse([&](Object3D& o) {
            if (o.type() == "Flock") ++n;
        });
        return n;
    };

    FlockPlaySession session;
    REQUIRE(countFlocks() == 0);

    session.start(scene);
    REQUIRE(countFlocks() == 1);

    // Find the session's flock and watch a bird move.
    Flock* flock = nullptr;
    scene.traverse([&](Object3D& o) {
        if (auto* f = dynamic_cast<Flock*>(&o)) flock = f;
    });
    REQUIRE(flock != nullptr);
    const Vector3 before = flock->birdPosition(0);
    for (int i = 0; i < 60; ++i) session.update(1.f / 60.f);
    const Vector3 after = flock->birdPosition(0);
    REQUIRE(before.distanceTo(after) > 0.1f);
    REQUIRE(std::isfinite(after.x));

    session.stop();
    REQUIRE(countFlocks() == 0);
}
