
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Ground: a flattened box rather than a plane, so the collider has real
    // thickness in every axis.
    std::shared_ptr<Mesh> makeGround(Scene& scene) {

        auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
        ground->name = "Ground";
        ground->scale.set(20.f, 0.2f, 20.f);
        ground->position.set(0.f, -0.1f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Static;
        config.shape = PhysicsConfig::Shape::Box;
        config.write(*ground);
        return ground;
    }

    std::shared_ptr<Mesh> makeFallingBox(Scene& scene, PhysicsConfig::Shape shape) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = "Falling";
        box->position.set(0.f, 4.f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.shape = shape;
        config.mass = 5.f;
        config.friction = 0.6f;
        config.restitution = 0.f;
        config.write(*box);
        return box;
    }

    void run(PhysicsPlaySession& session, int steps) {

        for (int i = 0; i < steps; ++i) session.update(1.f / 60.f);
    }

}// namespace


TEST_CASE("PhysicsPlaySession simulates userData-authored bodies", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto ground = makeGround(scene);
    auto box = makeFallingBox(scene, PhysicsConfig::Shape::Box);
    scene.add(ground);
    scene.add(box);

    // An object with no physics entry must be left entirely alone.
    auto decoration = ObjectFactory::createPrimitive(Primitive::Sphere, scene);
    decoration->name = "Decoration";
    decoration->position.set(3.f, 4.f, 0.f);
    scene.add(decoration);

    PhysicsPlaySession session;
    session.start(scene);

    // Ground + falling box, not the decoration.
    CHECK(session.bodyCount() == 2);

    run(session, 180);// 3 seconds

    using Catch::Matchers::WithinAbs;
    // Resting on the ground: the unit box's centre sits half a metre up.
    CHECK_THAT(box->position.y, WithinAbs(0.5f, 0.05f));
    CHECK_THAT(ground->position.y, WithinAbs(-0.1f, 1e-4));
    // Untouched.
    CHECK_THAT(decoration->position.y, WithinAbs(4.f, 1e-5));

    session.stop();
    CHECK(session.bodyCount() == 0);
}

TEST_CASE("play then stop leaves no trace of the simulation", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    scene.add(makeGround(scene));
    scene.add(makeFallingBox(scene, PhysicsConfig::Shape::Auto));

    PlayController controller;
    controller.addSession(std::make_shared<PhysicsPlaySession>());

    std::string error;
    REQUIRE(controller.play(document, &error));

    for (int i = 0; i < 120; ++i) controller.update(1.f / 60.f);

    auto* played = document.scene().getObjectByName("Falling");
    REQUIRE(played);
    CHECK(played->position.y < 3.f);// it fell

    REQUIRE(controller.stop(document, &error));

    auto* restored = document.scene().getObjectByName("Falling");
    REQUIRE(restored);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(restored->position.y, WithinAbs(4.f, 1e-5));
    CHECK(PhysicsConfig::read(*restored).has_value());
}
