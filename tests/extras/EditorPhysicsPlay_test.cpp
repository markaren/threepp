
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <iostream>

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

    std::shared_ptr<Mesh> makeSoftBall(Scene& scene) {

        auto ball = ObjectFactory::createPrimitive(Primitive::Sphere, scene);
        ball->name = "Jelly";
        ball->position.set(0.f, 3.f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Soft;
        config.mass = 2.f;
        config.friction = 0.5f;
        // Soft enough that a 3 m drop visibly squashes it, coarse enough that
        // the cook stays quick.
        config.youngsModulus = 5e4f;
        config.voxelResolution = 6;
        config.solverIterations = 15;
        config.write(*ball);
        return ball;
    }

    void run(PhysicsPlaySession& session, int steps) {

        for (int i = 0; i < steps; ++i) session.update(1.f / 60.f);
    }

    // A soft body's mesh IS the simulation output: its vertices are rewritten
    // in world space every step, so the shape is read off the position buffer
    // rather than off the transform.
    struct VertexBounds {
        float minY = 1e30f;
        float maxY = -1e30f;
        float meanY = 0.f;
    };

    VertexBounds boundsOf(const Mesh& mesh) {

        VertexBounds out;
        const auto* positions = mesh.geometry()->getAttribute<float>("position");
        REQUIRE(positions);
        double sum = 0;
        for (unsigned i = 0; i < positions->count(); ++i) {
            const float y = positions->getY(i);
            out.minY = std::min(out.minY, y);
            out.maxY = std::max(out.maxY, y);
            sum += y;
        }
        out.meanY = static_cast<float>(sum / positions->count());
        return out;
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

TEST_CASE("soft body config round-trips through the userData string", "[editor][physx]") {

    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Soft;
    config.mass = 3.5f;
    config.friction = 0.7f;
    config.youngsModulus = 25000.f;
    config.poissonsRatio = 0.3f;
    config.voxelResolution = 14;
    config.solverIterations = 32;
    config.selfCollision = true;

    const auto decoded = PhysicsConfig::decode(config.encode());
    REQUIRE(decoded);
    CHECK(decoded->body == PhysicsConfig::Body::Soft);
    CHECK(decoded->voxelResolution == 14);
    CHECK(decoded->solverIterations == 32);
    CHECK(decoded->selfCollision);
    CHECK(decoded->encode() == config.encode());

    // Soft parameters survive a document written before they existed, and an
    // entry that never mentions them keeps the defaults.
    const auto legacy = PhysicsConfig::decode("body=dynamic;shape=box;mass=2;friction=0.5;restitution=0.1");
    REQUIRE(legacy);
    CHECK(legacy->body == PhysicsConfig::Body::Dynamic);
    CHECK(legacy->voxelResolution == PhysicsConfig{}.voxelResolution);
    CHECK(legacy->youngsModulus == PhysicsConfig{}.youngsModulus);
}

TEST_CASE("a soft body deforms and settles on the ground", "[editor][physx][gpu]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto ground = makeGround(scene);
    auto ball = makeSoftBall(scene);
    scene.add(ground);
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);

    if (!session.gpuAvailable()) {
        // No CUDA device on this machine: the rigid half must still have run,
        // and the soft body is simply absent.
        std::cout << "[skip] no CUDA device - soft body simulation not exercised" << std::endl;
        CHECK(session.softBodyCount() == 0);
        CHECK(session.bodyCount() == 1);// the ground
        session.stop();
        return;
    }

    CHECK(session.softBodyCount() == 1);
    CHECK(session.bodyCount() == 2);

    // addSoftBody bakes the world matrix into the geometry, so the mesh sits at
    // the origin with world-space vertices from here on.
    const auto spawned = boundsOf(*ball);
    CHECK(spawned.minY > 2.f);

    // Sample every step: the deepest squash is at the impact instant, not in
    // the settled pose, and "it deformed at all" is too weak a claim to catch
    // a soft body that is being simulated as a rigid one.
    const float restHeight = spawned.maxY - spawned.minY;
    float flattest = restHeight;
    for (int i = 0; i < 240; ++i) {// 4 seconds
        session.update(1.f / 60.f);
        const auto now = boundsOf(*ball);
        flattest = std::min(flattest, now.maxY - now.minY);
    }

    const auto settled = boundsOf(*ball);
    // It fell to the ground...
    CHECK(settled.minY < 0.4f);
    CHECK(settled.meanY < 1.f);
    // ...and it did not fall through it.
    CHECK(settled.minY > -0.5f);
    // ...and it is a soft body: a rigid sphere dropped from 3 m would still be
    // exactly as tall on landing. 10% is well clear of solver noise while
    // leaving room for a stiffer default than this test authors.
    INFO("rest height " << restHeight << " m, flattest " << flattest << " m");
    CHECK(flattest < restHeight * 0.9f);

    session.stop();
    CHECK(session.softBodyCount() == 0);
}

TEST_CASE("stop restores the mesh a soft body deformed", "[editor][physx][gpu]") {

    SceneDocument document;
    auto& scene = document.scene();
    scene.add(makeGround(scene));
    scene.add(makeSoftBall(scene));

    PlayController controller;
    controller.addSession(std::make_shared<PhysicsPlaySession>());

    std::string error;
    REQUIRE(controller.play(document, &error));
    for (int i = 0; i < 120; ++i) controller.update(1.f / 60.f);
    REQUIRE(controller.stop(document, &error));

    // Whether or not the machine had a GPU, Stop must hand back the authored
    // mesh: rest-pose geometry, authored transform, physics entry intact.
    auto* restored = document.scene().getObjectByName("Jelly");
    REQUIRE(restored);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(restored->position.y, WithinAbs(3.f, 1e-5));

    const auto config = PhysicsConfig::read(*restored);
    REQUIRE(config);
    CHECK(config->body == PhysicsConfig::Body::Soft);

    auto* restoredMesh = restored->as<Mesh>();
    REQUIRE(restoredMesh);
    const auto bounds = boundsOf(*restoredMesh);
    // Rest-pose sphere geometry: local space, centred on the origin.
    CHECK_THAT(bounds.minY, WithinAbs(-0.5f, 1e-3));
    CHECK_THAT(bounds.maxY, WithinAbs(0.5f, 1e-3));
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
