// The export path, end to end: a conveyor authored as scene content, saved to
// the document format, loaded back with NO editor application anywhere, and
// rebuilt into working belt physics against a plain PhysxWorld. This is the
// contract that lets an external consumer (a soft-body sim, a headless data
// generator) run editor-authored conveyors — the fish-simulation workflow,
// with the scene document in place of layout.json.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

#include <PxPhysicsAPI.h>

#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Author a straight belt at working height, save it, load it into a fresh
    // document, and resolve every conveyor to a world-space spec — exactly the
    // walk an external consumer performs.
    std::vector<conveyor::ConveyorSpec> exportedSpecs(float speed, bool reverse) {

        SceneDocument authoring;
        auto conveyorNode = ObjectFactory::createConveyor(authoring.scene());
        authoring.scene().add(conveyorNode);

        auto config = ConveyorConfig::read(*conveyorNode).value();
        config.speed = speed;
        config.reverse = reverse;
        config.write(*conveyorNode);
        config.syncDerived(*conveyorNode);

        std::string error;
        const auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());

        auto loaded = std::make_shared<SceneDocument>();
        REQUIRE(loaded->openJson(json, &error));

        std::vector<conveyor::ConveyorSpec> specs;
        loaded->scene().updateMatrixWorld(true);
        loaded->scene().traverse([&](Object3D& object) {
            const auto found = ConveyorConfig::read(object);
            if (!found) return;
            auto spec = found->spec(object);
            object.updateWorldMatrix(true, false);
            for (auto& wp : spec.waypoints) wp.pos.applyMatrix4(*object.matrixWorld);
            specs.push_back(std::move(spec));
        });
        return specs;
    }

    ::physx::PxRigidDynamic* dropCargo(PhysxWorld& world, float x, float y, float z) {

        using namespace ::physx;

        auto* material = world.physics().createMaterial(0.6f, 0.6f, 0.f);
        auto* box = world.physics().createRigidDynamic(PxTransform(PxVec3(x, y, z)));
        auto* shape = world.physics().createShape(PxBoxGeometry(0.2f, 0.2f, 0.2f), *material, true);
        box->attachShape(*shape);
        shape->release();
        PxRigidBodyExt::setMassAndUpdateInertia(*box, 5.f);
        world.scene().addActor(*box);
        return box;
    }

}// namespace


TEST_CASE("a saved conveyor rebuilds belt physics that conveys, without the editor") {

    const auto specs = exportedSpecs(1.f, false);
    REQUIRE(specs.size() == 1);
    REQUIRE(specs[0].waypoints.size() == 3);
    CHECK(specs[0].speed == Catch::Approx(1.f));

    PhysxWorld world;
    conveyor::ConveyorPhysics belts(world, specs);
    CHECK(belts.beltCount() >= 2);

    // Dropped a hair over the upstream end (belt surface y = 0.75, travel +x).
    auto* cargo = dropCargo(world, -1.2f, 1.f, 0.f);

    // Sampled DURING the run: the belt ends at x = 1.5 and cargo that conveys
    // to the end falls off it — which is correct behaviour, not the assertion.
    bool conveyed = false;
    for (int i = 0; i < 300 && !conveyed; ++i) {
        world.step(1.f / 60.f);
        const auto p = cargo->getGlobalPose().p;
        // A metre of travel while ON the belt and still centred — the surface
        // dragged it, nothing pushed it sideways.
        if (p.x > -1.2f + 0.8f && p.y > 0.6f && std::abs(p.z) < 0.3f) conveyed = true;
    }
    CHECK(conveyed);
}

TEST_CASE("a rollers run conveys on REAL roller colliders, with no belt box underneath") {

    // Author a conveyor whose every segment is a roller bed, straight to the
    // spec (no document round-trip needed — the schema is covered elsewhere).
    SceneDocument authoring;
    auto conveyorNode = ObjectFactory::createConveyor(authoring.scene());
    authoring.scene().add(conveyorNode);
    auto config = ConveyorConfig::read(*conveyorNode).value();
    config.speed = 1.f;
    config.write(*conveyorNode);
    ConveyorWaypointConfig rollers;
    rollers.segKind = conveyor::SegKind::Rollers;
    for (auto* node : ConveyorConfig::waypointNodes(*conveyorNode)) rollers.write(*node);

    auto spec = config.spec(*conveyorNode);

    // Scoped: only one PhysX foundation may exist, so the forward world must
    // be gone before the reversed one comes up.
    {
        PhysxWorld world;
        conveyor::ConveyorPhysics belts(world, {spec});

        // The claim in numbers: rollers exist, drag boxes do not — whatever
        // carries the cargo below can only be the spinning capsules.
        CHECK(belts.beltCount() == 0);
        CHECK(belts.rollerCount() >= 10);

        auto* cargo = dropCargo(world, -1.2f, 1.f, 0.f);
        bool conveyed = false;
        for (int i = 0; i < 300 && !conveyed; ++i) {
            world.step(1.f / 60.f);
            const auto p = cargo->getGlobalPose().p;
            // Riding AT roller height (top of the bed = the authored surface),
            // carried along by rolling contact alone.
            if (p.x > -1.2f + 0.8f && p.y > 0.6f && std::abs(p.z) < 0.3f) conveyed = true;
        }
        CHECK(conveyed);
    }

    // And reverse spins the same rollers the other way.
    spec.reverse = true;
    PhysxWorld reversedWorld;
    conveyor::ConveyorPhysics reversedBelts(reversedWorld, {spec});
    auto* back = dropCargo(reversedWorld, 1.2f, 1.f, 0.f);
    bool conveyedBack = false;
    for (int i = 0; i < 300 && !conveyedBack; ++i) {
        reversedWorld.step(1.f / 60.f);
        const auto p = back->getGlobalPose().p;
        if (p.x < 1.2f - 0.8f && p.y > 0.6f) conveyedBack = true;
    }
    CHECK(conveyedBack);
}

TEST_CASE("an attached diverter wall FEEDS cargo across the belt") {

    // The factory's default wall is a plow angled across the path midpoint —
    // author it, save nothing, build physics straight from the spec.
    SceneDocument authoring;
    auto conveyorNode = ObjectFactory::createConveyor(authoring.scene());
    authoring.scene().add(conveyorNode);
    auto config = ConveyorConfig::read(*conveyorNode).value();
    config.speed = 1.f;
    config.write(*conveyorNode);
    auto wall = ObjectFactory::createConveyorWall(*conveyorNode);
    conveyorNode->add(wall);

    const auto spec = config.spec(*conveyorNode);
    REQUIRE(spec.walls.size() == 1);
    REQUIRE(spec.walls[0].points.size() == 2);

    PhysxWorld world;
    conveyor::ConveyorPhysics belts(world, {spec});
    CHECK(belts.wallCount() >= 1);

    // Cargo down the CENTRE of the belt: without the wall it would ride
    // z = 0 the whole way (the straight-belt case pins that). The plow leans
    // from the +z edge to past centre, so a diverted box exits pushed to -z —
    // the wall put it where it should go.
    auto* cargo = dropCargo(world, -1.2f, 1.f, 0.f);
    bool diverted = false;
    for (int i = 0; i < 400 && !diverted; ++i) {
        world.step(1.f / 60.f);
        const auto p = cargo->getGlobalPose().p;
        if (p.y < 0.4f) break;// fell off — that is a failure, not a divert
        if (p.x > 0.6f && p.z < -0.15f) diverted = true;
    }
    CHECK(diverted);
}

TEST_CASE("reverse flips the travel direction, same document") {

    const auto specs = exportedSpecs(1.f, true);
    REQUIRE(specs.size() == 1);

    PhysxWorld world;
    conveyor::ConveyorPhysics belts(world, specs);

    auto* cargo = dropCargo(world, 1.2f, 1.f, 0.f);
    bool conveyed = false;
    for (int i = 0; i < 300 && !conveyed; ++i) {
        world.step(1.f / 60.f);
        const auto p = cargo->getGlobalPose().p;
        if (p.x < 1.2f - 0.8f && p.y > 0.6f) conveyed = true;
    }
    CHECK(conveyed);
}

TEST_CASE("tearing the belts down mid-run leaves a healthy world") {

    const auto specs = exportedSpecs(1.f, false);

    PhysxWorld world;
    auto* cargo = dropCargo(world, -1.2f, 1.f, 0.f);

    {
        conveyor::ConveyorPhysics belts(world, specs);
        for (int i = 0; i < 60; ++i) world.step(1.f / 60.f);
    }
    // The belts unregistered their substep hooks and released their actors on
    // destruction; the world must keep stepping without them (the cargo now
    // simply falls, which is not the assertion — not crashing is).
    for (int i = 0; i < 60; ++i) world.step(1.f / 60.f);
    CHECK(cargo->getGlobalPose().p.y < 1.f);
}
