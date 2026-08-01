// The PhysX-free half of the conveyor feature: the userData schema, the
// waypoint model, and the generated content. What this file pins down is the
// EXPORT contract — an authored conveyor is plain scene content (config string,
// waypoint children, tagged parts group), so everything here must hold on a
// loaded document with no editor application anywhere near it.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/conveyor/ConveyorGeometry.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::size_t roleCount(const Object3D& conveyorNode, const char* role) {

        std::size_t n = 0;
        if (auto* group = ConveyorConfig::derivedGroup(conveyorNode)) {
            const_cast<Object3D*>(group)->traverse([&](Object3D& o) {
                if (ConveyorConfig::roleOf(o) == role) ++n;
            });
        }
        return n;
    }

}// namespace


TEST_CASE("ConveyorConfig encodes and decodes every field") {

    ConveyorConfig config;
    config.width = 0.85f;
    config.speed = 1.4f;
    config.reverse = true;
    config.smooth = false;
    config.separator = true;
    config.wallHeight = 0.75f;
    config.rollerRadius = 0.07f;
    config.cleatHeight = 0.2f;
    config.cleatSpacing = 0.45f;
    config.samples = 9;
    config.frame = false;

    const auto decoded = ConveyorConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == config);

    // Unknown keys are ignored, known ones still land — the forwards
    // compatibility rule every flat config in the editor follows.
    const auto partial = ConveyorConfig::decode("width=1.25;futureKey=7;speed=2");
    REQUIRE(partial.has_value());
    CHECK(partial->width == Catch::Approx(1.25f));
    CHECK(partial->speed == Catch::Approx(2.f));
    CHECK(partial->smooth);// untouched default
}

TEST_CASE("per-waypoint config lives on the waypoint node and defaults to absent") {

    Scene scene;
    auto conveyor = ObjectFactory::createConveyor(scene);
    scene.add(conveyor);

    auto nodes = ConveyorConfig::waypointNodes(*conveyor);
    REQUIRE(nodes.size() == 3);

    // Nothing authored: every waypoint reads as a plain flat point.
    CHECK(ConveyorWaypointConfig::read(*nodes[0]) == ConveyorWaypointConfig{});

    ConveyorWaypointConfig wp;
    wp.arcCenter = true;
    wp.segKind = conveyor::SegKind::Rollers;
    wp.write(*nodes[1]);
    CHECK(ConveyorWaypointConfig::read(*nodes[1]) == wp);

    // Writing the default ERASES the entry — no key, same meaning.
    ConveyorWaypointConfig{}.write(*nodes[1]);
    CHECK(!nodes[1]->userData.contains(ConveyorWaypointConfig::userDataKey));
}

TEST_CASE("the resolved spec merges the config with the waypoint children") {

    Scene scene;
    auto conveyor = ObjectFactory::createConveyor(scene);
    scene.add(conveyor);

    auto config = ConveyorConfig::read(*conveyor).value();
    config.width = 1.f;
    config.write(*conveyor);

    auto nodes = ConveyorConfig::waypointNodes(*conveyor);
    ConveyorWaypointConfig wp;
    wp.segKind = conveyor::SegKind::Cleats;
    wp.write(*nodes[0]);

    const auto spec = config.spec(*conveyor);
    REQUIRE(spec.waypoints.size() == 3);
    CHECK(spec.width == Catch::Approx(1.f));
    CHECK(spec.waypoints[0].segKind == conveyor::SegKind::Cleats);
    CHECK(spec.waypoints[1].segKind == conveyor::SegKind::Flat);
    CHECK(spec.waypoints[0].pos.distanceTo(nodes[0]->position) < 1e-6f);
}

TEST_CASE("syncDerived generates the parts group, wholesale and re-adoptable") {

    Scene scene;
    auto conveyor = ObjectFactory::createConveyor(scene);
    scene.add(conveyor);

    auto config = ConveyorConfig::read(*conveyor).value();
    config.syncDerived(*conveyor);

    auto* group = ConveyorConfig::derivedGroup(*conveyor);
    REQUIRE(group != nullptr);
    const auto groupUuid = group->uuid;

    CHECK(roleCount(*conveyor, "belt") == 1);
    CHECK(roleCount(*conveyor, "drum") == 2);
    CHECK(roleCount(*conveyor, "frame") >= 4);// two rails + legs
    CHECK(roleCount(*conveyor, "roller") == 0);

    // The parts are NOT waypoints: the tag is what separates them.
    CHECK(ConveyorConfig::waypointNodes(*conveyor).size() == 3);

    // Regeneration keeps the group node (uuid-stable), replaces the content.
    auto nodes = ConveyorConfig::waypointNodes(*conveyor);
    ConveyorWaypointConfig wp;
    wp.segKind = conveyor::SegKind::Rollers;
    wp.write(*nodes[0]);
    config.syncDerived(*conveyor);
    CHECK(ConveyorConfig::derivedGroup(*conveyor)->uuid == groupUuid);
    CHECK(roleCount(*conveyor, "roller") >= 3);

    // A separator is a wall and nothing else.
    config.separator = true;
    config.syncDerived(*conveyor);
    CHECK(roleCount(*conveyor, "wall") == 1);
    CHECK(roleCount(*conveyor, "belt") == 0);
    CHECK(roleCount(*conveyor, "frame") == 0);
}

TEST_CASE("a conveyor round-trips the document with no editor attached") {

    SceneDocument authoring;
    auto conveyor = ObjectFactory::createConveyor(authoring.scene());
    authoring.scene().add(conveyor);

    auto config = ConveyorConfig::read(*conveyor).value();
    config.speed = 1.2f;
    config.width = 0.7f;
    config.write(*conveyor);
    auto nodes = ConveyorConfig::waypointNodes(*conveyor);
    ConveyorWaypointConfig wp;
    wp.segKind = conveyor::SegKind::Cleats;
    wp.write(*nodes[1]);
    config.syncDerived(*conveyor);
    const auto conveyorUuid = conveyor->uuid;

    std::string error;
    const auto json = authoring.toJson(false, &error);
    REQUIRE(error.empty());
    REQUIRE(!json.empty());

    SceneDocument loaded;
    REQUIRE(loaded.openJson(json, &error));

    Object3D* found = nullptr;
    loaded.scene().traverse([&](Object3D& o) {
        if (ConveyorConfig::isConveyor(o)) found = &o;
    });
    REQUIRE(found != nullptr);
    CHECK(found->uuid == conveyorUuid);

    const auto reloaded = ConveyorConfig::read(*found);
    REQUIRE(reloaded.has_value());
    CHECK(*reloaded == config);

    auto reloadedNodes = ConveyorConfig::waypointNodes(*found);
    REQUIRE(reloadedNodes.size() == 3);
    CHECK(ConveyorWaypointConfig::read(*reloadedNodes[1]).segKind == conveyor::SegKind::Cleats);

    // The saved document carries the LOOK: belt, cleat bars, frame — so a
    // consumer that only renders needs nothing regenerated.
    CHECK(roleCount(*found, "belt") >= 1);
    CHECK(roleCount(*found, "cleat") >= 1);
    CHECK(roleCount(*found, "frame") >= 4);

    // And the resolved spec — what the physics side consumes — matches the
    // authored one, waypoint for waypoint.
    const auto spec = reloaded->spec(*found);
    REQUIRE(spec.waypoints.size() == 3);
    CHECK(spec.speed == Catch::Approx(1.2f));
    CHECK(spec.waypoints[1].segKind == conveyor::SegKind::Cleats);
}

TEST_CASE("path resampling honours arcs and per-segment kinds") {

    using namespace threepp::conveyor;

    // A quarter-circle bend: A at angle -90 about C, B at angle 0, radius 2.
    std::vector<Waypoint> wps(3);
    wps[0].pos.set(0.f, 0.f, 0.f);
    wps[1].pos.set(0.f, 0.f, 2.f);
    wps[1].arcCenter = true;
    wps[2].pos.set(2.f, 0.f, 2.f);

    const auto pts = resamplePath(wps, true, 12);
    REQUIRE(pts.size() >= 8);
    // Every sampled point past the start sits on the radius-2 circle about C.
    for (std::size_t i = 1; i < pts.size(); ++i) {
        const float r = std::hypot(pts[i].x - 0.f, pts[i].z - 2.f);
        CHECK(std::abs(r - 2.f) < 1e-3f);
    }
    // And it ends exactly at B.
    CHECK(pts.back().distanceTo(wps[2].pos) < 1e-4f);

    // Kind runs: flat, then rollers, then flat again — three runs sharing
    // boundary points so the surface meets gap-free.
    std::vector<Waypoint> lane(4);
    lane[0].pos.set(0.f, 0.f, 0.f);
    lane[1].pos.set(1.f, 0.f, 0.f);
    lane[1].segKind = SegKind::Rollers;
    lane[2].pos.set(2.f, 0.f, 0.f);
    lane[3].pos.set(3.f, 0.f, 0.f);

    const auto runs = resamplePathByKind(lane, false, 12);
    REQUIRE(runs.size() == 3);
    CHECK(runs[0].kind == SegKind::Flat);
    CHECK(runs[1].kind == SegKind::Rollers);
    CHECK(runs[2].kind == SegKind::Flat);
    CHECK(runs[0].pts.back().distanceTo(runs[1].pts.front()) < 1e-6f);
    CHECK(runs[1].pts.back().distanceTo(runs[2].pts.front()) < 1e-6f);
}
