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

#include "threepp/math/MathUtils.hpp"
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

    ConveyorWaypointConfig waypoint;
    waypoint.cornerRadius = 1.75f;
    waypoint.segKind = conveyor::SegKind::Cleats;
    CHECK(ConveyorWaypointConfig::decode(waypoint.encode()) == waypoint);

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
    wp.cornerRadius = 2.f;
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

TEST_CASE("attached walls are children, not waypoints, and snap their base to the deck") {

    Scene scene;
    auto conveyor = ObjectFactory::createConveyor(scene);
    scene.add(conveyor);
    auto config = ConveyorConfig::read(*conveyor).value();

    auto wall = ObjectFactory::createConveyorWall(*conveyor);
    conveyor->add(wall);

    // The wall child must NOT bend the path: waypoint bookkeeping skips it.
    CHECK(ConveyorConfig::waypointNodes(*conveyor).size() == 3);
    CHECK(ConveyorConfig::wallNodes(*conveyor).size() == 1);
    CHECK(ConveyorConfig::conveyorOf(*wall) == nullptr);
    const auto points = ConveyorWallConfig::pointNodes(*wall);
    REQUIRE(points.size() == 2);
    CHECK(ConveyorWallConfig::wallOf(*points[0]) == wall.get());

    // The spec composes wall points through the wall group's own transform:
    // rotate the group a quarter turn and the resolved span swings with it.
    auto spec = config.spec(*conveyor);
    REQUIRE(spec.walls.size() == 1);
    const Vector3 before = spec.walls[0].points[0];
    wall->rotation.y = math::PI / 2;
    wall->updateMatrix();
    spec = config.spec(*conveyor);
    const Vector3 rotated = spec.walls[0].points[0];
    CHECK(before.distanceTo(rotated) > 0.1f);
    wall->rotation.y = 0.f;

    // The default wall is ONE SHORT SEGMENT at the START of the belt — a
    // piece to slide and grow downstream, not a plan to fight. Length ~1.5x
    // the belt width.
    spec = config.spec(*conveyor);
    {
        const float span = spec.walls[0].points.front().distanceTo(spec.walls[0].points.back());
        CHECK(span > 0.5f);
        CHECK(span < 1.2f);
        const auto path = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
        const auto first = conveyor::projectOntoPath(spec.walls[0].points.front(), path);
        CHECK(first.station < 0.3f);// where building naturally begins
    }

    // Walls come in SECTIONS with open stretches between them, and both edges
    // eventually want some: repeated Add Wall TILES the same side, each new
    // section a gap after the previous, and starts over on the other side
    // when the first runs out of belt.
    {
        auto second = ObjectFactory::createConveyorWall(*conveyor);
        conveyor->add(second);
        auto third = ObjectFactory::createConveyorWall(*conveyor);
        conveyor->add(third);

        const auto tiled = config.spec(*conveyor);
        REQUIRE(tiled.walls.size() == 3);
        const auto path = conveyor::resamplePath(tiled.waypoints, tiled.smooth, tiled.samples);
        const auto endOfFirst = conveyor::projectOntoPath(tiled.walls[0].points.back(), path);
        const auto startOfSecond = conveyor::projectOntoPath(tiled.walls[1].points.front(), path);
        const auto startOfThird = conveyor::projectOntoPath(tiled.walls[2].points.front(), path);

        // Second: same side, a genuine gap downstream of the first.
        CHECK(endOfFirst.offset * startOfSecond.offset > 0.f);
        CHECK(startOfSecond.station > endOfFirst.station + 0.3f);
        // Third: that side is built out, so it starts over across the belt.
        CHECK(startOfThird.offset * endOfFirst.offset < 0.f);
        CHECK(startOfThird.station < 0.3f);

        third->removeFromParent();
        second->removeFromParent();
    }

    // And a PASSIVE EDGE GUIDE: every built sample rides at the belt edge
    // (half width + clearance) at deck height — inward offsets are the
    // user's edits, not the factory's guesses.
    const auto centerline = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
    const auto followed = conveyor::followWall(spec.walls[0].points, centerline);
    REQUIRE(followed.size() >= 2);
    const float edge = spec.width * 0.5f + 0.03f;
    for (const auto& p : followed) {
        CHECK(std::abs(std::abs(p.z) - edge) < 0.02f);
        CHECK(std::abs(p.y - 0.75f) < 1e-3f);// base on the deck
    }

    // Dragging a point toward the middle sweeps that stretch inward — the
    // wall still follows the path, its offset blending to the new value.
    auto plow = spec.walls[0];
    plow.points.back().z = 0.f;// end point pulled to the centreline
    const auto swept = conveyor::followWall(plow.points, centerline);
    REQUIRE(swept.size() >= 2);
    CHECK(std::abs(swept.back().z) < 0.05f);
    CHECK(std::abs(swept.front().z) > edge - 0.05f);

    // And the generated content grows the wall ribbon.
    config.syncDerived(*conveyor);
    CHECK(roleCount(*conveyor, "wall") == 1);
}

TEST_CASE("an edge wall HUGS a bend instead of cutting the chord") {

    Scene scene;
    auto conveyor = ObjectFactory::createConveyor(scene);
    scene.add(conveyor);
    auto config = ConveyorConfig::read(*conveyor).value();

    // A right-angle bend, radius 1.2.
    auto nodes = ConveyorConfig::waypointNodes(*conveyor);
    nodes[0]->position.set(-2.f, 0.75f, 0.f);
    nodes[1]->position.set(0.f, 0.75f, 0.f);
    nodes[2]->position.set(0.f, 0.75f, 2.f);
    ConveyorWaypointConfig corner;
    corner.cornerRadius = 1.2f;
    corner.write(*nodes[1]);

    auto wall = ObjectFactory::createConveyorWall(*conveyor);
    conveyor->add(wall);

    const auto spec = config.spec(*conveyor);
    REQUIRE(spec.walls.size() == 1);
    const auto centerline = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
    const auto followed = conveyor::followWall(spec.walls[0].points, centerline);
    REQUIRE(followed.size() >= 6);

    // Every built sample keeps the edge offset from the PATH — around the arc
    // too. Distance to the polyline's SEGMENTS (a corner-walk path keeps its
    // straights as single long spans). A straight chord between the wall's
    // end points would cut inside by far more than this tolerance.
    const float edge = spec.width * 0.5f + 0.03f;
    float worst = 0.f;
    for (const auto& p : followed) {
        float nearest = 1e30f;
        for (std::size_t i = 0; i + 1 < centerline.size(); ++i) {
            const auto& a = centerline[i];
            const auto& b = centerline[i + 1];
            const float abx = b.x - a.x, abz = b.z - a.z;
            const float len2 = abx * abx + abz * abz;
            if (len2 < 1e-10f) continue;
            const float u = std::clamp(
                    ((p.x - a.x) * abx + (p.z - a.z) * abz) / len2, 0.f, 1.f);
            nearest = std::min(nearest, std::hypot(p.x - (a.x + abx * u),
                                                   p.z - (a.z + abz * u)));
        }
        worst = std::max(worst, std::abs(nearest - edge));
    }
    CHECK(worst < 0.06f);

    // The station/offset coordinate system round-trips — including on the
    // arc, which is what path-aware point insertion leans on.
    for (const float s : {0.4f, 1.7f, 2.9f}) {
        for (const float o : {-0.4f, 0.25f}) {
            const Vector3 p = conveyor::pointOnPath(centerline, s, o);
            const auto back = conveyor::projectOntoPath(p, centerline);
            CHECK(std::abs(back.station - s) < 0.05f);
            CHECK(std::abs(back.offset - o) < 0.03f);
        }
    }
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

TEST_CASE("a rounded corner resolves to a tangent fillet, clamped to its segments") {

    using namespace threepp::conveyor;

    // A right-angle corner at P, radius 0.8: the arc must enter along +x,
    // leave along +z, and sit on the derived centre at exactly that radius.
    std::vector<Waypoint> wps(3);
    wps[0].pos.set(0.f, 0.f, 0.f);
    wps[1].pos.set(2.f, 0.f, 0.f);
    wps[1].cornerRadius = 0.8f;
    wps[2].pos.set(2.f, 0.f, 2.f);

    const auto fillet = cornerFillet(wps, 1);
    REQUIRE(fillet.valid);
    CHECK(fillet.radius == Catch::Approx(0.8f));
    // Tangent points sit ON their segments, offset d = r for a right angle.
    CHECK(fillet.t1.distanceTo(Vector3(1.2f, 0.f, 0.f)) < 1e-4f);
    CHECK(fillet.t2.distanceTo(Vector3(2.f, 0.f, 0.8f)) < 1e-4f);
    // The spokes are perpendicular to the segments — tangency, as one number.
    CHECK(std::abs(fillet.t1.x - fillet.centre.x) < 1e-4f);
    CHECK(std::abs(fillet.t2.z - fillet.centre.z) < 1e-4f);

    // The sampled path: starts and ends on the waypoints, every arc point on
    // the derived circle, and NO KINK anywhere — consecutive directions never
    // turn more than the arc's own per-step angle (plus slack). This is the
    // property the old authored-arc-centre model could not promise.
    const auto pts = resamplePath(wps, true, 12);
    REQUIRE(pts.size() >= 8);
    CHECK(pts.front().distanceTo(wps[0].pos) < 1e-4f);
    CHECK(pts.back().distanceTo(wps[2].pos) < 1e-4f);
    float maxTurn = 0.f;
    for (std::size_t i = 0; i + 2 < pts.size(); ++i) {
        Vector3 d1, d2;
        d1.subVectors(pts[i + 1], pts[i]).normalize();
        d2.subVectors(pts[i + 2], pts[i + 1]).normalize();
        maxTurn = std::max(maxTurn, std::acos(std::clamp(d1.dot(d2), -1.f, 1.f)));
    }
    const float perStep = std::abs(fillet.sweep) / 2.f;// steps >= 2 per quarter
    CHECK(maxTurn < perStep + 0.05f);

    // An impossible radius CLAMPS instead of overrunning: the tangent points
    // stay within their segments and the fillet stays tangent.
    wps[1].cornerRadius = 100.f;
    const auto clamped = cornerFillet(wps, 1);
    REQUIRE(clamped.valid);
    CHECK(clamped.radius < 100.f);
    CHECK(clamped.t1.x >= 0.f);
    CHECK(clamped.t2.z <= 2.f);
    CHECK(std::abs(clamped.t1.x - clamped.centre.x) < 1e-3f);

    // Chained corners split the straight they share instead of overlapping.
    std::vector<Waypoint> chain(4);
    chain[0].pos.set(0.f, 0.f, 0.f);
    chain[1].pos.set(2.f, 0.f, 0.f);
    chain[1].cornerRadius = 50.f;
    chain[2].pos.set(2.f, 0.f, 2.f);
    chain[2].cornerRadius = 50.f;
    chain[3].pos.set(0.f, 0.f, 2.f);
    const auto first = cornerFillet(chain, 1);
    const auto second = cornerFillet(chain, 2);
    REQUIRE((first.valid && second.valid));
    // Both trimmed onto the shared 2 m segment, meeting in its middle at most.
    CHECK(first.t2.z <= 1.f + 1e-3f);
    CHECK(second.t1.z >= 1.f - 1e-3f);

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
