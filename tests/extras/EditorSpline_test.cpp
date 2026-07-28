#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>

using namespace threepp;
using namespace threepp::editor;


TEST_CASE("SplineConfig round-trips through userData", "[editor]") {

    auto group = Group::create();

    // Nothing attached: not a spline, and read() says so rather than handing
    // back a default-constructed config nobody asked for.
    CHECK_FALSE(SplineConfig::isSpline(*group));
    CHECK_FALSE(SplineConfig::read(*group).has_value());

    SplineConfig config;
    config.type = SplineConfig::Type::CatmullRom;
    config.closed = true;
    config.tension = 0.25f;
    config.samples = 40;
    config.mesh = SplineConfig::MeshKind::Tube;
    config.radius = 0.75f;
    config.radialSegments = 12;
    config.write(*group);

    CHECK(SplineConfig::isSpline(*group));
    CHECK(group->userData.contains(SplineConfig::userDataKey));

    const auto read = SplineConfig::read(*group);
    REQUIRE(read.has_value());
    CHECK(*read == config);

    // Deterministic encoding: an unchanged value produces the same bytes every
    // time, which is what keeps saved documents diff-clean.
    CHECK(config.encode() == "type=catmullrom;closed=1;tension=0.25;samples=40;"
                             "mesh=tube;radius=0.75;radialSegments=12");
    CHECK(config.encode() == read->encode());

    // The key's presence is the definition, so an entry that says nothing is
    // still a spline — with every parameter at its default.
    const auto empty = SplineConfig::decode("");
    REQUIRE(empty.has_value());
    CHECK(*empty == SplineConfig{});
    CHECK(empty->mesh == SplineConfig::MeshKind::None);

    // Unknown keys are ignored, so a document written by a newer editor loads.
    const auto forward = SplineConfig::decode("type=chordal;bevel=3;samples=8");
    REQUIRE(forward.has_value());
    CHECK(forward->type == SplineConfig::Type::Chordal);
    CHECK(forward->samples == 8);

    // A document written BEFORE the mesh keys existed reads as "no mesh",
    // which is what keeps every spline already saved unchanged on load.
    const auto legacy = SplineConfig::decode("type=chordal;closed=1;tension=0.5;samples=24");
    REQUIRE(legacy.has_value());
    CHECK(legacy->mesh == SplineConfig::MeshKind::None);
    CHECK(legacy->radius == SplineConfig{}.radius);
    CHECK(legacy->radialSegments == SplineConfig{}.radialSegments);

    SplineConfig::erase(*group);
    CHECK_FALSE(SplineConfig::isSpline(*group));
    CHECK_FALSE(group->userData.contains(SplineConfig::userDataKey));
}

TEST_CASE("every direct child of a spline is a control point", "[editor]") {

    auto spline = Group::create();
    SplineConfig{}.write(*spline);

    auto a = Object3D::create();
    a->position.set(-1, 0, 0);
    auto b = Object3D::create();
    b->position.set(0, 1, 0);
    auto c = Object3D::create();
    c->position.set(1, 0, 0);
    spline->add(a);
    spline->add(b);
    spline->add(c);

    const auto points = SplineConfig::controlPoints(*spline);
    REQUIRE(points.size() == 3);
    // Child order is point order.
    CHECK(points[0].x == -1.f);
    CHECK(points[1].y == 1.f);
    CHECK(points[2].x == 1.f);

    // Only DIRECT children count: a node parented to a point is not one.
    auto nested = Object3D::create();
    b->add(nested);
    CHECK(SplineConfig::controlPoints(*spline).size() == 3);
    CHECK(SplineConfig::splineOf(*b) == spline.get());
    CHECK(SplineConfig::splineOf(*nested) == nullptr);
}

TEST_CASE("the generated mesh is a child without being a control point", "[editor]") {

    auto spline = Group::create();
    SplineConfig{}.write(*spline);

    auto a = Object3D::create();
    a->position.set(-1, 0, 0);
    auto b = Object3D::create();
    b->position.set(1, 0, 0);
    spline->add(a);
    spline->add(b);

    CHECK(SplineConfig::derivedMesh(*spline) == nullptr);

    auto derived = Object3D::create();
    SplineConfig::markDerived(*derived);
    spline->add(derived);

    // Three children, two points. Every count and index the editor shows has
    // to come off the second number, not the first.
    CHECK(spline->children.size() == 3);
    CHECK(SplineConfig::isDerived(*derived));
    CHECK(SplineConfig::derivedMesh(*spline) == derived.get());
    CHECK(SplineConfig::controlPoints(*spline).size() == 2);
    CHECK(SplineConfig::controlPointNodes(*spline).size() == 2);
    // ... and it is not a control point of anything, so it gets no marker and
    // no point form of the inspector section.
    CHECK(SplineConfig::splineOf(*derived) == nullptr);
    CHECK(SplineConfig::splineOf(*a) == spline.get());

    CHECK(SplineConfig::pointIndexOf(*spline, *a) == 0);
    CHECK(SplineConfig::pointIndexOf(*spline, *b) == 1);
    // Not a point: reported as "past the end", never as index 2 (which is what
    // its child index would say).
    CHECK(SplineConfig::pointIndexOf(*spline, *derived) == 2);

    // Inserting point i means inserting at child index i here, because the
    // mesh is last — and appending goes BEFORE it, so it stays last.
    CHECK(SplineConfig::childSlotForPointIndex(*spline, 0) == 0);
    CHECK(SplineConfig::childSlotForPointIndex(*spline, 1) == 1);
    CHECK(SplineConfig::childSlotForPointIndex(*spline, 2) == 2);
    CHECK(SplineConfig::childSlotForPointIndex(*spline, 99) == 2);

    // Same answers with the mesh FIRST among the children, which is what a
    // hand-edited or re-ordered document can hand back.
    derived->removeFromParent();
    auto reordered = Group::create();
    SplineConfig{}.write(*reordered);
    auto lead = Object3D::create();
    SplineConfig::markDerived(*lead);
    auto c = Object3D::create();
    auto d = Object3D::create();
    reordered->add(lead);
    reordered->add(c);
    reordered->add(d);
    CHECK(SplineConfig::controlPoints(*reordered).size() == 2);
    CHECK(SplineConfig::pointIndexOf(*reordered, *c) == 0);
    CHECK(SplineConfig::pointIndexOf(*reordered, *d) == 1);
    CHECK(SplineConfig::childSlotForPointIndex(*reordered, 0) == 1);
    CHECK(SplineConfig::childSlotForPointIndex(*reordered, 1) == 2);
    CHECK(SplineConfig::childSlotForPointIndex(*reordered, 2) == 3);
}

TEST_CASE("samples-per-segment is one formula", "[editor]") {

    auto spline = Group::create();
    SplineConfig config;
    config.samples = 10;
    config.write(*spline);

    for (int i = 0; i < 4; ++i) spline->add(Object3D::create());

    // Four points, three segments open; the loop adds the closing one.
    CHECK(config.divisions(*spline) == 30);
    config.closed = true;
    CHECK(config.divisions(*spline) == 40);

    // The generated mesh must not inflate the tessellation by counting as a
    // point — the reason divisions() goes through controlPoints().
    auto derived = Object3D::create();
    SplineConfig::markDerived(*derived);
    spline->add(derived);
    CHECK(config.divisions(*spline) == 40);

    // Clamped, and never zero: one point is no segments at all.
    config.closed = false;
    config.samples = 10000;
    CHECK(config.divisions(*spline) == static_cast<unsigned int>(SplineConfig::maxSamples) * 3);
}

TEST_CASE("a spline builds a curve from its children", "[editor]") {

    auto spline = Group::create();
    const SplineConfig config;
    config.write(*spline);

    // Below two points there is no segment to sample.
    CHECK(config.curve(*spline) == nullptr);
    auto first = Object3D::create();
    first->position.set(0, 0, 0);
    spline->add(first);
    CHECK(config.curve(*spline) == nullptr);

    auto second = Object3D::create();
    second->position.set(2, 0, 0);
    spline->add(second);

    auto curve = config.curve(*spline);
    REQUIRE(curve != nullptr);
    CHECK(curve->points.size() == 2);
    CHECK_FALSE(curve->closed);

    Vector3 start;
    curve->getPoint(0.f, start);
    Vector3 end;
    curve->getPoint(1.f, end);
    CHECK(start.distanceTo(Vector3(0, 0, 0)) < 1e-4f);
    CHECK(end.distanceTo(Vector3(2, 0, 0)) < 1e-4f);

    // The parameters reach the curve, not just the document.
    SplineConfig closed;
    closed.closed = true;
    closed.type = SplineConfig::Type::CatmullRom;
    closed.tension = 0.75f;
    auto loop = closed.curve(*spline);
    REQUIRE(loop != nullptr);
    CHECK(loop->closed);
    CHECK(loop->curveType == CatmullRomCurve3::catmullrom);
    CHECK(std::abs(loop->tension - 0.75f) < 1e-6f);
}

TEST_CASE("the factory creates a spline that is already a curve", "[editor]") {

    Scene scene;

    auto spline = ObjectFactory::createSpline(scene);
    REQUIRE(spline != nullptr);
    CHECK(spline->name == "Spline");
    REQUIRE(SplineConfig::isSpline(*spline));
    CHECK(spline->children.size() == 4);

    // Point names are unique within the spline, and the first is unsuffixed.
    CHECK(spline->children[0]->name == "Point");
    CHECK(spline->children[1]->name == "Point 2");
    CHECK(spline->children[3]->name == "Point 4");

    // Not a straight line: a default that draws as a segment says nothing about
    // what a spline is.
    const auto points = SplineConfig::controlPoints(*spline);
    Vector3 mid;
    mid.copy(points.front()).add(points.back()).multiplyScalar(0.5f);
    CHECK(points[1].distanceTo(mid) > 1e-2f);

    const auto config = SplineConfig::read(*spline).value();
    auto curve = config.curve(*spline);
    REQUIRE(curve != nullptr);
    CHECK(curve->getLength() > 0.f);

    // A second one names itself around the first.
    scene.add(spline);
    auto another = ObjectFactory::createSpline(scene);
    CHECK(another->name == "Spline 2");
    CHECK(another->children[0]->name == "Point");
}
