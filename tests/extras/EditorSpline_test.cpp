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
    config.write(*group);

    CHECK(SplineConfig::isSpline(*group));
    CHECK(group->userData.contains(SplineConfig::userDataKey));

    const auto read = SplineConfig::read(*group);
    REQUIRE(read.has_value());
    CHECK(*read == config);

    // Deterministic encoding: an unchanged value produces the same bytes every
    // time, which is what keeps saved documents diff-clean.
    CHECK(config.encode() == "type=catmullrom;closed=1;tension=0.25;samples=40");
    CHECK(config.encode() == read->encode());

    // The key's presence is the definition, so an entry that says nothing is
    // still a spline — with every parameter at its default.
    const auto empty = SplineConfig::decode("");
    REQUIRE(empty.has_value());
    CHECK(*empty == SplineConfig{});

    // Unknown keys are ignored, so a document written by a newer editor loads.
    const auto forward = SplineConfig::decode("type=chordal;radius=3;samples=8");
    REQUIRE(forward.has_value());
    CHECK(forward->type == SplineConfig::Type::Chordal);
    CHECK(forward->samples == 8);

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
