
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadGeometry.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <vector>

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

    // A spline that has generated its road, standing in for what the editor's
    // sync pass produces: the Group carries SplineConfig, its plain children
    // are the control points, and one tagged Mesh holds the surface. Generating
    // that mesh is app code (SplineOverlay.cpp), so a test builds it by hand
    // from the same curve and the same parameters the editor passes.
    struct Road {
        std::shared_ptr<Group> spline;
        std::shared_ptr<Mesh> mesh;
    };

    Road makeRoad(const Scene& scene, const std::vector<Vector3>& points, float width) {

        auto spline = Group::create();
        spline->name = ObjectFactory::uniqueName(scene, "Spline");

        SplineConfig config;
        config.mesh = SplineConfig::MeshKind::Road;
        config.width = width;
        config.write(*spline);

        for (const auto& point : points) {
            auto node = Object3D::create();
            node->position.copy(point);
            spline->add(node);
        }

        auto curve = config.curve(*spline);
        REQUIRE(curve != nullptr);
        auto mesh = Mesh::create(RoadGeometry::create(
                *curve, width, config.divisions(*spline), config.uvLength, config.closed));
        mesh->name = "Road";
        SplineConfig::markDerived(*mesh);
        spline->add(mesh);

        return {spline, mesh};
    }

    PhysicsConfig staticConfig(PhysicsConfig::Shape shape) {

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Static;
        config.shape = shape;
        config.friction = 0.8f;
        return config;
    }

    // Radius 0.5, so resting on a surface at y = 0 means a centre at y = 0.5.
    std::shared_ptr<Mesh> makeFallingSphere(const Scene& scene, const Vector3& from) {

        auto sphere = ObjectFactory::createPrimitive(Primitive::Sphere, scene);
        sphere->name = "Ball";
        sphere->position.copy(from);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.shape = PhysicsConfig::Shape::Auto;
        config.mass = 2.f;
        config.friction = 0.8f;
        config.restitution = 0.f;
        config.write(*sphere);
        return sphere;
    }

    // Straight along X at y = 0, four metres wide.
    Road makeStraightRoad(const Scene& scene) {

        return makeRoad(scene, {Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(10, 0, 0)}, 4.f);
    }

    // Station intervals of a road surface: it is laid down one cross-section at
    // a time, two vertices each, and the collider is one convex hull per
    // interval between them.
    int intervalsOf(const Mesh& mesh) {

        const auto geometry = mesh.geometry();
        if (!geometry) return -1;
        const auto* position = geometry->getAttribute<float>("position");
        if (!position) return -1;
        return position->count() / 2 - 1;
    }

    // Shapes attached to every static actor in the live scene. A road is one
    // convex per station interval — a count driven by the road's SHAPE, since
    // that is what chooses the stations, and not by how finely anybody sampled
    // the spline.
    int staticShapeCount(PhysicsPlaySession& session) {

        using namespace ::physx;
        auto* world = session.world();
        if (!world) return -1;
        auto& scene = world->scene();
        const PxU32 count = scene.getNbActors(PxActorTypeFlag::eRIGID_STATIC);
        std::vector<PxActor*> actors(count);
        if (count) scene.getActors(PxActorTypeFlag::eRIGID_STATIC, actors.data(), count);
        int shapes = 0;
        for (auto* actor : actors) {
            shapes += static_cast<int>(static_cast<PxRigidActor*>(actor)->getNbShapes());
        }
        return shapes;
    }

    void checkRestsOnRoad(PhysicsPlaySession& session, const Mesh& ball) {

        run(session, 240);// 4 seconds

        using Catch::Matchers::WithinAbs;
        // Never below the surface, and settled at the radius on top of it.
        CHECK(ball.position.y > 0.f);
        CHECK_THAT(ball.position.y, WithinAbs(0.5f, 0.06f));
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

TEST_CASE("a generated road holds a body up, on Auto", "[editor][physx]") {

    // The report this exists for: a dynamic sphere dropped on a spline road
    // fell straight through, because Shape::Auto resolved a ribbon to its
    // BOUNDING BOX — a flat one at the minimum half-extent, i.e. a razor at the
    // road's mid-height, and nowhere near where the ball came down.
    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeStraightRoad(scene);
    staticConfig(PhysicsConfig::Shape::Auto).write(*road.mesh);
    scene.add(road.spline);

    // Four metres out along the road, deliberately clear of the spline's origin
    // where the old unit-box fallback put its phantom collider.
    auto ball = makeFallingSphere(scene, Vector3(4.f, 3.f, 0.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 2);

    checkRestsOnRoad(session, *ball);
}

TEST_CASE("a generated road holds a body up, on an explicit TriMesh", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeStraightRoad(scene);
    staticConfig(PhysicsConfig::Shape::TriMesh).write(*road.mesh);
    scene.add(road.spline);

    auto ball = makeFallingSphere(scene, Vector3(4.f, 3.f, 0.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 2);

    checkRestsOnRoad(session, *ball);
}

TEST_CASE("physics authored on the spline itself collides as the road", "[editor][physx]") {

    // Where the user actually put it: the spline IS the road as far as they are
    // concerned. A Group has no geometry, so this used to fall through to the
    // unit-box placeholder — a phantom 1 m cube at the spline's origin, which
    // is exactly the "small collider somewhere in the centre" that got reported.
    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeStraightRoad(scene);
    staticConfig(PhysicsConfig::Shape::Auto).write(*road.spline);
    scene.add(road.spline);

    auto ball = makeFallingSphere(scene, Vector3(4.f, 3.f, 0.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    // The spline counts as one body however many shapes its subtree came to.
    CHECK(session.bodyCount() == 2);

    checkRestsOnRoad(session, *ball);
}

TEST_CASE("a road collides through a corner", "[editor][physx]") {

    // The whole point of a wedge chain rather than one hull per road: adjacent
    // spans share their joint cross-section exactly, so a corner is covered at
    // any angle with neither a gap to fall through nor an overlap to fight.
    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeRoad(scene, {Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 10)}, 4.f);
    staticConfig(PhysicsConfig::Shape::Auto).write(*road.mesh);
    scene.add(road.spline);

    // Straight down onto the bend.
    auto ball = makeFallingSphere(scene, Vector3(0.f, 3.f, 0.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 2);
    // One convex per station interval of the surface itself, and not one more:
    // the collider IS the drawing, span for span.
    CHECK(staticShapeCount(session) == intervalsOf(*road.mesh));
    CHECK(staticShapeCount(session) > 4);

    checkRestsOnRoad(session, *ball);
}

TEST_CASE("a road wider than its own bends still holds a body up", "[editor][physx]") {

    // The report this exists for: an S of two opposite bends, both tighter than
    // the half-width, at which the ribbon this replaced folded — and a collider
    // read off those folded triangles was a moiré of slivers a ball fell
    // through or bounced off. The primitives behind the surface have no such
    // state: the tight bends are pie sectors, and the wedges tiling them are
    // convex whatever the radius.
    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeRoad(scene,
                         {Vector3(-8, 0, 0), Vector3(-4, 0, 0), Vector3(-2, 0, 3),
                          Vector3(2, 0, -3), Vector3(4, 0, 0), Vector3(8, 0, 0)},
                         6.f);
    staticConfig(PhysicsConfig::Shape::Auto).write(*road.mesh);
    scene.add(road.spline);

    // One ball over the middle of the S, where the two bends meet, and one over
    // the straight run leading into it.
    auto onBend = makeFallingSphere(scene, Vector3(0.f, 3.f, 0.f));
    auto onStraight = makeFallingSphere(scene, Vector3(-6.f, 3.f, 0.f));
    onStraight->name = "Ball2";
    scene.add(onBend);
    scene.add(onStraight);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 3);
    // Both bends, one hull per span, and every one of them convex — a wedge
    // through a bend cannot be anything else once no bend is tighter than the
    // half-width.
    CHECK(staticShapeCount(session) == intervalsOf(*road.mesh));

    checkRestsOnRoad(session, *onBend);
    checkRestsOnRoad(session, *onStraight);
}

TEST_CASE("a kinematic road is a road that drives", "[editor][physx]") {

    // Why the collider is convex and not a triangle mesh. A triangle mesh comes
    // back from PhysX as a PxRigidStatic, which is not something code can move;
    // the hull chain goes on a kinematic PxRigidDynamic, so the same authoring
    // that makes a road makes a conveyor belt. A body resting on it comes along
    // for the ride.
    SceneDocument document;
    auto& scene = document.scene();

    auto road = makeStraightRoad(scene);
    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Kinematic;
    config.shape = PhysicsConfig::Shape::Auto;
    config.friction = 0.9f;
    config.write(*road.mesh);
    scene.add(road.spline);

    auto ball = makeFallingSphere(scene, Vector3(0.f, 1.f, 0.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.world() != nullptr);

    using namespace ::physx;
    auto& pxScene = session.world()->scene();
    const PxU32 count = pxScene.getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    std::vector<PxActor*> actors(count);
    if (count) pxScene.getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors.data(), count);

    PxRigidDynamic* driven = nullptr;
    for (auto* actor : actors) {
        auto* body = static_cast<PxRigidDynamic*>(actor);
        if (body->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC) driven = body;
    }
    REQUIRE(driven != nullptr);
    CHECK(static_cast<int>(driven->getNbShapes()) == intervalsOf(*road.mesh));

    run(session, 60);// the ball settles on it
    const float restingHeight = ball->position.y;
    CHECK(restingHeight > 0.f);

    // Drive it. A kinematic target is the whole point: the road moves, the
    // object bound to it moves, and what is standing on it is carried. Driven
    // ALONG its own length, so what is measured is the carry rather than the
    // surface being slid out from under the ball sideways.
    const float startX = ball->position.x;
    for (int i = 0; i < 120; ++i) {
        const auto pose = driven->getGlobalPose();
        driven->setKinematicTarget(PxTransform(
                PxVec3(pose.p.x + 2.f / 60.f, pose.p.y, pose.p.z), pose.q));
        session.update(1.f / 60.f);
    }

    // The road itself went where it was driven...
    CHECK(road.mesh->position.x > 3.f);
    // ...and took the ball with it, still on top.
    CHECK(ball->position.x > startX + 1.f);
    CHECK_THAT(ball->position.y, Catch::Matchers::WithinAbs(restingHeight, 0.1f));
}

TEST_CASE("a static mesh that is no primitive collides as its triangles", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    // A 12 m plate at y = 0 with one 3 m spike off to the side. Auto used to
    // mean "the AABB", and this geometry's AABB has its top face at the spike's
    // tip: a ball dropped over the plate came to rest three metres in the air.
    const std::vector<float> positions{
            -6.f, 0.f, -6.f, 6.f, 0.f, -6.f, 6.f, 0.f, 6.f, -6.f, 0.f, 6.f,
            -5.5f, 0.f, -0.5f, -4.5f, 0.f, -0.5f, -5.f, 3.f, 0.f};
    const std::vector<unsigned int> indices{0, 2, 1, 0, 3, 2, 4, 5, 6};

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry->setIndex(indices);

    auto plate = Mesh::create(geometry);
    plate->name = "Plate";
    staticConfig(PhysicsConfig::Shape::Auto).write(*plate);
    scene.add(plate);

    auto ball = makeFallingSphere(scene, Vector3(2.f, 3.f, 2.f));
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 2);

    run(session, 240);

    using Catch::Matchers::WithinAbs;
    CHECK(ball->position.y > 0.f);           // not through the plate
    CHECK(ball->position.y < 1.f);           // and not floating on the AABB slab
    CHECK_THAT(ball->position.y, WithinAbs(0.5f, 0.06f));
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
