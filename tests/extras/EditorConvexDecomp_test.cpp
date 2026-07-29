// Convex decomposition and compound colliders: the proof that an imported model
// no longer collides as a 1 m unit box, and that a concave shape collides like
// itself. These are behavioural tests — a ball dropped into a decomposed tray
// comes to rest INSIDE it (a single hull would roof it over), and a ball dropped
// between two boxes welded into one compound actor falls through the gap.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/physx/ConvexDecomposition.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iostream>
#include <memory>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // An open-top tray (a bin): a floor and four walls, no lid. The interior is a
    // genuine concavity — a single convex hull of these vertices is a solid block
    // level with the wall tops, so a ball on that hull rests at the RIM; only a
    // decomposition (floor + walls) lets it settle down INSIDE. Built by hand as
    // an indexed box-with-a-cavity so V-HACD has real triangles to voxelize.
    //
    // Outer half-extent `o`, wall thickness `t`, so the inner cavity is
    // 2*(o - t) wide and reaches from the floor up to +o.
    std::shared_ptr<BufferGeometry> makeTray(float o = 1.0f, float t = 0.2f) {

        const float in = o - t;      // inner wall half-extent
        const float floorTop = -o + t;// top surface of the floor slab

        std::vector<float> pos;
        std::vector<unsigned int> idx;

        // Append one axis-aligned box [min,max] as 12 triangles.
        const auto addBox = [&](const Vector3& mn, const Vector3& mx) {
            const unsigned base = static_cast<unsigned>(pos.size() / 3);
            const float xs[2] = {mn.x, mx.x};
            const float ys[2] = {mn.y, mx.y};
            const float zs[2] = {mn.z, mx.z};
            for (int i = 0; i < 8; ++i) {
                pos.push_back(xs[(i >> 0) & 1]);
                pos.push_back(ys[(i >> 1) & 1]);
                pos.push_back(zs[(i >> 2) & 1]);
            }
            // 6 quads (12 tris). Vertex index bit layout: x=bit0, y=bit1, z=bit2.
            const unsigned faces[6][4] = {
                    {0, 2, 6, 4}, {1, 5, 7, 3},// -x, +x
                    {0, 4, 5, 1}, {2, 3, 7, 6},// -y, +y
                    {0, 1, 3, 2}, {4, 6, 7, 5},// -z, +z
            };
            for (auto& f : faces) {
                idx.push_back(base + f[0]);
                idx.push_back(base + f[1]);
                idx.push_back(base + f[2]);
                idx.push_back(base + f[0]);
                idx.push_back(base + f[2]);
                idx.push_back(base + f[3]);
            }
        };

        addBox({-o, -o, -o}, {o, floorTop, o});     // floor slab
        addBox({-o, floorTop, -o}, {-in, o, o});     // -x wall
        addBox({in, floorTop, -o}, {o, o, o});       // +x wall
        addBox({-in, floorTop, -o}, {in, o, -in});   // -z wall
        addBox({-in, floorTop, in}, {in, o, o});     // +z wall

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geometry->setIndex(std::move(idx));
        return geometry;
    }

    void run(PhysicsPlaySession& session, int steps) {
        for (int i = 0; i < steps; ++i) session.update(1.f / 60.f);
    }

    // A static ground box big enough to catch anything that falls through.
    std::shared_ptr<Mesh> addGround(Scene& scene) {
        auto ground = Mesh::create(BoxGeometry::create(40.f, 0.4f, 40.f), MeshBasicMaterial::create());
        ground->name = "Ground";
        ground->position.set(0.f, -0.2f, 0.f);
        PhysicsConfig cfg;
        cfg.enabled = true;
        cfg.body = PhysicsConfig::Body::Static;
        cfg.shape = PhysicsConfig::Shape::Box;
        cfg.write(*ground);
        scene.add(ground);
        return ground;
    }

}// namespace


TEST_CASE("V-HACD splits a concave tray into several hulls", "[editor][physx][vhacd]") {

    auto tray = makeTray();
    const auto* pos = tray->getAttribute<float>("position");
    const auto* index = tray->getIndex();
    REQUIRE(pos);
    REQUIRE(index);

    std::vector<std::uint32_t> indices(index->array().begin(), index->array().end());

    ConvexDecompositionParams params;// defaults: 16 hulls, 64 verts, 100k voxels
    const auto hulls = decomposeConvex(pos->array().data(), pos->count(),
                                       indices.data(), indices.size(), params);

    // The floor and the walls cannot be one convex piece: a decomposition of a
    // real concavity is two or more hulls. (The single-hull fallback lives above
    // this call, in the play session, not in decomposeConvex itself.)
    INFO("hull count " << hulls.size());
    CHECK(hulls.size() >= 2);
    for (const auto& h : hulls) CHECK(h.size() >= 12);// >= 4 points each
}

TEST_CASE("a ball settles inside a decomposed tray, not on its rim", "[editor][physx][vhacd]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // The tray: a static concave collider, decomposed with Convex Pieces. Its
    // rim is at y = +1 (top half-extent), its floor surface at y = -0.8.
    auto tray = Mesh::create(makeTray(), MeshBasicMaterial::create());
    tray->name = "Tray";
    tray->position.set(0.f, 1.f, 0.f);// floor surface now at world y = 0.2
    PhysicsConfig trayCfg;
    trayCfg.enabled = true;
    trayCfg.body = PhysicsConfig::Body::Static;
    trayCfg.shape = PhysicsConfig::Shape::Pieces;
    trayCfg.write(*tray);
    scene.add(tray);

    // A small dynamic ball dropped straight into the cavity from above the rim.
    auto ball = Mesh::create(BoxGeometry::create(0.3f, 0.3f, 0.3f), MeshBasicMaterial::create());
    ball->name = "Ball";
    ball->position.set(0.f, 4.f, 0.f);
    PhysicsConfig ballCfg;
    ballCfg.enabled = true;
    ballCfg.body = PhysicsConfig::Body::Dynamic;
    ballCfg.shape = PhysicsConfig::Shape::Box;
    ballCfg.mass = 1.f;
    ballCfg.restitution = 0.f;
    ballCfg.write(*ball);
    scene.add(ball);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 3);// ground, tray, ball

    run(session, 240);// 4 s

    // The rim is at world y = 1 + 1 = 2. If the tray collided as a single hull
    // (a solid block), the ball would rest ON it, near y = 2. Decomposed, the
    // ball drops THROUGH the open top and settles on the inner floor at world
    // y ~ 0.2 + half the ball (0.15) = ~0.35. Well below the rim is the proof.
    INFO("ball rest height " << ball->position.y);
    CHECK(ball->position.y < 1.2f);       // clearly below the rim
    CHECK(ball->position.y > 0.f);        // and not through the floor
    // And it stayed inside the walls rather than skittering out.
    CHECK(std::abs(ball->position.x) < 0.8f);
    CHECK(std::abs(ball->position.z) < 0.8f);

    session.stop();
}

TEST_CASE("a Group of two boxes becomes one compound with a gap", "[editor][physx][vhacd]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // One root Group, two box sub-meshes separated by a gap along X. Authoring
    // physics on the GROUP (which has no geometry of its own) is exactly the
    // imported-model case: Auto on a moving Group must build a compound of the
    // sub-meshes, not a 1 m unit box.
    auto model = Group::create();
    model->name = "Model";
    model->position.set(0.f, 3.f, 0.f);

    auto left = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), MeshBasicMaterial::create());
    left->position.set(-1.2f, 0.f, 0.f);
    auto right = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), MeshBasicMaterial::create());
    right->position.set(1.2f, 0.f, 0.f);
    model->add(left);
    model->add(right);

    PhysicsConfig cfg;
    cfg.enabled = true;
    cfg.body = PhysicsConfig::Body::Dynamic;
    cfg.shape = PhysicsConfig::Shape::Auto;// resolves to a per-sub-mesh compound
    cfg.mass = 2.f;
    cfg.restitution = 0.f;
    cfg.write(*model);
    scene.add(model);

    // A probe ball dropped through the GAP between the two boxes (x = 0). If the
    // Group collided as a unit box it would sit ON the probe's path; a correct
    // compound leaves the centre open, so the probe falls to the ground.
    auto probe = Mesh::create(BoxGeometry::create(0.3f, 0.3f, 0.3f), MeshBasicMaterial::create());
    probe->name = "Probe";
    probe->position.set(0.f, 6.f, 0.f);
    PhysicsConfig probeCfg;
    probeCfg.enabled = true;
    probeCfg.body = PhysicsConfig::Body::Dynamic;
    probeCfg.shape = PhysicsConfig::Shape::Box;
    probeCfg.mass = 1.f;
    probeCfg.restitution = 0.f;
    probeCfg.write(*probe);
    scene.add(probe);

    PhysicsPlaySession session;
    session.start(scene);
    CHECK(session.bodyCount() == 3);// ground, model, probe

    run(session, 300);// 5 s

    // The model fell and landed on the ground on its two boxes (bottom of each
    // box at y ~ 0.2). A unit-box fallback would have landed at y ~ 0.7 (half a
    // metre), so a low resting model is itself evidence the compound is real.
    INFO("model rest height " << model->position.y);
    CHECK(model->position.y < 0.6f);

    // The probe fell through the gap to the ground — it did not land on the
    // model. Ground top is at y = 0, probe half is 0.15, so it rests near 0.15.
    INFO("probe rest height " << probe->position.y);
    CHECK(probe->position.y < 0.4f);

    session.stop();
}

TEST_CASE("duplicate geometries decompose once", "[editor][physx][vhacd]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // Two meshes SHARING one BufferGeometry (same uuid) — what a clone/duplicate
    // does. Both authored with Convex Pieces. The decomposition is keyed on the
    // geometry uuid, so the second must be a cache hit: one cook, not two.
    auto shared = makeTray();

    for (int i = 0; i < 2; ++i) {
        auto mesh = Mesh::create(shared, MeshBasicMaterial::create());
        mesh->name = "Tray" + std::to_string(i);
        mesh->position.set(static_cast<float>(i) * 4.f, 2.f, 0.f);
        PhysicsConfig cfg;
        cfg.enabled = true;
        cfg.body = PhysicsConfig::Body::Dynamic;
        cfg.shape = PhysicsConfig::Shape::Pieces;
        cfg.mass = 1.f;
        cfg.write(*mesh);
        scene.add(mesh);
    }

    PhysicsPlaySession session;
    session.start(scene);

    CHECK(session.bodyCount() == 3);// ground + two trays
    // The whole point: two bodies, one decomposition.
    INFO("cook count " << session.decompositionCookCount());
    CHECK(session.decompositionCookCount() == 1);

    session.stop();
}

TEST_CASE("convex-pieces config round-trips through userData", "[editor][physx]") {

    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Dynamic;
    config.shape = PhysicsConfig::Shape::Pieces;
    config.hulls = 24;
    config.hullVerts = 48;
    config.voxels = 250000;

    const auto decoded = PhysicsConfig::decode(config.encode());
    REQUIRE(decoded);
    CHECK(decoded->shape == PhysicsConfig::Shape::Pieces);
    CHECK(decoded->hulls == 24);
    CHECK(decoded->hullVerts == 48);
    CHECK(decoded->voxels == 250000);
    CHECK(decoded->encode() == config.encode());

    // A document written before these keys existed keeps the defaults.
    const auto legacy = PhysicsConfig::decode("body=dynamic;shape=convex;mass=2;friction=0.5;restitution=0.1");
    REQUIRE(legacy);
    CHECK(legacy->hulls == PhysicsConfig{}.hulls);
    CHECK(legacy->hullVerts == PhysicsConfig{}.hullVerts);
    CHECK(legacy->voxels == PhysicsConfig{}.voxels);
}
