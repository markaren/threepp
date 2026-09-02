// PhysxWorld's Mesh-based add* overloads: pose from the FULL ancestor chain,
// bounds from the WORLD scale. Regression tests for the parented-mesh bugs
// where a stale parent matrixWorld placed the body at local coordinates, and
// where add()/addStatic() dropped the decomposed scale from inferred
// primitives entirely.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    // The bug's natural habitat: build the graph, add the body, never render.
    // No scene.updateMatrixWorld() anywhere — the add* overloads must refresh
    // the ancestor chain themselves.
    struct ParentedMesh {
        Scene scene;
        std::shared_ptr<Group> parent;
        std::shared_ptr<Mesh> mesh;
    };

    ParentedMesh makeParented(const std::shared_ptr<BufferGeometry>& geometry,
                              const Vector3& parentScale = Vector3(1.f, 1.f, 1.f)) {

        ParentedMesh out;
        out.parent = Group::create();
        out.parent->position.set(5.f, 0.f, 3.f);
        out.parent->scale.copy(parentScale);
        out.scene.add(out.parent);

        out.mesh = Mesh::create(geometry, MeshBasicMaterial::create());
        out.mesh->position.set(0.f, 4.f, 0.f);
        out.parent->add(out.mesh);
        return out;
    }

    Vector3 sizeOf(const ::physx::PxBounds3& b) {
        return {b.maximum.x - b.minimum.x, b.maximum.y - b.minimum.y, b.maximum.z - b.minimum.z};
    }

}// namespace


TEST_CASE("add() places the body from the full ancestor chain", "[physx]") {

    auto s = makeParented(BoxGeometry::create(1.f, 1.f, 1.f));

    PhysxWorld world;
    auto* body = world.add(*s.mesh, 1.f);
    REQUIRE(body);

    const auto pose = body->getGlobalPose();
    CHECK_THAT(pose.p.x, WithinAbs(5.f, 1e-3));
    CHECK_THAT(pose.p.y, WithinAbs(4.f, 1e-3));
    CHECK_THAT(pose.p.z, WithinAbs(3.f, 1e-3));
}

TEST_CASE("add() applies the world scale to an inferred primitive", "[physx]") {

    auto s = makeParented(BoxGeometry::create(1.f, 1.f, 1.f), Vector3(2.f, 2.f, 2.f));

    PhysxWorld world;
    auto* body = world.add(*s.mesh, 1.f);
    REQUIRE(body);

    // getWorldBounds inflates by 1.01 by default; 0.1 absolute tolerance
    // comfortably covers that while still catching an unscaled 1 m box.
    const auto size = sizeOf(body->getWorldBounds());
    CHECK_THAT(size.x, WithinAbs(2.f, 0.1f));
    CHECK_THAT(size.y, WithinAbs(2.f, 0.1f));
    CHECK_THAT(size.z, WithinAbs(2.f, 0.1f));
}

TEST_CASE("addDynamicConvex under a transformed parent", "[physx]") {

    // Not a Box/Sphere/Capsule: the convex-hull path a complex mesh takes.
    auto s = makeParented(CylinderGeometry::create(0.5f, 0.5f, 1.f), Vector3(2.f, 2.f, 2.f));

    PhysxWorld world;
    auto* body = world.addDynamicConvex(*s.mesh, 1000.f);
    REQUIRE(body);

    // The 2x parent scale also scales the child's local (0,4,0) offset.
    const auto pose = body->getGlobalPose();
    CHECK_THAT(pose.p.x, WithinAbs(5.f, 1e-3));
    CHECK_THAT(pose.p.y, WithinAbs(8.f, 1e-3));
    CHECK_THAT(pose.p.z, WithinAbs(3.f, 1e-3));

    const auto size = sizeOf(body->getWorldBounds());
    CHECK_THAT(size.y, WithinAbs(2.f, 0.1f));
}

TEST_CASE("addStaticTrimesh under a transformed parent", "[physx]") {

    auto s = makeParented(BoxGeometry::create(1.f, 1.f, 1.f), Vector3(2.f, 2.f, 2.f));

    PhysxWorld world;
    auto* body = world.addStaticTrimesh(*s.mesh);
    REQUIRE(body);

    // The 2x parent scale also scales the child's local (0,4,0) offset.
    const auto pose = body->getGlobalPose();
    CHECK_THAT(pose.p.x, WithinAbs(5.f, 1e-3));
    CHECK_THAT(pose.p.y, WithinAbs(8.f, 1e-3));
    CHECK_THAT(pose.p.z, WithinAbs(3.f, 1e-3));

    const auto size = sizeOf(body->getWorldBounds());
    CHECK_THAT(size.x, WithinAbs(2.f, 0.1f));
}

TEST_CASE("add(InstancedMesh) carries per-instance scale into the collider", "[physx]") {

    auto mesh = InstancedMesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                                      MeshBasicMaterial::create(), 2);
    Matrix4 m;
    m.compose(Vector3(0.f, 1.f, 0.f), Quaternion(), Vector3(1.f, 1.f, 1.f));
    mesh->setMatrixAt(0, m);
    m.compose(Vector3(4.f, 1.f, 0.f), Quaternion(), Vector3(3.f, 3.f, 3.f));
    mesh->setMatrixAt(1, m);

    PhysxWorld world;
    const auto actors = world.add(*mesh, 1.f);
    REQUIRE(actors.size() == 2);

    CHECK_THAT(sizeOf(actors[0]->getWorldBounds()).x, WithinAbs(1.f, 0.1f));
    CHECK_THAT(sizeOf(actors[1]->getWorldBounds()).x, WithinAbs(3.f, 0.1f));
}


// -- addStaticHeightField ----------------------------------------------------
//
// The height field is a Z-UP contract laid over a shape PhysX defines Y-up, with
// a handedness flip in the middle of it (see PhysxWorld.hpp). A comment cannot
// prove that mapping; an ASYMMETRIC analytic surface can. These tests build
//
//     z = 0.1*x + 0.3*sin(y)   over x in [-2, 2], y in [-1, 1] at 10 cm
//
// which is monotone in x, odd in y, and equal at no two probe points, so a
// transposed grid, a mirrored row order or a swapped axis each move at least one
// probe by far more than the tolerance.

namespace {

    constexpr int kHfNx = 41;// x samples: -2.0 .. 2.0
    constexpr int kHfNy = 21;// y samples: -1.0 .. 1.0
    constexpr float kHfCell = 0.1f;
    constexpr float kHfX0 = -2.f;
    constexpr float kHfY0 = -1.f;

    float rampZ(float x, float y) {
        return 0.1f * x + 0.3f * std::sin(y);
    }

    std::vector<float> rampField() {
        std::vector<float> h(static_cast<std::size_t>(kHfNx) * kHfNy);
        for (int iy = 0; iy < kHfNy; ++iy) {
            for (int ix = 0; ix < kHfNx; ++ix) {
                h[static_cast<std::size_t>(iy) * kHfNx + ix] =
                        rampZ(kHfX0 + ix * kHfCell, kHfY0 + iy * kHfCell);
            }
        }
        return h;
    }

    // The same surface as a regular-grid triangle soup: the Phase 1 collider, and
    // the control the height field is measured against.
    std::shared_ptr<BufferGeometry> rampTrimesh(const std::vector<float>& h) {
        std::vector<float> pos;
        pos.reserve(static_cast<std::size_t>(kHfNx - 1) * (kHfNy - 1) * 18);
        auto push = [&](int ix, int iy) {
            pos.push_back(kHfX0 + ix * kHfCell);
            pos.push_back(kHfY0 + iy * kHfCell);
            pos.push_back(h[static_cast<std::size_t>(iy) * kHfNx + ix]);
        };
        for (int iy = 0; iy + 1 < kHfNy; ++iy) {
            for (int ix = 0; ix + 1 < kHfNx; ++ix) {
                push(ix, iy);
                push(ix + 1, iy);
                push(ix + 1, iy + 1);
                push(ix, iy);
                push(ix + 1, iy + 1);
                push(ix, iy + 1);
            }
        }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        return g;
    }

    struct Probe {
        float x, y;
        const char* what;
    };

    // Four corners (a flipped axis moves those the most), the centre, and seven
    // asymmetric points, most of them off the sample lattice.
    const Probe kProbes[] = {
            {-2.00f, -1.00f, "corner -x -y"},
            {2.00f, -1.00f, "corner +x -y"},
            {-2.00f, 1.00f, "corner -x +y"},
            {2.00f, 1.00f, "corner +x +y"},
            {0.00f, 0.00f, "centre"},
            {1.50f, -0.70f, "asym +x -y"},
            {-1.50f, 0.70f, "asym -x +y"},
            {0.35f, -0.45f, "off-lattice"},
            {-0.85f, 0.25f, "off-lattice"},
            {1.15f, 0.95f, "near +y edge"},
            {-1.95f, -0.05f, "near -x edge"},
            {1.85f, 0.85f, "asym near corner"},
    };

    // A plumb bob: a small sphere with x/y translation and all rotation locked, so
    // it measures the surface height under a known (x, y) instead of rolling off a
    // 17-degree slope (a free ball on this ramp rolls to the -y edge).
    ::physx::PxRigidDynamic* dropProbe(PhysxWorld& w, float x, float y, float z,
                                       float radius, float vz, ::physx::PxMaterial* mat) {
        using namespace ::physx;
        auto* body = w.addDynamic(PxSphereGeometry(radius), PxTransform(PxVec3(x, y, z)), 500.f, mat);
        body->setRigidDynamicLockFlags(
                PxRigidDynamicLockFlag::eLOCK_LINEAR_X | PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_X | PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
        body->setLinearVelocity(PxVec3(0.f, 0.f, vz));
        return body;
    }

    PhysxWorld::Settings zUpSettings() {
        PhysxWorld::Settings s;
        s.gravity = Vector3(0.f, 0.f, -9.81f);// the Spot / calico convention
        return s;
    }

}// namespace


TEST_CASE("addStaticHeightField puts H[iy][ix] at world (x0+ix*c, y0+iy*c)", "[physx]") {

    const auto field = rampField();

    PhysxWorld world(zUpSettings());
    auto* mat = world.physics().createMaterial(0.8f, 0.8f, 0.f);

    auto* body = world.addStaticHeightField(field, kHfNx, kHfNy, kHfCell,
                                            Vector3(kHfX0, kHfY0, 0.f), 0.5f);
    REQUIRE(body);

    // The shape's bounds are the first, cheapest witness that nothing is transposed:
    // 4 m of x, 2 m of y. A transpose swaps them.
    const auto size = sizeOf(body->getWorldBounds());
    CHECK_THAT(size.x, WithinAbs(4.f, 0.05f));
    CHECK_THAT(size.y, WithinAbs(2.f, 0.05f));

    constexpr float kR = 0.05f;
    std::vector<::physx::PxRigidDynamic*> probes;
    for (const auto& p : kProbes) {
        probes.push_back(dropProbe(world, p.x, p.y, rampZ(p.x, p.y) + 0.5f, kR, 0.f, mat));
    }

    for (int i = 0; i < 240; ++i) world.step(1.f / 60.f);

    std::printf("\n  height-field probes (rest z - radius vs analytic 0.1x + 0.3 sin y)\n");
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const auto& p = kProbes[i];
        const float want = rampZ(p.x, p.y);
        const float got = probes[i]->getGlobalPose().p.z - kR;
        std::printf("   (%+5.2f,%+5.2f) %-18s want %+7.4f  got %+7.4f  err %+7.4f\n",
                    p.x, p.y, p.what, want, got, got - want);
        INFO("probe " << p.what << " at (" << p.x << ", " << p.y << ")");
        CHECK_THAT(got, WithinAbs(want, 0.02f));
    }
}

TEST_CASE("addStaticHeightField at 6 m/s: the limit is the substep, not the shape", "[physx]") {

    // The premise this test was written to check -- "a height field is solid below
    // the surface, so a fast foot cannot tunnel" -- is a PhysX 3 fact. PhysX 5.5
    // has no PxHeightFieldDesc::thickness at all, and the numbers below say the
    // height field is a SURFACE: at 6 m/s a small probe goes through it exactly as
    // it goes through the same surface cooked as triangles. What survives is the
    // ordinary discrete-collision rule -- nothing tunnels while the probe's
    // diameter exceeds its per-substep travel -- and that is what is asserted.

    const auto field = rampField();

    // 12 probes thrown DOWN at 6 m/s from 0.5 m up: impact ~6.8 m/s, 0.113 m of
    // travel per 1/60 s substep, no CCD anywhere in PhysxWorld.
    auto run = [&](bool heightField, float radius, std::vector<float>& out) {
        PhysxWorld world(zUpSettings());
        auto* mat = world.physics().createMaterial(0.8f, 0.8f, 0.f);
        std::shared_ptr<BufferGeometry> geom;// kept alive across the cook
        if (heightField) {
            REQUIRE(world.addStaticHeightField(field, kHfNx, kHfNy, kHfCell,
                                               Vector3(kHfX0, kHfY0, 0.f), 0.5f));
        } else {
            geom = rampTrimesh(field);
            REQUIRE(world.addStaticTrimesh(*geom));
        }
        std::vector<::physx::PxRigidDynamic*> probes;
        for (const auto& p : kProbes) {
            probes.push_back(dropProbe(world, p.x, p.y, rampZ(p.x, p.y) + 0.5f, radius, -6.f, mat));
        }
        for (int i = 0; i < 240; ++i) world.step(1.f / 60.f);
        out.clear();
        for (auto* b : probes) out.push_back(b->getGlobalPose().p.z - radius);
    };

    auto through = [&](const std::vector<float>& rest) {
        int n = 0;
        for (std::size_t i = 0; i < rest.size(); ++i) {
            if (rest[i] < rampZ(kProbes[i].x, kProbes[i].y) - 0.02f) ++n;
        }
        return n;
    };

    // -- a 16 cm ball: diameter 0.16 m > 0.113 m of substep travel. Must hold.
    constexpr float kBig = 0.08f;
    std::vector<float> hfBig;
    run(true, kBig, hfBig);
    std::printf("\n  6 m/s onto the height field, 16 cm ball (diameter > substep travel)\n");
    for (std::size_t i = 0; i < hfBig.size(); ++i) {
        const auto& p = kProbes[i];
        const float want = rampZ(p.x, p.y);
        std::printf("   (%+5.2f,%+5.2f) %-18s want %+7.4f  got %+7.4f  err %+7.4f\n",
                    p.x, p.y, p.what, want, hfBig[i], hfBig[i] - want);
        INFO("probe " << p.what);
        CHECK_THAT(hfBig[i], WithinAbs(want, 0.02f));
    }
    CHECK(through(hfBig) == 0);

    // -- a 4 cm ball: 5x its own diameter per substep. Height field vs the same
    // surface as a trimesh, which is the control.
    constexpr float kSmall = 0.02f;
    std::vector<float> hfSmall, triSmall;
    run(true, kSmall, hfSmall);
    run(false, kSmall, triSmall);

    std::printf("\n  6 m/s, 4 cm ball: height field vs the same surface as a trimesh\n");
    for (std::size_t i = 0; i < hfSmall.size(); ++i) {
        const auto& p = kProbes[i];
        std::printf("   (%+5.2f,%+5.2f) want %+7.4f   hf %+9.4f   trimesh %+9.4f\n",
                    p.x, p.y, rampZ(p.x, p.y), hfSmall[i], triSmall[i]);
    }
    std::printf("   through: height field %d / %d, trimesh %d / %d\n",
                through(hfSmall), static_cast<int>(hfSmall.size()),
                through(triSmall), static_cast<int>(triSmall.size()));

    // The contract is comparative, and it is the honest one: whatever a sheet of
    // triangles catches at this speed, the height field catches too. If PhysX ever
    // regrows a solid volume under a height field, this goes green the other way
    // and the message above is what needs rewriting.
    CHECK(through(hfSmall) <= through(triSmall));
}

TEST_CASE("addStaticHeightField rejects a degenerate field", "[physx]") {

    PhysxWorld world(zUpSettings());
    const std::vector<float> flat(4, 0.f);

    CHECK(world.addStaticHeightField(flat, 1, 4, 0.1f, Vector3(), 0.5f) == nullptr);
    CHECK(world.addStaticHeightField(flat, 2, 2, 0.f, Vector3(), 0.5f) == nullptr);
    CHECK(world.addStaticHeightField(flat, 2, 2, 0.1f, Vector3(), 0.f) == nullptr);
    CHECK_THROWS(world.addStaticHeightField(flat, 3, 3, 0.1f, Vector3(), 0.5f));

    // A perfectly flat field still has to cook: its height range is zero, and the
    // scale falls back to its floor rather than dividing by it.
    CHECK(world.addStaticHeightField(flat, 2, 2, 0.1f, Vector3(), 0.5f) != nullptr);
}
