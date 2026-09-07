// PhysxWorld's Mesh-based add* overloads: pose from the FULL ancestor chain,
// bounds from the WORLD scale. Regression tests for the parented-mesh bugs
// where a stale parent matrixWorld placed the body at local coordinates, and
// where add()/addStatic() dropped the decomposed scale from inferred
// primitives entirely.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

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
