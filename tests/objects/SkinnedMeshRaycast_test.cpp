// Picking a skinned mesh.
//
// glTF says a skinned mesh's own node transform MUST be ignored — the joints
// place every vertex — and threepp honours that: SkinnedMesh keeps
// bindMatrixInverse equal to the inverse of its world matrix, so the two
// cancel and the vertices land wherever the bones put them.
//
// That makes the mesh's GEOMETRY bounds a description of its BIND pose, in a
// frame nothing is drawn in. A raycaster that rejects on those bounds pushed
// through matrixWorld rejects on a phantom: on a Mixamo-style rig, whose
// armature node carries the 0.01 unit scale, the reject volume is a
// centimetre-wide bubble near the origin and every click on the character
// misses it. The rig below reproduces exactly that arrangement.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/Raycaster.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    struct Rig {
        std::shared_ptr<Group> root;
        std::shared_ptr<SkinnedMesh> mesh;
    };

    // A one-bone "character": a 1 m cube standing with its centre 1 m up,
    // under an armature node scaled by `armatureScale`.
    //
    // The skin is bound with an IDENTITY bindMatrix — what threepp's
    // GLTFLoader does — and the bone's inverse-bind is the inverse of its bind
    // world matrix, so at rest `boneWorld * boneInverse` is the identity and
    // every vertex renders at its raw geometry position. The armature's scale
    // therefore changes NOTHING about where the cube is drawn, which is the
    // whole point: it only pollutes matrixWorld.
    Rig makeRig(float armatureScale) {

        auto root = Group::create();
        root->name = "Armature";
        root->scale.set(armatureScale, armatureScale, armatureScale);

        auto bone = Bone::create();
        bone->name = "root";
        root->add(bone);

        auto mesh = SkinnedMesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                                        MeshBasicMaterial::create());
        // Standing on the ground with its centre at y = 1, like a body.
        mesh->geometry()->translate(0.f, 1.f, 0.f);
        root->add(mesh);

        root->updateMatrixWorld(true);

        // Every vertex fully weighted to the single bone.
        const auto count = mesh->geometry()->getAttribute<float>("position")->count();
        std::vector<float> indices(count * 4, 0.f);
        std::vector<float> weights(count * 4, 0.f);
        for (unsigned i = 0; i < count; ++i) weights[i * 4] = 1.f;
        mesh->geometry()->setAttribute("skinIndex",
                                       FloatBufferAttribute::create(std::move(indices), 4));
        mesh->geometry()->setAttribute("skinWeight",
                                       FloatBufferAttribute::create(std::move(weights), 4));

        Matrix4 boneInverse(*bone->matrixWorld);
        boneInverse.invert();
        auto skeleton = Skeleton::create({bone}, {boneInverse});
        mesh->bind(skeleton, Matrix4());

        root->updateMatrixWorld(true);
        return {root, mesh};
    }

    // Straight at the cube's centre, from 5 m away down -Z.
    std::vector<Intersection> shootAtChest(Object3D& scene) {

        Raycaster raycaster;
        raycaster.set(Vector3(0.f, 1.f, 5.f), Vector3(0.f, 0.f, -1.f));
        return raycaster.intersectObject(scene, true);
    }

}// namespace


TEST_CASE("a skinned mesh is pickable under a unit-scaled armature", "[objects][skinning]") {

    auto rig = makeRig(1.f);
    const auto hits = shootAtChest(*rig.root);

    REQUIRE_FALSE(hits.empty());
    CHECK(hits.front().object == rig.mesh.get());
    // The near face of a 1 m cube centred at z = 0.
    CHECK_THAT(hits.front().distance, WithinAbs(4.5f, 1e-3f));
}

TEST_CASE("a skinned mesh is pickable under a 0.01 armature, like a Mixamo rig",
          "[objects][skinning]") {

    // The regression. The mesh renders in exactly the same place as the case
    // above — the skin cancels the armature transform — so it must pick the
    // same way. Rejecting on the geometry's bind-pose sphere pushed through
    // matrixWorld gives a 1 cm bubble down at the origin, and this ray, aimed
    // a metre up, misses it entirely.
    auto rig = makeRig(0.01f);
    const auto hits = shootAtChest(*rig.root);

    REQUIRE_FALSE(hits.empty());
    CHECK(hits.front().object == rig.mesh.get());
    CHECK_THAT(hits.front().distance, WithinAbs(4.5f, 1e-3f));
}

TEST_CASE("the posed bounds describe where the mesh is drawn", "[objects][skinning]") {

    auto rig = makeRig(0.01f);

    // The bounds are in the MESH'S OWN local space — the frame the raycaster
    // reduces its ray to — and that frame is not metres: the node transform
    // the skin cancels is still on matrixWorld, so a 1 m cube under a 0.01
    // armature measures 100 units here. Pushing the box back out through
    // matrixWorld is what makes it comparable to the world, and that is
    // exactly what raycast() does with the sphere.
    Box3 world(rig.mesh->posedBoundingBox());
    REQUIRE_FALSE(world.isEmpty());
    world.applyMatrix4(*rig.mesh->matrixWorld);
    CHECK_THAT(world.min().y, WithinAbs(0.5f, 1e-3f));
    CHECK_THAT(world.max().y, WithinAbs(1.5f, 1e-3f));

    // The geometry's own bounds are the same numbers in the same frame, and
    // that is the trap: they only agree at the bind pose. What made picking
    // fail was never the numbers but the FRAME — a bind-pose box pushed
    // through a 0.01 matrixWorld describes a centimetre.
    CHECK(rig.mesh->posedBoundingSphere().radius > 0.f);
}

TEST_CASE("moving the bone moves what can be picked", "[objects][skinning]") {

    // The bounds are recomputed from the POSE, so a mesh whose skeleton has
    // walked away is no longer under the old ray — and is under a new one.
    auto rig = makeRig(1.f);
    REQUIRE_FALSE(shootAtChest(*rig.root).empty());

    // The bone is in armature space; 3 m along +X moves the cube with it.
    rig.mesh->skeleton->bones.front()->position.set(3.f, 0.f, 0.f);
    rig.root->updateMatrixWorld(true);

    CHECK(shootAtChest(*rig.root).empty());

    Raycaster raycaster;
    raycaster.set(Vector3(3.f, 1.f, 5.f), Vector3(0.f, 0.f, -1.f));
    const auto moved = raycaster.intersectObject(*rig.root, true);
    REQUIRE_FALSE(moved.empty());
    CHECK(moved.front().object == rig.mesh.get());
}
