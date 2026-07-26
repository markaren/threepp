
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "../equals_util.hpp"

#include <cmath>

using namespace threepp;


TEST_CASE("applyMatrix4") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto m = Matrix4();
    auto expectedPos = Vector3(x, y, z);
    auto expectedQuat = Quaternion(0.5f * static_cast<float>(std::sqrt(2)), 0, 0, 0.5f * static_cast<float>(std::sqrt(2)));

    m.makeRotationX(math::PI / 2);
    m.setPosition(Vector3(x, y, z));

    a->applyMatrix4(m);

    REQUIRE(a->position == expectedPos);
    auto result = std::abs(a->quaternion.x - expectedQuat.x) <= eps &&
                  std::abs(a->quaternion.y - expectedQuat.y) <= eps &&
                  std::abs(a->quaternion.z - expectedQuat.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("applyQuaternion") {

    auto a = Object3D::create();
    auto sqrt = 0.5f * static_cast<float>(std::sqrt(2));
    auto quat = Quaternion(0, sqrt, 0, sqrt);
    auto expected = Quaternion(sqrt / 2, sqrt / 2, 0, 0);

    a->quaternion.set(0.25, 0.25, 0.25, 0.25);
    a->applyQuaternion(quat);

    auto result = std::abs(a->quaternion.x - expected.x) <= eps &&
                  std::abs(a->quaternion.y - expected.y) <= eps &&
                  std::abs(a->quaternion.z - expected.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("localToWorld") {

    auto v = Vector3();
    const auto expectedPosition = Vector3(5, -1, -4);

    const auto parent = Object3D::create();
    const auto child = Object3D::create();

    parent->position.set(1, 0, 0);
    parent->rotation.set(0, math::PI / 2, 0);
    parent->scale.set(2, 1, 1);

    child->position.set(0, 1, 0);
    child->rotation.set(math::PI / 2, 0, 0);
    child->scale.set(1, 2, 1);

    parent->add(child);
    parent->updateMatrixWorld();

    child->localToWorld(v.set(2, 2, 2));

    auto result = std::abs(v.x - expectedPosition.x) <= eps &&
                  std::abs(v.y - expectedPosition.y) <= eps &&
                  std::abs(v.z - expectedPosition.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("worldToLocal") {

    auto v = Vector3();
    const auto expectedPosition = Vector3(-1, 0.5, -1);

    const auto parent = Object3D::create();
    const auto child = Object3D::create();

    parent->position.set(1, 0, 0);
    parent->rotation.set(0, math::PI / 2, 0);
    parent->scale.set(2, 1, 1);

    child->position.set(0, 1, 0);
    child->rotation.set(math::PI / 2, 0, 0);
    child->scale.set(1, 2, 1);

    parent->add(child);
    parent->updateMatrixWorld();

    child->worldToLocal(v.set(2, 2, 2));

    auto result = std::abs(v.x - expectedPosition.x) <= eps &&
                  std::abs(v.y - expectedPosition.y) <= eps &&
                  std::abs(v.z - expectedPosition.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("lookAt") {

    auto obj = Object3D::create();
    obj->lookAt(Vector3(0, -1, 1));

    REQUIRE_THAT(obj->rotation.x * math::RAD2DEG, Catch::Matchers::WithinRel(45.f));
}

TEST_CASE("getWorldPosition") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto b = Object3D::create();
    auto expectedSingle = Vector3(x, y, z);
    auto expectedParent = Vector3(x, y, 0);
    auto expectedChild = Vector3(x, y, 7);
    auto position = Vector3();

    a->translateX(x);
    a->translateY(y);
    a->translateZ(z);

    a->getWorldPosition(position);
    REQUIRE(position == expectedSingle);

    // translate child and then parent
    b->translateZ(7);
    a->add(b);
    a->translateZ(-z);

    a->getWorldPosition(position);
    REQUIRE(position == expectedParent);
    b->getWorldPosition(position);
    REQUIRE(position == expectedChild);
}

TEST_CASE("getWorldScale") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto m = Matrix4().makeScale(x, y, z);
    auto expected = Vector3(x, y, z);

    a->applyMatrix4(m);

    Vector3 scale;
    a->getWorldScale(scale);
    REQUIRE(scale == expected);
}

TEST_CASE("updateMatrixWorld") {

    auto parent = Object3D::create();
    auto child = Object3D::create();

    // -- Standard usage test

    parent->position.set(1, 2, 3);
    child->position.set(4, 5, 6);
    parent->add(child);

    parent->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{
                                                1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0,
                                                1, 2, 3, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    REQUIRE(child->matrix->elements == std::array<float, 16>{
                                               1, 0, 0, 0,
                                               0, 1, 0, 0,
                                               0, 0, 1, 0,
                                               4, 5, 6, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    5, 7, 9, 1});

    REQUIRE((parent->matrixWorldNeedsUpdate || child->matrixWorldNeedsUpdate) == false);

    // -- No sync between local position/quaternion/scale/matrix and world matrix test

    parent->position.set(0, 0, 0);
    parent->updateMatrix();

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    // -- matrixAutoUpdate = false test

    // Resetting local and world matrices to the origin
    child->position.set(0, 0, 0);
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    parent->matrixAutoUpdate = false;
    child->matrixAutoUpdate = false;
    parent->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{
                                                1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0,
                                                0, 0, 0, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1});

    // -- matrixWorldAutoUpdate = false test

    parent->position.set(3, 2, 1);
    parent->updateMatrix();
    parent->matrixWorldNeedsUpdate = false;

    parent->updateMatrixWorld();

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1});

    // -- Propagation to children world matrices test

    child->position.set(0, 0, 0);
    parent->position.set(1, 2, 3);

    parent->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    1, 2, 3, 1});

    // -- force argument test

    // Resetting the local and world matrices to the origin
    child->position.set(0, 0, 0);
    child->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    parent->updateMatrix();
    parent->matrixAutoUpdate = false;
    parent->matrixWorldNeedsUpdate = false;

    parent->updateMatrixWorld(true);

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    // -- Restriction test: No effect to parent matrices

    // Resetting the local and world matrices to the origin
    parent->position.set(0, 0, 0);
    child->position.set(0, 0, 0);
    parent->matrixAutoUpdate = true;
    child->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    child->position.set(4, 5, 6);

    child->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{1, 0, 0, 0,
                                                                  0, 1, 0, 0,
                                                                  0, 0, 1, 0,
                                                                  4, 5, 6, 1});
}

TEST_CASE("external matrix writes propagate when matrixAutoUpdate is false") {

    // The helper pattern: CameraHelper and the light helpers alias another
    // object's matrixWorld as their local matrix with matrixAutoUpdate=false,
    // relying on updateMatrixWorld() (no force) to pick up external writes.
    // Before updateMatrix() gained its early-out this was carried by the
    // every-frame force cascade from the root; now it's polled.

    auto root = Object3D::create();
    auto tracker = Object3D::create();
    tracker->matrixAutoUpdate = false;
    root->add(tracker);

    root->updateMatrixWorld();// settle — nothing dirty

    REQUIRE(tracker->matrixWorld->elements == std::array<float, 16>{
                                                      1, 0, 0, 0,
                                                      0, 1, 0, 0,
                                                      0, 0, 1, 0,
                                                      0, 0, 0, 1});

    // external write to the local matrix (as if the tracked camera moved)
    tracker->matrix->setPosition(Vector3(1, 2, 3));

    root->updateMatrixWorld();// no force

    REQUIRE(tracker->matrixWorld->elements == std::array<float, 16>{
                                                      1, 0, 0, 0,
                                                      0, 1, 0, 0,
                                                      0, 0, 1, 0,
                                                      1, 2, 3, 1});

    // re-enabling matrixAutoUpdate must recompose from position/quaternion/
    // scale and clobber the external matrix, exactly like three.js
    tracker->matrixAutoUpdate = true;
    root->updateMatrixWorld();

    REQUIRE(tracker->matrix->elements == std::array<float, 16>{
                                                 1, 0, 0, 0,
                                                 0, 1, 0, 0,
                                                 0, 0, 1, 0,
                                                 0, 0, 0, 1});

    REQUIRE(tracker->matrixWorld->elements == std::array<float, 16>{
                                                      1, 0, 0, 0,
                                                      0, 1, 0, 0,
                                                      0, 0, 1, 0,
                                                      0, 0, 0, 1});
}

TEST_CASE("CameraHelper follows camera without forced world update") {

    auto scene = Object3D::create();
    auto camera = PerspectiveCamera::create(60.f, 1.f, 0.1f, 100.f);
    auto helper = CameraHelper::create(*camera);
    scene->add(helper);

    scene->updateMatrixWorld();

    camera->position.set(5, 6, 7);
    camera->updateMatrixWorld();
    scene->updateMatrixWorld();// renderer per-frame pass, no force

    REQUIRE(helper->matrixWorld->elements == camera->matrixWorld->elements);

    // move again — must keep tracking on subsequent frames too
    camera->position.set(-3, 0, 1);
    camera->updateMatrixWorld();
    scene->updateMatrixWorld();

    REQUIRE(helper->matrixWorld->elements == camera->matrixWorld->elements);
    REQUIRE(helper->matrixWorld->elements[12] == -3.f);
}

TEST_CASE("updateWorldMatrix") {

    auto object = Object3D::create();
    auto parent = Object3D::create();
    auto child = Object3D::create();

    auto m = Matrix4();
    auto v = Vector3();

    parent->add(object);
    object->add(child);

    parent->position.set(1, 2, 3);
    object->position.set(4, 5, 6);
    child->position.set(7, 8, 9);

    // Update the world matrix of an object

    object->updateWorldMatrix();

    REQUIRE(parent->matrix->elements == m.elements);

    REQUIRE(parent->matrixWorld->elements == m.elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(object->position).elements);

    REQUIRE(child->matrix->elements == m.identity().elements);

    REQUIRE(child->matrixWorld->elements == m.elements);

    // Update the world matrices of an object and its parents

    object->matrix->identity();
    object->matrixWorld->identity();

    object->updateWorldMatrix(true, false);

    REQUIRE(parent->matrix->elements == m.setPosition(parent->position).elements);

    REQUIRE(parent->matrixWorld->elements == m.setPosition(parent->position).elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position)).elements);

    REQUIRE(child->matrix->elements == m.identity().elements);

    REQUIRE(child->matrixWorld->elements == m.identity().elements);

    // Update the world matrices of an object and its children

    parent->matrix->identity();
    parent->matrixWorld->identity();
    object->matrix->identity();
    object->matrixWorld->identity();

    object->updateWorldMatrix(false, true);

    REQUIRE(parent->matrix->elements == m.elements);

    REQUIRE(parent->matrixWorld->elements == m.elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(object->position).elements);

    REQUIRE(child->matrix->elements == m.setPosition(child->position).elements);

    REQUIRE(child->matrixWorld->elements == m.setPosition(v.copy(object->position).add(child->position)).elements);

    // Update the world matrices of an object and its parents and children

    object->matrix->identity();
    object->matrixWorld->identity();
    child->matrix->identity();
    child->matrixWorld->identity();

    object->updateWorldMatrix(true, true);

    REQUIRE(parent->matrix->elements == m.setPosition(parent->position).elements);

    REQUIRE(parent->matrixWorld->elements == m.setPosition(parent->position).elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position)).elements);

    REQUIRE(child->matrix->elements == m.setPosition(child->position).elements);

    REQUIRE(child->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position).add(child->position)).elements);

    // object->matrixAutoUpdate = false test

    object->matrix->identity();
    object->matrixWorld->identity();

    object->matrixAutoUpdate = false;
    object->updateWorldMatrix(true, false);

    REQUIRE(object->matrix->elements == m.identity().elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(parent->position).elements);
}


// ---------------------------------------------------------------------------
// Ownership / lifetime of the scene-graph links
// ---------------------------------------------------------------------------

TEST_CASE("addRef'd child unlinks itself on destruction") {

    auto parent = Object3D::create();

    {
        Object3D child;
        parent->addRef(child);
        REQUIRE(parent->children.size() == 1);
    }// child dies here; it was never owned by parent

    // Before the destructor unlinked itself, `children` kept a dangling raw
    // pointer and the traverse below was a use-after-free.
    CHECK(parent->children.empty());

    int visited = 0;
    parent->traverse([&](Object3D&) { ++visited; });
    CHECK(visited == 1);// just the parent
}

TEST_CASE("destroying a parent clears its children's parent pointer") {

    Object3D child;

    {
        auto parent = Object3D::create();
        parent->addRef(child);
        REQUIRE(child.parent != nullptr);
    }// parent dies; child (not owned) outlives it

    CHECK(child.parent == nullptr);
}

TEST_CASE("removeFromParent hands back ownership") {

    auto parent = Object3D::create();
    auto child = Object3D::create();
    parent->add(child);

    Object3D* raw = child.get();
    child.reset();// the parent is now the only owner

    SECTION("keeping the returned reference keeps the object alive") {
        auto kept = raw->removeFromParent();
        REQUIRE(kept);
        CHECK(kept.get() == raw);
        CHECK(kept->parent == nullptr);
        CHECK(parent->children.empty());
    }

    SECTION("an addRef'd child reports no ownership to hand back") {
        auto other = Object3D::create();
        Object3D loose;
        other->addRef(loose);
        CHECK(loose.removeFromParent() == nullptr);
        CHECK(other->children.empty());
    }
}

TEST_CASE("remove fires its event against a live object") {

    auto parent = Object3D::create();
    auto child = Object3D::create();
    parent->add(child);
    child.reset();// parent solely owns it

    Object3D* raw = parent->children.front();

    bool sawEvent = false;
    unsigned int idDuringEvent = 0;
    LambdaEventListener listener([&](Event&) {
        sawEvent = true;
        idDuringEvent = raw->id;// must not be a read of freed memory
    });
    raw->addEventListener("remove", listener);

    const unsigned int expectedId = raw->id;
    parent->remove(*raw);

    CHECK(sawEvent);
    CHECK(idDuringEvent == expectedId);
    CHECK(parent->children.empty());
}

// ---------------------------------------------------------------------------
// getObjectByName
// ---------------------------------------------------------------------------

TEST_CASE("getObjectByName matches on name and type together") {

    auto scene = Object3D::create();

    // A non-Camera node with the wanted name comes FIRST in traversal order.
    auto decoy = Object3D::create();
    decoy->name = "target";
    scene->add(decoy);

    auto wanted = PerspectiveCamera::create();
    wanted->name = "target";
    scene->add(wanted);

    // Untyped lookup finds the first name match, as before.
    CHECK(scene->getObjectByName("target") == decoy.get());

    // Typed lookup used to resolve the decoy and then fail the cast, returning
    // nullptr even though a matching Camera was present.
    CHECK(scene->getObjectByName<PerspectiveCamera>("target") == wanted.get());
}

TEST_CASE("getObjectByName returns null when no node matches the type") {

    auto scene = Object3D::create();
    auto child = Object3D::create();
    child->name = "plain";
    scene->add(child);

    CHECK(scene->getObjectByName("plain") == child.get());
    CHECK(scene->getObjectByName<PerspectiveCamera>("plain") == nullptr);
}

// ---------------------------------------------------------------------------
// renderOrder
// ---------------------------------------------------------------------------

TEST_CASE("renderOrder accepts negative values") {

    auto object = Object3D::create();
    object->renderOrder = -1;
    CHECK(object->renderOrder == -1);
    CHECK(object->renderOrder < 0);// would have wrapped huge when unsigned
}

// ---------------------------------------------------------------------------
// Graph integrity: add()/addRef() validation and move semantics
// ---------------------------------------------------------------------------

TEST_CASE("add ignores null") {

    auto parent = Object3D::create();
    parent->add(nullptr);// used to dereference the null pointer
    CHECK(parent->children.empty());
}

TEST_CASE("add rejects self-insertion") {

    auto object = Object3D::create();
    object->add(object);

    CHECK(object->children.empty());
    CHECK(object->parent == nullptr);

    // the graph must still be traversable (a self-cycle recursed forever)
    int visited = 0;
    object->traverse([&](Object3D&) { ++visited; });
    CHECK(visited == 1);
}

TEST_CASE("add rejects ancestor insertion") {

    auto root = Object3D::create();
    auto mid = Object3D::create();
    auto leaf = Object3D::create();
    root->add(mid);
    mid->add(leaf);

    leaf->add(root);// would create a cycle

    CHECK(leaf->children.empty());
    CHECK(root->parent == nullptr);

    int visited = 0;
    root->traverse([&](Object3D&) { ++visited; });
    CHECK(visited == 3);
}

TEST_CASE("re-adding an owned child does not duplicate ownership") {

    auto parent = Object3D::create();
    auto child = Object3D::create();
    parent->add(child);
    parent->add(child);

    REQUIRE(parent->children.size() == 1);
    CHECK(child->parent == parent.get());

    // exactly one owning reference must come back out
    auto kept = child->removeFromParent();
    CHECK(kept != nullptr);
    CHECK(parent->children.empty());
    CHECK(child.use_count() == 2);// this local + kept; a third would be a leaked duplicate
}

TEST_CASE("addRef reparenting a solely-owned child keeps it alive") {

    auto oldParent = Object3D::create();
    auto newParent = Object3D::create();

    auto child = Object3D::create();
    Object3D* raw = child.get();
    oldParent->add(child);
    child.reset();// oldParent is the only owner

    // addRef used to remove() from the old parent, destroying the child
    // mid-call; ownership now follows the object to the new parent.
    newParent->addRef(*raw);

    CHECK(oldParent->children.empty());
    REQUIRE(newParent->children.size() == 1);
    CHECK(newParent->children.front() == raw);
    CHECK(raw->parent == newParent.get());
    CHECK(raw->removeFromParent() != nullptr);// the new parent owned it
}

TEST_CASE("moving an addRef-attached node retargets the parent's child slot") {

    auto parent = Object3D::create();

    Object3D a;
    parent->addRef(a);
    REQUIRE(parent->children.front() == &a);

    Object3D b{std::move(a)};

    // the parent must track the surviving object, not the moved-from husk
    REQUIRE(parent->children.size() == 1);
    CHECK(parent->children.front() == &b);
    CHECK(b.parent == parent.get());
    CHECK(a.parent == nullptr);
}

TEST_CASE("moving a parent-owned node leaves the graph memory-safe") {

    auto parent = Object3D::create();
    auto child = Object3D::create();
    parent->add(child);

    // Ownership cannot follow a move to a new address: the hollowed-out source
    // stays attached where its owner expects it, the destination starts
    // detached, and no child list ends up pointing at freed memory.
    Object3D moved{std::move(*child)};

    CHECK(moved.parent == nullptr);
    REQUIRE(parent->children.size() == 1);
    CHECK(parent->children.front() == child.get());

    int visited = 0;
    parent->traverse([&](Object3D&) { ++visited; });
    CHECK(visited == 2);
}
