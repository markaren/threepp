
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/Selection.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::shared_ptr<Object3D> named(const std::string& name) {

        auto object = Object3D::create();
        object->name = name;
        return object;
    }

    // Enough splats to be worth weighing (~6 MB at SH degree 0) and few enough
    // that a test builds several without noticing. The budget tests set their
    // limit from the measured weight rather than a hardcoded number, so the
    // count here is free to change.
    constexpr std::size_t kTestSplats = 50'000;

    std::shared_ptr<SplatCloud> splatScan(std::size_t splats) {

        SplatData data;
        data.resize(splats, 0);
        return SplatCloud::create(std::move(data));
    }

}// namespace


TEST_CASE("CommandStack applies, undoes and redoes", "[editor]") {

    auto object = named("a");
    CommandStack stack;

    REQUIRE_FALSE(stack.canUndo());
    REQUIRE_FALSE(stack.canRedo());

    auto* raw = object.get();
    stack.execute(makeProperty<std::string>(
            "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "a", "b"));

    CHECK(object->name == "b");
    CHECK(stack.canUndo());
    CHECK(stack.undoName() == "Rename");

    REQUIRE(stack.undo());
    CHECK(object->name == "a");
    CHECK(stack.canRedo());

    REQUIRE(stack.redo());
    CHECK(object->name == "b");
}

TEST_CASE("a new command clears the redo branch", "[editor]") {

    auto object = named("a");
    auto* raw = object.get();
    CommandStack stack;

    stack.execute(makeProperty<std::string>(
            "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "a", "b"));
    REQUIRE(stack.undo());
    REQUIRE(stack.canRedo());

    stack.execute(makeProperty<std::string>(
            "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "a", "c"));

    CHECK_FALSE(stack.canRedo());
    CHECK(object->name == "c");
}

TEST_CASE("a transaction coalesces a drag into one undo step", "[editor]") {

    auto object = named("dragged");
    const auto start = SetTransformCommand::read(*object);

    CommandStack stack;
    stack.beginTransaction();
    for (int i = 1; i <= 10; ++i) {
        auto after = start;
        after.position.set(static_cast<float>(i), 0, 0);
        stack.execute(std::make_unique<SetTransformCommand>(*object, SetTransformCommand::read(*object), after, "Move"));
    }
    stack.endTransaction();

    CHECK(object->position.x == 10.f);
    // Ten frames of dragging, one entry.
    CHECK(stack.undoCount() == 1);

    REQUIRE(stack.undo());
    CHECK(object->position.x == 0.f);
    CHECK_FALSE(stack.canUndo());
}

TEST_CASE("commands outside a transaction are not merged", "[editor]") {

    auto object = named("stepped");
    CommandStack stack;

    for (int i = 1; i <= 3; ++i) {
        auto after = SetTransformCommand::read(*object);
        after.position.set(static_cast<float>(i), 0, 0);
        stack.execute(std::make_unique<SetTransformCommand>(*object, SetTransformCommand::read(*object), after, "Move"));
    }

    CHECK(stack.undoCount() == 3);
    REQUIRE(stack.undo());
    CHECK(object->position.x == 2.f);
}

TEST_CASE("commands for different objects never merge", "[editor]") {

    auto first = named("first");
    auto second = named("second");

    CommandStack stack;
    stack.beginTransaction();

    auto afterFirst = SetTransformCommand::read(*first);
    afterFirst.position.x = 5.f;
    stack.execute(std::make_unique<SetTransformCommand>(*first, SetTransformCommand::read(*first), afterFirst, "Move"));

    auto afterSecond = SetTransformCommand::read(*second);
    afterSecond.position.y = 7.f;
    stack.execute(std::make_unique<SetTransformCommand>(*second, SetTransformCommand::read(*second), afterSecond, "Move"));

    stack.endTransaction();

    CHECK(stack.undoCount() == 2);
    CHECK(first->position.x == 5.f);
    CHECK(second->position.y == 7.f);
}

TEST_CASE("AddObjectCommand adds and removes", "[editor]") {

    auto scene = Scene::create();
    auto object = named("added");

    CommandStack stack;
    stack.execute(std::make_unique<AddObjectCommand>(*scene, object, "Add"));

    REQUIRE(scene->children.size() == 1);
    CHECK(scene->children[0] == object.get());

    REQUIRE(stack.undo());
    CHECK(scene->children.empty());
    // The command keeps the object alive so redo has something to add back.
    CHECK(object.use_count() >= 1);

    REQUIRE(stack.redo());
    REQUIRE(scene->children.size() == 1);
    CHECK(scene->children[0]->name == "added");
}

TEST_CASE("RemoveObjectCommand is valid before execution", "[editor]") {

    // The editor consults valid() BEFORE executing; it must mean "this object
    // can be removed", not "the removal already happened". Regression: it once
    // tested the ownership handle that only redo() populates, which vetoed
    // every delete in the app.
    auto scene = Scene::create();
    auto parented = named("parented");
    scene->add(parented);
    CHECK(RemoveObjectCommand(*parented, "Delete").valid());

    auto orphan = named("orphan");
    CHECK_FALSE(RemoveObjectCommand(*orphan, "Delete").valid());
}

TEST_CASE("RemoveObjectCommand restores parent and child index", "[editor]") {

    auto scene = Scene::create();
    auto a = named("a");
    auto b = named("b");
    auto c = named("c");
    scene->add(a);
    scene->add(b);
    scene->add(c);

    CommandStack stack;
    auto remove = std::make_unique<RemoveObjectCommand>(*b, "Delete");
    REQUIRE(remove->valid());
    stack.execute(std::move(remove));

    REQUIRE(scene->children.size() == 2);
    CHECK(scene->children[0]->name == "a");
    CHECK(scene->children[1]->name == "c");

    REQUIRE(stack.undo());

    REQUIRE(scene->children.size() == 3);
    CHECK(scene->children[0]->name == "a");
    // Back in the middle, not appended at the end.
    CHECK(scene->children[1]->name == "b");
    CHECK(scene->children[1] == b.get());
    CHECK(scene->children[2]->name == "c");
    CHECK(b->parent == scene.get());
}

TEST_CASE("removing a nested subtree keeps its children", "[editor]") {

    auto scene = Scene::create();
    auto group = Group::create();
    group->name = "group";
    auto child = named("child");
    group->add(child);
    scene->add(group);

    CommandStack stack;
    stack.execute(std::make_unique<RemoveObjectCommand>(*group, "Delete"));
    CHECK(scene->children.empty());

    REQUIRE(stack.undo());
    REQUIRE(scene->children.size() == 1);
    REQUIRE(scene->children[0]->children.size() == 1);
    CHECK(scene->children[0]->children[0]->name == "child");
}

TEST_CASE("ReparentCommand preserves the world transform", "[editor]") {

    auto scene = Scene::create();
    auto group = Group::create();
    group->name = "group";
    group->position.set(10, 0, 0);
    scene->add(group);

    auto object = named("object");
    object->position.set(1, 2, 3);
    scene->add(object);

    scene->updateMatrixWorld(true);

    CommandStack stack;
    auto command = std::make_unique<ReparentCommand>(*object, *group, "Reparent");
    REQUIRE(command->valid());
    stack.execute(std::move(command));

    CHECK(object->parent == group.get());
    // World position unchanged: local x is now 1 - 10.
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(object->position.x, WithinAbs(-9.f, 1e-4));

    REQUIRE(stack.undo());
    CHECK(object->parent == scene.get());
    CHECK_THAT(object->position.x, WithinAbs(1.f, 1e-4));
    CHECK(scene->children.size() == 2);
}

TEST_CASE("reparenting under a descendant is rejected", "[editor]") {

    auto scene = Scene::create();
    auto group = Group::create();
    auto child = Group::create();
    group->add(child);
    scene->add(group);

    ReparentCommand command(*group, *child, "Reparent");
    CHECK_FALSE(command.valid());
}

TEST_CASE("insertChildAt keeps referenced children referenced", "[editor]") {

    auto scene = Scene::create();
    auto owned = named("owned");
    auto overlay = named("overlay");// attached by reference, like editor helpers

    scene->add(owned);
    scene->addRef(*overlay);

    auto inserted = named("inserted");
    insertChildAt(*scene, inserted, 0);

    REQUIRE(scene->children.size() == 3);
    CHECK(scene->children[0]->name == "inserted");
    CHECK(scene->children[1]->name == "owned");
    CHECK(scene->children[2]->name == "overlay");
    // The referenced child is still owned by us alone.
    CHECK(overlay.use_count() == 1);
}

// ---------------------------------------------------------------- byte budget
//
// The count cap says nothing about size, and the editor's heavyweight is the
// Gaussian-splat scan: one deleted 6M-splat cloud held for undo is 1.8 GB of
// host memory. These pin the budget that bounds it — and, just as much, the two
// places it deliberately refuses to bite.

TEST_CASE("retainedSubtreeBytes weighs splat clouds and nothing else", "[editor]") {

    auto plain = Group::create();
    plain->add(named("child"));
    CHECK(retainedSubtreeBytes(*plain) == 0);

    auto scan = splatScan(kTestSplats);
    const auto weight = retainedSubtreeBytes(*scan);

    // The splat arrays plus the identity instanceMatrix the base class allocates
    // and this class never uses — 64 bytes a splat that a budget has to see,
    // since dropping the command really does free them. No GL copy in the total:
    // nothing has rendered this cloud (that is what makes a Vulkan-only scan
    // cheaper to hold than a drawn one).
    const auto floorBytes = scan->data().byteSize() + scan->splatCount() * 16 * sizeof(float);
    CHECK(weight >= floorBytes);
    CHECK(weight < 2 * floorBytes);
    CHECK_FALSE(scan->glResourcesBuilt());

    // Found through a subtree, not just at the root.
    auto wrapper = Group::create();
    wrapper->add(scan);
    CHECK(retainedSubtreeBytes(*wrapper) == weight);
}

TEST_CASE("the budget charges the history only for detached subtrees", "[editor]") {

    auto scene = Scene::create();
    auto scan = splatScan(kTestSplats);
    const auto weight = retainedSubtreeBytes(*scan);
    REQUIRE(weight > 0);

    CommandStack stack;
    stack.execute(std::make_unique<AddObjectCommand>(*scene, scan, "Import"));

    // In the scene: co-owned, so the history is not what keeps it alive.
    CHECK(stack.retainedBytes() == 0);

    REQUIRE(stack.undo());
    CHECK(stack.retainedBytes() == weight);

    REQUIRE(stack.redo());
    CHECK(stack.retainedBytes() == 0);
}

TEST_CASE("one subtree held by two commands is counted once", "[editor]") {

    auto scene = Scene::create();
    auto scan = splatScan(kTestSplats);
    const auto weight = retainedSubtreeBytes(*scan);

    CommandStack stack;
    stack.execute(std::make_unique<AddObjectCommand>(*scene, scan, "Import"));
    stack.execute(std::make_unique<RemoveObjectCommand>(*scan, "Delete"));

    // Both commands hold the same detached cloud now. Summing per command would
    // report double the memory that dropping either would free.
    CHECK(stack.retainedBytes() == weight);
}

TEST_CASE("the byte budget evicts the oldest deletion that no longer fits", "[editor]") {

    auto scene = Scene::create();
    auto first = splatScan(kTestSplats);
    auto second = splatScan(kTestSplats);
    scene->add(first);
    scene->add(second);

    const auto weight = retainedSubtreeBytes(*first);
    CommandStack stack(CommandStack::defaultLimit, weight + weight / 2);

    std::vector<std::string> prunedNames;
    std::size_t prunedBytes = 0;
    stack.onPrune([&](const std::vector<std::string>& dropped, std::size_t bytes) {
        prunedNames.insert(prunedNames.end(), dropped.begin(), dropped.end());
        prunedBytes += bytes;
    });

    stack.execute(std::make_unique<RemoveObjectCommand>(*first, "Delete First"));
    REQUIRE(stack.retainedBytes() == weight);
    REQUIRE(prunedNames.empty());

    stack.execute(std::make_unique<RemoveObjectCommand>(*second, "Delete Second"));

    CHECK(stack.undoCount() == 1);
    CHECK(stack.undoName() == "Delete Second");
    CHECK(stack.retainedBytes() == weight);
    // Named, not counted: the user has to be told which undo left.
    CHECK(prunedNames == std::vector<std::string>{"Delete First"});
    CHECK(prunedBytes == weight);
    // Really let go: this test's own pointer is the last owner of the first scan.
    CHECK(first.use_count() == 1);
}

TEST_CASE("the newest undo entry survives a budget it cannot fit", "[editor]") {

    auto scene = Scene::create();
    auto scan = splatScan(kTestSplats);
    scene->add(scan);

    CommandStack stack(CommandStack::defaultLimit, 1);
    stack.execute(std::make_unique<RemoveObjectCommand>(*scan, "Delete"));

    // Over budget by design: a deletion the budget cannot hold is still a
    // deletion the user must be able to take back.
    CHECK(stack.undoCount() == 1);
    CHECK(stack.retainedBytes() == retainedSubtreeBytes(*scan));

    REQUIRE(stack.undo());
    CHECK(scan->parent == scene.get());
    CHECK(stack.retainedBytes() == 0);
}

TEST_CASE("a held deletion survives unrelated edits pushed after it", "[editor]") {

    // Protecting only the NEWEST entry would keep this promise for exactly one
    // push: move a box after deleting a scan the budget cannot hold, and the
    // deletion becomes the second-newest entry and therefore droppable — the undo
    // would vanish while the user was looking somewhere else. What is protected is
    // the newest entry still HOLDING something, and everything newer than it.
    auto scene = Scene::create();
    auto scan = splatScan(kTestSplats);
    auto box = named("box");
    scene->add(scan);
    scene->add(box);
    auto* raw = box.get();

    CommandStack stack(CommandStack::defaultLimit, 1);
    stack.execute(std::make_unique<RemoveObjectCommand>(*scan, "Delete Scan"));
    REQUIRE(stack.undoCount() == 1);

    for (int i = 0; i < 5; ++i) {
        stack.execute(makeProperty<std::string>(
                "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "box", "renamed"));
    }

    CHECK(stack.undoCount() == 6);
    CHECK(stack.retainedBytes() > stack.byteLimit());// exceeded, not truncated

    // And the deletion is genuinely still there to undo.
    for (int i = 0; i < 5; ++i) REQUIRE(stack.undo());
    CHECK(stack.undoName() == "Delete Scan");
    REQUIRE(stack.undo());
    CHECK(scan->parent == scene.get());
}

TEST_CASE("the count cap reports a dropped entry that was holding memory", "[editor]") {

    // The byte budget is not the only thing that can drop a held scan — a session
    // long enough to hit the 200-command cap does it too, and owes the same
    // notice rather than silently losing a multi-gigabyte undo.
    auto scene = Scene::create();
    auto scan = splatScan(kTestSplats);
    scene->add(scan);
    auto box = named("box");
    scene->add(box);
    auto* raw = box.get();

    CommandStack stack(/*limit*/ 2);
    std::vector<std::string> prunedNames;
    std::size_t prunedBytes = 0;
    stack.onPrune([&](const std::vector<std::string>& dropped, std::size_t bytes) {
        prunedNames.insert(prunedNames.end(), dropped.begin(), dropped.end());
        prunedBytes += bytes;
    });

    stack.execute(std::make_unique<RemoveObjectCommand>(*scan, "Delete Scan"));
    const auto weight = stack.retainedBytes();
    REQUIRE(weight > 0);

    // Two more edits: the cap is 2, so the deletion falls off the bottom.
    for (int i = 0; i < 2; ++i) {
        stack.execute(makeProperty<std::string>(
                "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "box", "renamed"));
    }

    CHECK(stack.undoCount() == 2);
    CHECK(prunedNames == std::vector<std::string>{"Delete Scan"});
    CHECK(prunedBytes == weight);
    CHECK(stack.retainedBytes() == 0);
    CHECK(scan.use_count() == 1);
}

TEST_CASE("the redo branch is relieved before the undo history", "[editor]") {

    auto scene = Scene::create();
    auto kept = splatScan(kTestSplats);
    auto added = splatScan(kTestSplats);
    scene->add(kept);

    const auto weight = retainedSubtreeBytes(*kept);
    CommandStack stack(CommandStack::defaultLimit, weight + weight / 2);

    // An applied deletion below, an undone import above: the one arrangement
    // where both stacks hold a detached scan at the same time.
    stack.execute(std::make_unique<RemoveObjectCommand>(*kept, "Delete Kept"));
    stack.execute(std::make_unique<AddObjectCommand>(*scene, added, "Import"));
    REQUIRE(stack.undoCount() == 2);

    REQUIRE(stack.undo());

    // Two scans do not fit. The speculative future pays, not the history.
    CHECK(stack.undoCount() == 1);
    CHECK(stack.undoName() == "Delete Kept");
    CHECK_FALSE(stack.canRedo());
    CHECK(stack.retainedBytes() == weight);
    CHECK(added.use_count() == 1);
}

TEST_CASE("lowering the byte limit prunes on the spot", "[editor]") {

    auto scene = Scene::create();
    auto first = splatScan(kTestSplats);
    auto second = splatScan(kTestSplats);
    scene->add(first);
    scene->add(second);

    CommandStack stack;
    stack.execute(std::make_unique<RemoveObjectCommand>(*first, "Delete First"));
    stack.execute(std::make_unique<RemoveObjectCommand>(*second, "Delete Second"));
    REQUIRE(stack.undoCount() == 2);

    const auto weight = retainedSubtreeBytes(*second);
    stack.setByteLimit(weight + weight / 2);

    CHECK(stack.undoCount() == 1);
    CHECK(first.use_count() == 1);
}

TEST_CASE("an ordinary editing session retains nothing", "[editor]") {

    auto scene = Scene::create();
    auto object = named("a");
    scene->add(object);
    auto* raw = object.get();

    CommandStack stack;
    for (int i = 0; i < 50; ++i) {
        stack.execute(makeProperty<std::string>(
                "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "a", "b"));
    }

    CHECK(stack.retainedBytes() == 0);
    CHECK(stack.undoCount() == 50);
}

TEST_CASE("Selection notifies only on change", "[editor]") {

    auto a = named("a");
    auto b = named("b");

    Selection selection;
    int notifications = 0;
    selection.onChange([&](Object3D*) { ++notifications; });

    selection.set(a.get());
    selection.set(a.get());
    CHECK(notifications == 1);
    CHECK(selection.uuid() == a->uuid);

    selection.set(b.get());
    CHECK(notifications == 2);

    selection.clear();
    CHECK(notifications == 3);
    CHECK(selection.empty());
    CHECK(selection.uuid().empty());
}

TEST_CASE("ObjectFactory names are unique within the scene", "[editor]") {

    auto scene = Scene::create();

    auto first = ObjectFactory::createPrimitive(Primitive::Box, *scene);
    CHECK(first->name == "Box");
    scene->add(first);

    auto second = ObjectFactory::createPrimitive(Primitive::Box, *scene);
    CHECK(second->name == "Box 2");
    scene->add(second);

    auto third = ObjectFactory::createPrimitive(Primitive::Box, *scene);
    CHECK(third->name == "Box 3");

    // A different type starts its own sequence.
    auto sphere = ObjectFactory::createPrimitive(Primitive::Sphere, *scene);
    CHECK(sphere->name == "Sphere");

    auto light = ObjectFactory::createLight(LightKind::Point, *scene);
    CHECK(light->name == "Point Light");

    auto group = ObjectFactory::createGroup(*scene);
    CHECK(group->name == "Group");
}

TEST_CASE("ObjectFactory primitives are usable meshes", "[editor]") {

    auto scene = Scene::create();

    for (const auto type : ObjectFactory::primitives) {
        auto mesh = ObjectFactory::createPrimitive(type, *scene);
        REQUIRE(mesh);
        REQUIRE(mesh->geometry());
        CHECK(mesh->material());
        CHECK(mesh->geometry()->getAttribute<float>("position") != nullptr);
    }
}
