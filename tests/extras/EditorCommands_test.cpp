
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/Selection.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::shared_ptr<Object3D> named(const std::string& name) {

        auto object = Object3D::create();
        object->name = name;
        return object;
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
