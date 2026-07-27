
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

// Undo across Play. Stopping play restores the scene from a JSON snapshot,
// which destroys the graph every recorded command points into. The editor
// wires CommandStack::rebind into SceneDocument::onSceneReplaced so commands
// re-resolve their targets by uuid (stable through the snapshot round trip)
// — or are dropped when they cannot. That wiring is reproduced here; before
// it existed, undoing a pre-play edit after stop wrote through dangling
// pointers.

namespace {

    struct Harness {

        SceneDocument document;
        CommandStack stack;
        PlayController controller;

        Harness() {
            document.onSceneReplaced([this](Scene& scene) { stack.rebind(scene); });
        }

        Scene& scene() { return document.scene(); }

        // One full play round: snapshot, an idle update, stop-restore.
        void playRound() {
            std::string error;
            REQUIRE(controller.play(document, &error));
            controller.update(0.1f);
            REQUIRE(controller.stop(document, &error));
        }
    };

    std::shared_ptr<Object3D> named(const std::string& name) {

        auto object = Object3D::create();
        object->name = name;
        return object;
    }

}// namespace


TEST_CASE("undoing a pre-play transform targets the restored instance", "[editor]") {

    Harness h;
    auto box = named("Box");
    box->position.set(0, 0.5f, 0);
    h.scene().add(box);
    const auto uuid = box->uuid;

    const auto before = SetTransformCommand::read(*box);
    auto after = before;
    after.position.x = 4.f;
    h.stack.execute(std::make_unique<SetTransformCommand>(*box, before, after, "Move"));
    CHECK_THAT(box->position.x, WithinAbs(4.f, 1e-5));

    // Keep the pre-play scene alive so we can prove the undo did NOT go
    // through the old pointer (and so comparing against it stays legal).
    const auto oldScene = h.document.scenePtr();
    h.playRound();
    REQUIRE(&h.scene() != oldScene.get());

    auto* restored = findByUuid(h.scene(), uuid);
    REQUIRE(restored != nullptr);
    REQUIRE(restored != box.get());
    CHECK_THAT(restored->position.x, WithinAbs(4.f, 1e-5));

    REQUIRE(h.stack.canUndo());
    REQUIRE(h.stack.undo());
    CHECK_THAT(restored->position.x, WithinAbs(0.f, 1e-5));
    // The old instance is untouched: the command re-resolved by uuid.
    CHECK_THAT(box->position.x, WithinAbs(4.f, 1e-5));

    REQUIRE(h.stack.redo());
    CHECK_THAT(restored->position.x, WithinAbs(4.f, 1e-5));
}

TEST_CASE("property commands are dropped on scene replace instead of dangling", "[editor]") {

    Harness h;
    auto box = named("Box");
    h.scene().add(box);

    auto* raw = box.get();
    h.stack.execute(makeProperty<std::string>(
            "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "Box", "Crate"));
    CHECK(box->name == "Crate");
    CHECK(h.stack.canUndo());

    h.playRound();

    // The setter captured a raw pointer into the old graph; the stack dropped
    // the command rather than let undo write through it.
    CHECK_FALSE(h.stack.canUndo());
    CHECK_FALSE(h.stack.undo());
    // The rename itself was captured by the snapshot and survives.
    CHECK(h.scene().getObjectByName("Crate") != nullptr);
}

TEST_CASE("undoing a pre-play delete reinserts the retained subtree", "[editor]") {

    Harness h;
    auto keeper = named("Keeper");
    auto doomed = Group::create();
    doomed->name = "Doomed";
    doomed->add(named("Child"));
    h.scene().add(keeper);
    h.scene().add(doomed);

    auto remove = std::make_unique<RemoveObjectCommand>(*doomed, "Delete");
    REQUIRE(remove->valid());
    h.stack.execute(std::move(remove));
    CHECK(h.scene().children.size() == 1);

    h.playRound();
    CHECK(h.scene().getObjectByName("Doomed") == nullptr);

    // The deleted subtree was never in the snapshot; the command retained the
    // ORIGINAL instance, children and all, and reinserts it into the new
    // scene under the uuid-resolved parent.
    REQUIRE(h.stack.undo());
    auto* back = h.scene().getObjectByName("Doomed");
    REQUIRE(back != nullptr);
    CHECK(back == doomed.get());
    CHECK(back->parent == &h.scene());
    REQUIRE(back->children.size() == 1);
    CHECK(back->children[0]->name == "Child");

    REQUIRE(h.stack.redo());
    CHECK(h.scene().getObjectByName("Doomed") == nullptr);
}

TEST_CASE("undoing a pre-play add removes the restored instance", "[editor]") {

    Harness h;
    auto added = named("Added");
    h.stack.execute(std::make_unique<AddObjectCommand>(h.scene(), added, "Add"));
    const auto uuid = added->uuid;

    const auto oldScene = h.document.scenePtr();
    h.playRound();

    auto* restored = findByUuid(h.scene(), uuid);
    REQUIRE(restored != nullptr);
    REQUIRE(restored != added.get());

    REQUIRE(h.stack.undo());
    CHECK(findByUuid(h.scene(), uuid) == nullptr);
    // The original instance still hangs where it always did; the undo removed
    // the adopted restored one.
    CHECK(added->parent == oldScene.get());

    REQUIRE(h.stack.redo());
    CHECK(findByUuid(h.scene(), uuid) == restored);
}

TEST_CASE("undoing a pre-play reparent restores the original parent by uuid", "[editor]") {

    Harness h;
    auto group = Group::create();
    group->name = "Group";
    group->position.set(10, 0, 0);
    h.scene().add(group);

    auto object = named("Object");
    object->position.set(1, 2, 3);
    h.scene().add(object);
    h.scene().updateMatrixWorld(true);

    auto command = std::make_unique<ReparentCommand>(*object, *group, "Reparent");
    REQUIRE(command->valid());
    h.stack.execute(std::move(command));

    const auto objectUuid = object->uuid;
    const auto groupUuid = group->uuid;

    h.playRound();

    auto* restoredObject = findByUuid(h.scene(), objectUuid);
    auto* restoredGroup = findByUuid(h.scene(), groupUuid);
    REQUIRE(restoredObject != nullptr);
    REQUIRE(restoredGroup != nullptr);
    CHECK(restoredObject->parent == restoredGroup);

    REQUIRE(h.stack.undo());
    CHECK(restoredObject->parent == &h.scene());
    CHECK_THAT(restoredObject->position.x, WithinAbs(1.f, 1e-4));

    REQUIRE(h.stack.redo());
    CHECK(restoredObject->parent == restoredGroup);
    CHECK_THAT(restoredObject->position.x, WithinAbs(-9.f, 1e-4));
}

TEST_CASE("commands undone before play redo onto the restored scene", "[editor]") {

    Harness h;
    auto box = named("Box");
    h.scene().add(box);
    const auto uuid = box->uuid;

    const auto before = SetTransformCommand::read(*box);
    auto after = before;
    after.position.y = 2.f;
    h.stack.execute(std::make_unique<SetTransformCommand>(*box, before, after, "Move"));
    REQUIRE(h.stack.undo());// parked on the redo stack across play
    CHECK_THAT(box->position.y, WithinAbs(0.f, 1e-5));

    h.playRound();

    REQUIRE(h.stack.canRedo());
    REQUIRE(h.stack.redo());
    auto* restored = findByUuid(h.scene(), uuid);
    REQUIRE(restored != nullptr);
    CHECK_THAT(restored->position.y, WithinAbs(2.f, 1e-5));
}

TEST_CASE("unresolvable commands are pruned, the rest keep working", "[editor]") {

    Harness h;
    auto box = named("Box");
    h.scene().add(box);
    const auto uuid = box->uuid;

    auto t1 = SetTransformCommand::read(*box);
    auto t2 = t1;
    t2.position.x = 1.f;
    h.stack.execute(std::make_unique<SetTransformCommand>(*box, t1, t2, "Move"));

    auto* raw = box.get();
    h.stack.execute(makeProperty<std::string>(
            "Rename", {}, [raw](const std::string& v) { raw->name = v; }, "Box", "Crate"));

    auto t3 = t2;
    t3.position.x = 5.f;
    h.stack.execute(std::make_unique<SetTransformCommand>(*box, t2, t3, "Move"));

    CHECK(h.stack.undoCount() == 3);
    h.playRound();
    // The property command in the middle is gone; both transforms survive.
    CHECK(h.stack.undoCount() == 2);

    auto* restored = findByUuid(h.scene(), uuid);
    REQUIRE(restored != nullptr);
    REQUIRE(h.stack.undo());
    CHECK_THAT(restored->position.x, WithinAbs(1.f, 1e-5));
    REQUIRE(h.stack.undo());
    CHECK_THAT(restored->position.x, WithinAbs(0.f, 1e-5));
}
