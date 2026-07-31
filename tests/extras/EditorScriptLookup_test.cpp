
#include <catch2/catch_test_macros.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Inline source throughout: the code under test is the session's instance
    // list, not the loader, and inline scripts need no files to keep in step.
    std::shared_ptr<Group> attach(Scene& scene, const std::string& name, const std::string& source,
                                  const std::vector<ScriptConfig::Field>& fields = {}) {

        auto group = Group::create();
        group->name = name;
        ScriptConfig config;
        config.source = source;
        config.fields = fields;
        config.write(*group);
        scene.add(group);
        return group;
    }

    std::shared_ptr<Group> plain(Scene& scene, const std::string& name) {

        auto group = Group::create();
        group->name = name;
        scene.add(group);
        return group;
    }

    // The pair from the doc, in one direction and the other. Both resolve their
    // neighbour in start() — which is the whole point of the two-phase start —
    // and both signal it from update(), one by calling a method and one by
    // setting an attribute. Nothing here is an event bus: the instance IS the
    // API.
    constexpr const char* kDoor = R"(
import threepp


class Door:
    opened = 0

    def start(self, obj):
        self.obj = obj
        # The Button exists by now whether or not its own start() has run.
        self.obj.position.y = 1.0 if threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Button")) is not None else -1.0

    def on_opened(self):
        self.opened += 1
        self.obj.position.x = float(self.opened)

    def update(self, dt):
        # ...and mid-session too, which is the other half of the seam. Note the
        # scene is asked for again rather than stashed in start(): it answers for
        # as long as the session runs.
        button = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Button"))
        if button is not None:
            button.acknowledged = True
)";

    constexpr const char* kButton = R"(
import threepp


class Button:
    acknowledged = False

    def start(self, obj):
        self.obj = obj
        self.door = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Door"))
        self.obj.position.y = 1.0 if self.door is not None else -1.0

    def update(self, dt):
        if self.door is not None:
            self.door.on_opened()
        if self.acknowledged:
            self.obj.position.z = 1.0
)";

}// namespace


TEST_CASE("two scripts signal each other through their instances", "[editor][scripting]") {

    auto scene = Scene::create();
    auto button = attach(*scene, "Button", kButton);
    auto door = attach(*scene, "Door", kDoor);

    ScriptPlaySession session;
    session.start(*scene);
    for (int i = 0; i < 3; ++i) session.update(0.1f);

    CHECK(session.errorCount() == 0);
    // Each found the other in its own start().
    CHECK(button->position.y == 1.f);
    CHECK(door->position.y == 1.f);
    // Button called a method on Door's instance, once per frame.
    CHECK(door->position.x == 3.f);
    // Door set an attribute on Button's instance, and Button read it back.
    CHECK(button->position.z == 1.f);

    session.stop();
}

TEST_CASE("the lookup does not depend on scene order", "[editor][scripting]") {

    // The same pair, declared the other way round. Interleaved instantiation and
    // start() would leave whichever script came first resolving None; two phases
    // make the question order-free.
    auto scene = Scene::create();
    auto door = attach(*scene, "Door", kDoor);
    auto button = attach(*scene, "Button", kButton);

    ScriptPlaySession session;
    session.start(*scene);
    for (int i = 0; i < 3; ++i) session.update(0.1f);

    CHECK(session.errorCount() == 0);
    CHECK(door->position.y == 1.f);
    CHECK(button->position.y == 1.f);
    CHECK(door->position.x == 3.f);
    CHECK(button->position.z == 1.f);

    session.stop();
}

TEST_CASE("authored fields are on every instance before any start runs", "[editor][scripting]") {

    // Reader comes FIRST, so its start() reads a neighbour whose own start() has
    // not run — and must still see the value the document authored rather than
    // the class attribute the source declares.
    auto scene = Scene::create();
    auto reader = attach(*scene, "Reader", R"(
import threepp


class Reader:
    def start(self, obj):
        other = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Tuned"))
        obj.position.x = other.speed if other is not None else -1.0
)");
    attach(*scene, "Tuned", R"(
class Tuned:
    speed = 1.5

    def update(self, dt):
        pass
)",
           {{"speed", "7"}});

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(reader->position.x == 7.f);
}

TEST_CASE("a script resolves itself to its own instance", "[editor][scripting]") {

    // Identity, not equality: what comes back is the object the session drives.
    constexpr const char* mirror = R"(
import threepp


class Mirror:
    def start(self, obj):
        self.obj = obj
        obj.position.x = 1.0 if threepp.editor.script_from_object(obj) is self else -1.0

    def stop(self):
        # Still inside the session: nothing is released until the sweep ends.
        self.obj.position.z = 1.0 if threepp.editor.script_from_object(
            self.obj) is self else -1.0
)";

    auto scene = Scene::create();
    auto group = attach(*scene, "Mirror", mirror);

    // ...and on a Mesh, which reaches Object3D through a VIRTUAL base — the
    // handle has to survive the round trip back into a uuid.
    auto mesh = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
    mesh->name = "MirrorMesh";
    ScriptConfig config;
    config.source = mirror;
    config.write(*mesh);
    scene->add(mesh);

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(group->position.x == 1.f);
    CHECK(mesh->position.x == 1.f);
    // ...and from stop(), the last moment of the session.
    CHECK(group->position.z == 1.f);
    CHECK(mesh->position.z == 1.f);
}

TEST_CASE("the lookup names the exact object, never an ancestor", "[editor][scripting]") {

    // A collider governs a subtree, so rigid_body_from_object walks up. A script
    // governs nothing but the node it was authored on, so this does not.
    auto scene = Scene::create();
    auto parent = attach(*scene, "Parent", R"(
import threepp


class Parent:
    def start(self, obj):
        child = threepp.editor.scene().get_object_by_name("Child")
        obj.position.x = 1.0 if threepp.editor.script_from_object(child) is None else -1.0
)");
    auto child = Group::create();
    child->name = "Child";
    parent->add(child);

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(parent->position.x == 1.f);
}

TEST_CASE("a missing or dead script resolves to None", "[editor][scripting]") {

    auto scene = Scene::create();
    plain(*scene, "Bare");
    attach(*scene, "Boom", R"(
class Boom:
    def __init__(self):
        raise RuntimeError("no instance for you")

    def update(self, dt):
        pass
)");
    // Before the Observer in the scene, so it has already raised and been
    // disabled by the time the same sweep reaches the lookup below.
    attach(*scene, "Thrower", R"(
class Thrower:
    def update(self, dt):
        raise RuntimeError("every single frame")
)");
    auto observer = attach(*scene, "Observer", R"(
import threepp


class Observer:
    def start(self, obj):
        self.obj = obj
        self.frames = 0
        scene = threepp.editor.scene()
        # No script at all.
        obj.position.x = 1.0 if threepp.editor.script_from_object(
            scene.get_object_by_name("Bare")) is None else -1.0
        # A constructor that raised: phase 1 recorded the failure, so there is
        # no instance to hand out.
        obj.position.y = 1.0 if threepp.editor.script_from_object(
            scene.get_object_by_name("Boom")) is None else -1.0
        # This one came up and has not raised yet.
        obj.position.z = 1.0 if threepp.editor.script_from_object(
            scene.get_object_by_name("Thrower")) is not None else -1.0

    def update(self, dt):
        self.frames += 1
        if self.frames == 1:
            # ...and now it has. A disabled script is dead to a new lookup.
            self.obj.rotation.x = 1.0 if threepp.editor.script_from_object(
                threepp.editor.scene().get_object_by_name("Thrower")) is None else -1.0
)");

    ScriptPlaySession session;
    session.start(*scene);
    session.update(0.1f);
    session.stop();

    // The two broken scripts, and nothing from the Observer.
    CHECK(session.errorCount() == 2);
    CHECK(session.errorFor(observer->uuid).empty());

    CHECK(observer->position.x == 1.f);
    CHECK(observer->position.y == 1.f);
    CHECK(observer->position.z == 1.f);
    CHECK(observer->rotation.x == 1.f);
}

TEST_CASE("a handle kept across sessions is not the new instance", "[editor][scripting]") {

    // The stash has to outlive a session, and inline modules are rebuilt from
    // scratch on every Play — so it goes somewhere neither the session nor the
    // loader owns. Cleared again at the end of the test.
    auto scene = Scene::create();
    auto group = attach(*scene, "Keeper", R"(
import builtins
import threepp


class Keeper:
    def start(self, obj):
        self.marker = 42
        me = threepp.editor.script_from_object(obj)
        obj.position.y = 1.0 if me is self else -1.0

        previous = getattr(builtins, "_threepp_lookup_previous", None)
        if previous is None:
            obj.position.x = 0.0
        elif previous is me:
            obj.position.x = -1.0
        else:
            obj.position.x = 1.0
            # A stale handle is a harmless dead object, not a dangling one:
            # reading it still works, it just is not what is playing.
            obj.position.z = 1.0 if previous.marker == 42 else -1.0
        builtins._threepp_lookup_previous = me
)");

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(group->position.y == 1.f);
    CHECK(group->position.x == 0.f);// nothing stashed yet

    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    // The fresh lookup is this session's instance...
    CHECK(group->position.y == 1.f);
    // ...and the one kept from last time is not.
    CHECK(group->position.x == 1.f);
    CHECK(group->position.z == 1.f);

    // Put the interpreter back the way we found it — the stash is the only
    // thing in this file that outlives its session on purpose.
    ScriptConfig cleanup;
    cleanup.source = R"(
import builtins


class Cleanup:
    def start(self, obj):
        if hasattr(builtins, "_threepp_lookup_previous"):
            del builtins._threepp_lookup_previous
)";
    cleanup.write(*group);
    session.start(*scene);
    session.stop();
    CHECK(session.errorCount() == 0);
}

TEST_CASE("threepp.editor.scene() answers for the whole session and no longer",
          "[editor][scripting]") {

    // The scene used to arrive as a second argument to start(), which made it
    // reachable in start() and NOWHERE ELSE — a script wanting it from update()
    // had to stash it on self. It is a named call now, so the interesting claim
    // is the LIFETIME: every callback can ask, and nothing can ask outside a
    // session.
    auto scene = Scene::create();
    auto watcher = attach(*scene, "Watcher", R"(
import threepp


class Watcher:
    def start(self, obj):
        self.obj = obj
        # The scene knows this object by name, which is the identity check:
        # a wrong scene would not have it.
        self.obj.position.x = 1.0 if threepp.editor.scene().get_object_by_name(
            "Watcher").uuid == obj.uuid else -1.0

    def update(self, dt):
        # Asked for again rather than stashed. This is the half that could not
        # be written at all before.
        self.obj.position.y = 1.0 if threepp.editor.scene().get_object_by_name(
            "Watcher") is not None else -1.0

    def stop(self):
        # Still inside the session: the accessor goes down WITH the instances,
        # not before them.
        self.obj.position.z = 1.0 if threepp.editor.scene() is not None else -1.0
)");

    // Nothing playing, nothing generating: nothing to answer with.
    CHECK(scripting::playScene() == nullptr);

    ScriptPlaySession session;
    session.start(*scene);
    CHECK(scripting::playScene() == scene.get());

    session.update(0.1f);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(watcher->position.x == 1.f);
    CHECK(watcher->position.y == 1.f);
    CHECK(watcher->position.z == 1.f);
    // Down again, so a script handle kept across sessions cannot reach a
    // document nobody is playing.
    CHECK(scripting::playScene() == nullptr);
}

TEST_CASE("the documented Button/Door pair runs as written", "[editor][scripting]") {

    // doc/editor.md, "Talking to other scripts", verbatim - the sibling
    // convention, and for the same reason: a doc that hands somebody a script
    // which does not run is worse than no doc.
    auto scene = Scene::create();
    auto door = attach(*scene, "Door", R"(
import threepp


class Door:
    open = False

    def start(self, obj: threepp.Object3D):
        self.obj = obj

    def on_opened(self):
        self.obj.position.y += 1.0
)");
    auto button = attach(*scene, "Button", R"(
import threepp


class Button:
    def start(self, obj: threepp.Object3D):
        # Resolve neighbours here, once: every instance exists by now.
        self.door = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Door"))

    def update(self, dt: float):
        if self.door is None or self.door.open:
            return
        if threepp.editor.is_key_down("SPACE"):
            self.door.on_opened()      # calling a method IS the signal
            self.door.open = True      # and so is setting an attribute
)");

    const float rest = door->position.y;

    ScriptPlaySession session;

    // Nobody is pressing anything: the example holds still rather than raising.
    session.start(*scene);
    for (int i = 0; i < 5; ++i) session.update(0.1f);
    CHECK(session.errorCount() == 0);
    CHECK(door->position.y == rest);
    session.stop();

    // The editor app installs this from its window; a headless pass installs it
    // by hand, which is what lets the example be driven exactly as written.
    scripting::keyStateProvider() = [](const std::string& key) { return key == "SPACE"; };

    session.start(*scene);
    for (int i = 0; i < 5; ++i) session.update(0.1f);

    CHECK(session.errorCount() == 0);
    // Held for five frames, opened once: `door.open`, written from the Button
    // onto the Door's own instance, is what makes the second press a no-op.
    CHECK(door->position.y == rest + 1.f);
    CHECK(button->position.y == 0.f);

    session.stop();
    scripting::keyStateProvider() = nullptr;
}
