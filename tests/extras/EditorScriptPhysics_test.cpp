// threepp.editor's physics handles, driven exactly as the editor drives them:
// the physics session and the script session are independent PlaySessions,
// stepped in that order, and a script reaches the body through a free function
// with no context but "the session that is playing".

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iostream>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::shared_ptr<Mesh> addGround(Scene& scene) {

        auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
        ground->name = "Ground";
        ground->scale.set(40.f, 0.2f, 40.f);
        ground->position.set(0.f, -0.1f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Static;
        config.shape = PhysicsConfig::Shape::Box;
        config.write(*ground);
        scene.add(ground);
        return ground;
    }

    std::shared_ptr<Mesh> addBox(Scene& scene, const std::string& name,
                                 PhysicsConfig::Body body, const std::string& script = "") {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(0.f, 2.f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = body;
        config.mass = 2.f;
        config.write(*box);

        if (!script.empty()) {
            ScriptConfig scriptConfig;
            scriptConfig.source = script;
            scriptConfig.write(*box);
        }
        scene.add(box);
        return box;
    }

    // The editor's own order: physics first, scripts last, so a script's reads
    // see the step that just happened and its forces land on the next one.
    void run(PhysicsPlaySession& physics, ScriptPlaySession& scripts, int steps) {

        for (int i = 0; i < steps; ++i) {
            physics.update(1.f / 60.f);
            scripts.update(1.f / 60.f);
        }
    }

}// namespace


TEST_CASE("a script drives the body physics is simulating", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // Thrust that beats gravity: 2 kg under 9.81 needs ~19.6 N to hover, so
    // 60 N lifts. If apply_force did not reach PhysX the box would just fall.
    auto box = addBox(scene, "Thruster", PhysicsConfig::Body::Dynamic, R"(
import threepp

class Thruster:
    def start(self, obj):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)
        self.seen_static = self.body.is_static if self.body else None
        self.mass = self.body.mass if self.body else 0.0

    def update(self, dt):
        if self.body:
            self.body.apply_force(threepp.Vector3(0, 60, 0))
            self.speed = self.body.velocity.y
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;
    std::string error;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());

    run(physics, scripts, 90);

    CHECK(box->position.y > 3.f);// it climbed rather than fell
    scripts.stop();
    physics.stop();
}

TEST_CASE("the handle reflects what was authored", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    auto box = addBox(scene, "Probe", PhysicsConfig::Body::Dynamic, R"(
import threepp

class Probe:
    def start(self, obj):
        body = threepp.editor.rigid_body_from_object(obj)
        obj.name = "static" if body.is_static else "dynamic-%d" % round(body.mass)
        # A child of a physics object resolves to the body governing it.
        child = threepp.Object3D()
        obj.add(child)
        obj.name += "/child" if threepp.editor.rigid_body_from_object(child) else "/none"

    def update(self, dt):
        pass
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());

    // Mass is the authored 2 kg, not a density-derived number, and the lookup
    // walked up from the child.
    CHECK(box->name == "dynamic-2/child");

    scripts.stop();
    physics.stop();
}

TEST_CASE("a body is None outside Play and dead after it", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // Records what it saw at each phase into the object's name, which is the
    // only channel a script has back into the test.
    auto box = addBox(scene, "Ledger", PhysicsConfig::Body::Dynamic, R"(
import threepp

class Ledger:
    def start(self, obj):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)
        obj.name = "got" if self.body else "none"

    def update(self, dt):
        pass

    def stop(self):
        # Physics has already stopped by now: the handle must say so rather
        # than read a released actor.
        self.obj.name += "/valid" if self.body.valid else "/dead"
        try:
            self.body.velocity
            self.obj.name += "/read"
        except RuntimeError:
            self.obj.name += "/raised"
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    // Scripts alone: no play session means no body.
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());
    CHECK(box->name == "none");
    scripts.stop();

    box->name = "Ledger";
    physics.start(scene);
    scripts.start(scene);
    CHECK(box->name == "got");

    run(physics, scripts, 10);

    // Physics DELIBERATELY stopped first. This is no longer the controller's
    // order — PlayController stops sessions in reverse registration order, so a
    // script's stop() normally sees a live world (the test below pins that) —
    // but the world dying under a held handle is still reachable: the editor
    // torn down mid-Play runs destructors in member order, and a handle stashed
    // beyond its session outlives any ordering promise. THIS is what the
    // lifetime token is for, so the test produces the hostile order by hand and
    // asserts a dead handle answers rather than reads freed memory.
    physics.stop();
    scripts.stop();
    CHECK(box->name == "got/dead/raised");
}

TEST_CASE("a script's stop() runs against a live world", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    // The controller's order — scripts stop FIRST, reverse of registration —
    // driven here by hand the same way `run` mirrors its update order. stop()
    // is where cleanup code naturally goes (park the robot, log the pose,
    // release the grip), so everything a script could hold is exercised there:
    // the checked handle, editor.world, and the raw handle world.add returned —
    // the last of which was a SEGFAULT in this window before sessions stopped
    // in reverse.
    auto box = addBox(scene, "Closer", PhysicsConfig::Body::Dynamic, R"(
import threepp

editor = threepp.editor


class Closer:
    def start(self, obj):
        self.obj = obj
        self.body = editor.rigid_body_from_object(obj)
        crate = threepp.Mesh(threepp.BoxGeometry(0.5, 0.5, 0.5),
                             threepp.MeshStandardMaterial())
        crate.position.set(3.0, 3.0, 0.0)
        editor.scene().add(crate)
        self.raw = editor.world().add(crate)
        obj.name = "got" if self.body else "none"

    def stop(self):
        self.obj.name += "/valid" if self.body.valid else "/dead"
        try:
            self.body.velocity
            self.obj.name += "/read"
        except RuntimeError:
            self.obj.name += "/raised"
        self.obj.name += "/world" if editor.world() is not None else "/noworld"
        p = self.raw.position
        self.obj.name += "/raw" if p.y < 3.0 else "/unmoved"
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());
    CHECK(box->name == "got");

    run(physics, scripts, 60);

    scripts.stop();
    physics.stop();

    CHECK(scripts.errorFor(box->uuid).empty());
    // Every read answered, and the raw handle reported a crate that actually
    // fell — a live world on both counts.
    CHECK(box->name == "got/valid/read/world/raw");
}

TEST_CASE("a kinematic body is steered through its target", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    auto box = addBox(scene, "Platform", PhysicsConfig::Body::Kinematic, R"(
import threepp

class Platform:
    def start(self, obj):
        self.body = threepp.editor.rigid_body_from_object(obj)
        self.t = 0.0

    def update(self, dt):
        self.t += dt
        # Exercise BOTH forms: rotation defaulted, and rotation passed.
        if int(self.t * 60) % 2 == 0:
            self.body.set_kinematic_target(threepp.Vector3(self.t * 2.0, 2.0, 0))
        else:
            self.body.set_kinematic_target(threepp.Vector3(self.t * 2.0, 2.0, 0),
                                           threepp.Quaternion(0, 0, 0, 1))
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());

    run(physics, scripts, 60);

    // Swept to roughly t*2 along x, and held at its authored height.
    CHECK(box->position.x > 1.4f);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(box->position.y, WithinAbs(2.f, 0.05));

    scripts.stop();
    physics.stop();
}

TEST_CASE("a static body reports rather than answering nonsense", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto ground = addGround(scene);
    ScriptConfig config;
    config.source = R"(
import threepp

class Poke:
    def start(self, obj):
        body = threepp.editor.rigid_body_from_object(obj)
        obj.name = "static" if body.is_static else "dynamic"
        try:
            body.apply_force(threepp.Vector3(0, 1, 0))
            obj.name += "/pushed"
        except RuntimeError:
            obj.name += "/refused"

    def update(self, dt):
        pass
)";
    config.write(*ground);

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(ground->uuid).empty());

    CHECK(ground->name == "static/refused");

    scripts.stop();
    physics.stop();
}

// The Hover script from doc/editor.md, verbatim. If this stops working the
// documentation is handing users a script that does not run.
TEST_CASE("the documented hover script holds its height", "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    auto box = addBox(scene, "Hover", PhysicsConfig::Body::Dynamic, R"(
import threepp


class Hover:
    height = 3.0
    stiffness = 40.0

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)

    def update(self, dt: float):
        if self.body is None:
            return
        # A spring toward `height`, damped by the body's own velocity.
        error = self.height - self.body.position.y
        lift = self.stiffness * error - 5.0 * self.body.velocity.y
        self.body.apply_force(threepp.Vector3(0, lift * self.body.mass, 0))
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    scripts.start(scene);
    REQUIRE(scripts.errorFor(box->uuid).empty());

    run(physics, scripts, 400);

    // Settled at the authored height, from a 2 m start, under gravity.
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(box->position.y, WithinAbs(3.f, 0.3));

    scripts.stop();
    physics.stop();
}

TEST_CASE("a soft body reports where it deformed to", "[editor][scripting][physx][gpu]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);

    auto ball = ObjectFactory::createPrimitive(Primitive::Sphere, scene);
    ball->name = "Jelly";
    ball->position.set(0.f, 3.f, 0.f);

    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Soft;
    config.mass = 2.f;
    config.youngsModulus = 5e4f;
    config.voxelResolution = 6;
    config.write(*ball);

    ScriptConfig script;
    // The object's own transform is zeroed for the whole of Play, so `center`
    // is the only way a script can follow a soft body.
    script.source = R"(
import threepp

class Follow:
    def start(self, obj):
        self.body = threepp.editor.soft_body_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Marker"
        obj.parent.add(self.marker)
        self.verts = self.body.vertex_count if self.body else -1

    def update(self, dt):
        if self.body:
            self.marker.position.copy(self.body.center)
)";
    script.write(*ball);
    scene.add(ball);

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(scene);
    if (!physics.gpuAvailable()) {
        std::cout << "[skip] no CUDA device - soft body handle not exercised" << std::endl;
        physics.stop();
        return;
    }
    scripts.start(scene);
    REQUIRE(scripts.errorFor(ball->uuid).empty());

    run(physics, scripts, 150);

    auto* marker = scene.getObjectByName("Marker");
    REQUIRE(marker);
    // The marker followed the jelly down to the ground, while the object it is
    // attached to never moved at all.
    CHECK(marker->position.y < 1.f);
    CHECK(marker->position.y > 0.f);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(ball->position.y, WithinAbs(0.f, 1e-5));

    scripts.stop();
    physics.stop();
}

TEST_CASE("a script gives a spawned mesh a body through editor.world()",
          "[editor][scripting][physx]") {

    // The editor used to withhold the general physics bindings entirely, so the
    // only bodies a script could touch were the ones the DOCUMENT authored.
    // Between them, scene().add() (plain scene-graph parenting, which always
    // worked) and editor.world() (the session's own PhysxWorld) make a spawned
    // object a first-class one: it renders because the graph has it, and it
    // falls because the world does.
    //
    // Deliberately written the way a standalone threepp program is written -
    // that is the claim being tested. No editor-shaped spawn verb exists, and
    // none is needed.
    auto scene = Scene::create();
    addGround(*scene);

    // The findings go into the spawner's SCALE: physics never writes a scale
    // back, and this one is a static body whose pose is not bound either, so
    // nothing but the script can have touched them by the time they are read.
    auto spawner = addBox(*scene, "Spawner", PhysicsConfig::Body::Static, R"(
import threepp


class Spawner:
    def start(self, obj):
        self.obj = obj
        self.done = False
        # 2.0 for yes, -1.0 for no, and NEVER 1.0 - that is the scale a box is
        # created with, so a start() that never ran must not read as a pass.
        # The type is REACHABLE - it used to be a NameError...
        obj.scale.x = 2.0 if hasattr(threepp, "PhysxWorld") else -1.0
        # ...but a second world is still refused, which is the whole reason it
        # was withheld in the first place. The explaining message is the feature.
        try:
            threepp.PhysxWorld()
            obj.scale.y = -1.0
        except RuntimeError as e:
            # No parentheses in the needle. This source sits in a C++ raw string
            # literal, and a close-paren followed by a quote would end it early.
            obj.scale.y = 2.0 if "editor.world" in str(e) else -2.0
        # And the running world is reachable, which is what makes the refusal
        # above a redirection rather than a dead end.
        obj.scale.z = 2.0 if threepp.editor.world() is not None else -1.0

    def update(self, dt):
        if self.done:
            return
        self.done = True
        box = threepp.Mesh(threepp.BoxGeometry(0.6, 0.6, 0.6),
                           threepp.MeshStandardMaterial())
        box.name = "Spawned"
        # Clear of the spawner's own body at x=0, so what it lands on is the
        # ground and the resting height means what the assertion says it does.
        box.position.set(3.0, 5.0, 0.0)
        threepp.editor.scene().add(box)
        # The ordinary call, on the world the session is already stepping.
        threepp.editor.world().add(box)
)");

    PhysicsPlaySession physics;
    ScriptPlaySession scripts;

    physics.start(*scene);
    scripts.start(*scene);
    REQUIRE(scripts.errorFor(spawner->uuid).empty());

    using Catch::Matchers::WithinAbs;
    // What start() found, before a single step: the type is there, constructing
    // one raises with the message that redirects, and the running world answers.
    CHECK_THAT(spawner->scale.x, WithinAbs(2.f, 1e-5));
    CHECK_THAT(spawner->scale.y, WithinAbs(2.f, 1e-5));
    CHECK_THAT(spawner->scale.z, WithinAbs(2.f, 1e-5));

    run(physics, scripts, 180);
    CHECK(scripts.errorFor(spawner->uuid).empty());

    auto* spawned = scene->getObjectByName("Spawned");
    REQUIRE(spawned != nullptr);
    // Dropped from 5 m and came to rest ON THE GROUND rather than hanging where
    // it was spawned: the body is real, not a mesh parked in mid-air. Half the
    // 0.6 box sits above a ground whose top is y = 0.
    CHECK_THAT(spawned->position.y, WithinAbs(0.3f, 0.05f));
    // And it fell straight down - it did not land on the spawner.
    CHECK_THAT(spawned->position.x, WithinAbs(3.f, 0.05f));

    scripts.stop();
    physics.stop();
}
