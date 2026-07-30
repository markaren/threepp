// threepp.editor.raycast: the synchronous scene query a play script puts to the
// physics world it is running inside.
//
// Unlike the handles beside it this answers about the WHOLE world rather than
// about one authored body, so what is asserted here is mostly about the seams it
// crosses: which actors a query sees in this world's configuration, which object
// an actor answers as (including the two cases the registry does not hold
// directly — a subtree collider's many actors, and an articulation's links), what
// `ignore` has to exclude before a ground check stops hitting itself, and the two
// failures that must NOT come back as a miss.
//
// The sessions are driven exactly as the editor drives them (physics, sensors,
// scripts, in that order) — a query from fixed_update lands in a different place
// in the substep loop than one from update(), and both are meant to work.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // The default PhysxWorld timestep: a frame of exactly this length is exactly
    // one substep.
    constexpr float kFixed = 1.f / 60.f;

    // A unit box with a body of the requested kind.
    std::shared_ptr<Mesh> addBox(Scene& scene, const char* name, const Vector3& position,
                                 PhysicsConfig::Body body, const std::string& script = "") {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.copy(position);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = body;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 2.f;
        physics.restitution = 0.f;
        physics.write(*box);

        if (!script.empty()) {
            ScriptConfig config;
            config.source = script;
            config.write(*box);
        }

        scene.add(box);
        return box;
    }

    // A wide static slab whose top face is at y = 0.
    std::shared_ptr<Mesh> addGround(Scene& scene, const char* name = "Ground") {

        auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
        ground->name = name;
        ground->scale.set(20.f, 0.2f, 20.f);
        ground->position.set(0.f, -0.1f, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Static;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.write(*ground);

        scene.add(ground);
        return ground;
    }

    // The caster: a script and deliberately no physics of its own, so nothing it
    // asks about can be confused with a body it owns.
    std::shared_ptr<Mesh> addCaster(Scene& scene, const char* name, const std::string& script) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(8.f, 8.f, 8.f);// far away from everything cast at

        ScriptConfig config;
        config.source = script;
        config.write(*box);

        scene.add(box);
        return box;
    }

    // The editor's registration order, and its stop order (physics first).
    struct Rig {

        SceneDocument document;
        PhysicsPlaySession physics;
        PhysxSensorPlaySession sensors;
        ScriptPlaySession scripts;
        std::vector<std::string> log;

        Rig() {
            sensors.setPhysics(&physics);
            scripts.setLogger([this](const std::string& line) { log.push_back(line); });
        }

        Scene& scene() { return document.scene(); }

        void start(bool withPhysics = true) {
            if (withPhysics) {
                physics.start(scene());
                sensors.start(scene());
            }
            scripts.start(scene());
        }

        void run(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) frame(dt);
        }

        void frame(float dt) {
            physics.update(dt);
            sensors.update(dt);
            scripts.update(dt);
        }

        // Scripts only: no world was ever built, so nothing steps.
        void runScriptsOnly(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) scripts.update(dt);
        }

        void stop() {
            physics.stop();
            sensors.stop();
            scripts.stop();
        }

        Object3D* marker(const char* name) { return scene().getObjectByName(name); }

        // The first object whose name starts with `prefix` — how a script reports
        // a STRING back to this test (it renames a node of its own).
        std::string tagged(const std::string& prefix) {
            std::string found;
            scene().traverse([&](Object3D& object) {
                if (found.empty() && object.name.rfind(prefix, 0) == 0) found = object.name;
            });
            return found;
        }
    };

    // A script's only channel back into a test is the scene, so every script
    // below keeps four nodes of its own — two REPORTING SLOTS, because most of
    // these tests are about the difference between two casts:
    //
    //   "<tag>-out<i>"  position = hit point
    //                   scale    = (normal.y, distance, hits)
    //                   rotation = (misses, flag, 0)
    //   "<tag>-who<i>"  renamed to "<tag>-who<i>|<type>|<name>" on every hit
    //
    // `methods` is appended as further methods of the same class; it may use
    // self.obj, self.body, self.record(i, hit), self.flags[i] and self.publish(i).
    std::string probe(const std::string& tag, const std::string& methods) {

        std::string source = R"(
import threepp

class Probe:
    def start(self, obj):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)
        self.out = []
        self.who = []
        for i in (0, 1):
            o = threepp.Object3D()
            o.name = "TAG-out%d" % i
            o.scale.set(0.0, 0.0, 0.0)  # so an untouched slot reads as zero hits
            obj.parent.add(o)
            self.out.append(o)
            w = threepp.Object3D()
            w.name = "TAG-who%d" % i
            obj.parent.add(w)
            self.who.append(w)
        self.hits = [0, 0]
        self.misses = [0, 0]
        self.flags = [0.0, 0.0]

    def record(self, i, hit):
        # The one place a hit turns into numbers this test can read.
        if hit is None:
            self.misses[i] += 1
        else:
            self.hits[i] += 1
            self.out[i].position.set(hit.point.x, hit.point.y, hit.point.z)
            self.out[i].scale.set(hit.normal.y, hit.distance, float(self.hits[i]))
            if hit.object is None:
                self.who[i].name = "TAG-who%d|None|" % i
            else:
                self.who[i].name = "TAG-who%d|%s|%s" % (
                        i, type(hit.object).__name__, hit.object.name)
        self.publish(i)

    def publish(self, i):
        self.out[i].rotation.set(float(self.misses[i]), float(self.flags[i]), 0.0)
METHODS
)";
        // Substituted rather than formatted: the source is Python and full of
        // braces.
        for (std::string::size_type at = source.find("TAG"); at != std::string::npos;
             at = source.find("TAG", at)) {
            source.replace(at, 3, tag);
            at += tag.size();
        }
        source.replace(source.find("METHODS"), 7, methods);
        return source;
    }

    // A three-link arm, primitives only, so the articulation builder can cook it
    // with no external mesh file. Written to disk because both URDFLoader and the
    // articulation builder take a path.
    std::filesystem::path writeArmFixture() {

        const auto dir = std::filesystem::temp_directory_path() / "threepp-raycast-test";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm.urdf";
        std::ofstream(path, std::ios::trunc) << R"(
        <robot name="arm">
          <link name="base_link">
            <visual><geometry><box size="0.4 0.4 0.4"/></geometry></visual>
            <collision><geometry><box size="0.4 0.4 0.4"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.4 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.4 0.1"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 0.6" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
        </robot>)";
        return path;
    }

}// namespace


TEST_CASE("a ray straight down finds what is under it", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene());
    // A static unit box centred at y = 1.5, so its top face is at y = 2.
    addBox(rig.scene(), "Target", Vector3(0.f, 1.5f, 0.f), PhysicsConfig::Body::Static);
    // And a KINEMATIC one beside it: a kinematic actor is a PxRigidDynamic, so
    // the query's default eSTATIC|eDYNAMIC covers it, but that is worth pinning
    // rather than assuming — it is the one body type a script drives by hand.
    addBox(rig.scene(), "Mover", Vector3(4.f, 1.5f, 0.f), PhysicsConfig::Body::Kinematic);

    auto caster = addCaster(rig.scene(), "Caster", probe("down", R"(
    def update(self, dt):
        down = threepp.Vector3(0.0, -1.0, 0.0)
        self.record(0, threepp.editor.raycast(threepp.Vector3(0.0, 4.0, 0.0), down, 10.0))
        self.record(1, threepp.editor.raycast(threepp.Vector3(4.0, 4.0, 0.0), down, 10.0))
)"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(caster->uuid).empty());
    rig.run(5);
    CHECK(rig.scripts.errorFor(caster->uuid).empty());

    auto* out = rig.marker("down-out0");
    REQUIRE(out != nullptr);

    CHECK(static_cast<int>(out->scale.z) == 5);  // a hit on every one of the 5 frames
    CHECK(static_cast<int>(out->rotation.x) == 0);// and never a miss

    // The top face of the box, dead centre, 2 m below the origin of the ray.
    CHECK_THAT(out->position.y, WithinAbs(2.f, 1e-3));
    CHECK_THAT(out->position.x, WithinAbs(0.f, 1e-3));
    CHECK_THAT(out->position.z, WithinAbs(0.f, 1e-3));
    CHECK_THAT(out->scale.y, WithinAbs(2.f, 1e-3));// distance
    CHECK_THAT(out->scale.x, WithinAbs(1.f, 1e-3));// normal points up, out of the face

    // The object it names is the one the physics was authored on, as its
    // concrete type rather than as a bare Object3D.
    CHECK(rig.tagged("down-who0|") == "down-who0|Mesh|Target");

    // The kinematic body answers exactly the same way.
    auto* kinematic = rig.marker("down-out1");
    REQUIRE(kinematic != nullptr);
    CHECK(static_cast<int>(kinematic->scale.z) == 5);
    CHECK_THAT(kinematic->scale.y, WithinAbs(2.f, 1e-3));
    CHECK(rig.tagged("down-who1|") == "down-who1|Mesh|Mover");

    rig.stop();
}

TEST_CASE("a ray that hits nothing answers None", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene());
    addBox(rig.scene(), "Target", Vector3(0.f, 1.5f, 0.f), PhysicsConfig::Body::Static);

    // Horizontally, well above everything: 20 m of empty air.
    auto caster = addCaster(rig.scene(), "Caster", probe("miss", R"(
    def update(self, dt):
        self.record(0, threepp.editor.raycast(threepp.Vector3(0.0, 6.0, 0.0),
                                              threepp.Vector3(1.0, 0.0, 0.0), 20.0))
        # And straight down, where the target IS - but stopping short of it.
        self.record(1, threepp.editor.raycast(threepp.Vector3(0.0, 4.0, 0.0),
                                              threepp.Vector3(0.0, -1.0, 0.0), 1.5))
)"));

    rig.start();
    rig.run(4);
    CHECK(rig.scripts.errorFor(caster->uuid).empty());

    auto* empty = rig.marker("miss-out0");
    auto* tooShort = rig.marker("miss-out1");
    REQUIRE(empty != nullptr);
    REQUIRE(tooShort != nullptr);

    CHECK(static_cast<int>(empty->rotation.x) == 4);// four misses
    CHECK(static_cast<int>(empty->scale.z) == 0);   // and not one hit

    // max_distance is a real cutoff: the target's face is 2 m down and the ray
    // was allowed 1.5.
    CHECK(static_cast<int>(tooShort->rotation.x) == 4);
    CHECK(static_cast<int>(tooShort->scale.z) == 0);

    // Nothing was named, either way.
    CHECK(rig.tagged("miss-who0|").empty());
    CHECK(rig.tagged("miss-who1|").empty());

    rig.stop();
}

TEST_CASE("ignore excludes the caller's own body", "[editor][scripting][physx]") {

    // The first thing every user does: cast down from your own centre to find the
    // ground, from a point that is inside your own collider.
    Rig rig;
    addGround(rig.scene());
    auto self = addBox(rig.scene(), "Self", Vector3(0.f, 1.2f, 0.f), PhysicsConfig::Body::Dynamic,
                       probe("self", R"(
    def update(self, dt):
        if self.body is None:
            return
        down = threepp.Vector3(0.0, -1.0, 0.0)
        self.record(0, threepp.editor.raycast(self.body.position, down, 10.0))
        self.record(1, threepp.editor.raycast(self.body.position, down, 10.0, self.obj))
)"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(self->uuid).empty());
    rig.run(120);// falls 0.7 m and settles with its centre at y = 0.5
    CHECK(rig.scripts.errorFor(self->uuid).empty());

    auto* naive = rig.marker("self-out0");
    auto* ignoring = rig.marker("self-out1");
    REQUIRE(naive != nullptr);
    REQUIRE(ignoring != nullptr);

    // Without ignore the ray never leaves the body it started in.
    CHECK(rig.tagged("self-who0|") == "self-who0|Mesh|Self");
    CHECK_THAT(naive->scale.y, WithinAbs(0.f, 1e-3));// zero distance: it is already inside

    // With it, the ray passes through and lands on the ground: half a metre down
    // from the resting centre of a unit box, on a face pointing up.
    CHECK(rig.tagged("self-who1|") == "self-who1|Mesh|Ground");
    CHECK_THAT(ignoring->scale.y, WithinAbs(0.5f, 0.02));
    CHECK_THAT(ignoring->position.y, WithinAbs(0.f, 0.02));
    CHECK_THAT(ignoring->scale.x, WithinAbs(1.f, 1e-3));
    CHECK(static_cast<int>(ignoring->rotation.x) == 0);// and never missed

    rig.stop();
}

TEST_CASE("a hit on any part of a subtree collider names the authored root",
          "[editor][scripting][physx]") {

    // The imported-model case: physics authored on a GROUP, which cooks one actor
    // per sub-mesh. There is no "fourth cooked mesh" object for a script to be
    // handed — the thing the user put physics on is the group.
    Rig rig;
    addGround(rig.scene());

    auto model = Group::create();
    model->name = "Model";
    model->position.set(0.f, 2.f, 0.f);

    auto left = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), MeshBasicMaterial::create());
    left->name = "LeftPart";
    left->position.set(-1.2f, 0.f, 0.f);
    auto right = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), MeshBasicMaterial::create());
    right->name = "RightPart";
    right->position.set(1.2f, 0.f, 0.f);
    model->add(left);
    model->add(right);

    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Static;
    config.shape = PhysicsConfig::Shape::Auto;// a static Group collides as its subtree
    config.write(*model);
    rig.scene().add(model);

    auto caster = addCaster(rig.scene(), "Caster", probe("parts", R"(
    def update(self, dt):
        down = threepp.Vector3(0.0, -1.0, 0.0)
        self.record(0, threepp.editor.raycast(threepp.Vector3(-1.2, 5.0, 0.0), down, 10.0))
        self.record(1, threepp.editor.raycast(threepp.Vector3(1.2, 5.0, 0.0), down, 10.0))
)"));

    rig.start();
    rig.run(3);
    CHECK(rig.scripts.errorFor(caster->uuid).empty());

    auto* leftHit = rig.marker("parts-out0");
    auto* rightHit = rig.marker("parts-out1");
    REQUIRE(leftHit != nullptr);
    REQUIRE(rightHit != nullptr);

    // Two different actors, one answer: the group, as a Group.
    CHECK(rig.tagged("parts-who0|") == "parts-who0|Group|Model");
    CHECK(rig.tagged("parts-who1|") == "parts-who1|Group|Model");

    // And they really are the two separate parts — top face of each at y = 2.3.
    CHECK_THAT(leftHit->position.y, WithinAbs(2.3f, 1e-2));
    CHECK_THAT(leftHit->position.x, WithinAbs(-1.2f, 1e-2));
    CHECK_THAT(rightHit->position.x, WithinAbs(1.2f, 1e-2));
    CHECK(static_cast<int>(leftHit->scale.z) == 3);
    CHECK(static_cast<int>(rightHit->scale.z) == 3);

    rig.stop();
}

TEST_CASE("a hit on an articulation link names the robot", "[editor][scripting][physx]") {

    // A robot's links belong to the articulation, not to the session's actor
    // registry — but the session knows its articulations, so the link still
    // resolves, to the ROBOT the user authored rather than to a link node.
    const auto urdf = writeArmFixture();

    Rig rig;
    URDFLoader loader;
    auto robot = loader.load(urdf);
    robot->name = "Arm";

    RobotConfig rc;
    rc.urdf = urdf.string();
    rc.joints = {0.f};
    rc.write(*robot);
    ArticulationConfig ac;
    ac.enabled = true;
    ac.fixedBase = true;
    ac.write(*robot);
    rig.scene().add(robot);

    auto caster = addCaster(rig.scene(), "Caster", probe("robot", R"(
    def update(self, dt):
        self.record(0, threepp.editor.raycast(threepp.Vector3(0.0, 4.0, 0.0),
                                              threepp.Vector3(0.0, -1.0, 0.0), 10.0))
)"));

    rig.start();
    REQUIRE(rig.physics.articulationCount() == 1);
    rig.run(3);
    CHECK(rig.scripts.errorFor(caster->uuid).empty());

    auto* out = rig.marker("robot-out0");
    REQUIRE(out != nullptr);

    CHECK(static_cast<int>(out->scale.z) == 3);
    // The base link is a 0.4 box centred on the origin, so its top face is at
    // y = 0.2 — 3.8 m below the ray's origin.
    CHECK_THAT(out->scale.y, WithinAbs(3.8f, 1e-2));
    CHECK(rig.tagged("robot-who0|") == "robot-who0|Robot|Arm");

    rig.stop();
}

TEST_CASE("a raycast from fixed_update reads the substep it is standing in",
          "[editor][scripting][physx]") {

    // fixed_update runs from inside PhysxWorld::step(), between one substep's
    // fetchResults and the next simulate — where a scene query is legal. The
    // assertion is consistency, not accuracy: the ground's top face is y = 0, so
    // the drop below the body's own centre IS the height its own pose reports.
    Rig rig;
    addGround(rig.scene());
    auto faller = addBox(rig.scene(), "Faller", Vector3(0.f, 3.f, 0.f), PhysicsConfig::Body::Dynamic,
                         probe("sub", R"(
    def fixed_update(self, dt):
        if self.body is None:
            return
        p = self.body.position
        hit = threepp.editor.raycast(p, threepp.Vector3(0.0, -1.0, 0.0), 50.0, self.obj)
        if hit is None:
            self.misses[0] += 1
            self.publish(0)
            return
        err = abs(hit.distance - p.y)
        if err > self.flags[0]:
            self.flags[0] = err
        self.record(0, hit)
)"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(faller->uuid).empty());
    rig.run(200);
    CHECK(rig.scripts.errorFor(faller->uuid).empty());

    auto* out = rig.marker("sub-out0");
    REQUIRE(out != nullptr);

    // One call per substep, one hit each, never a miss on the way down.
    CHECK(static_cast<int>(out->scale.z) == 200);
    CHECK(static_cast<int>(out->rotation.x) == 0);
    // The worst disagreement over the whole fall between "how high the body says
    // it is" and "how far down the ground is".
    INFO("worst |distance - body.y| = " << out->rotation.y);
    CHECK(out->rotation.y < 0.01f);

    // And it ended up resting on the thing it was measuring.
    CHECK(rig.tagged("sub-who0|") == "sub-who0|Mesh|Ground");
    CHECK_THAT(out->scale.y, WithinAbs(0.5f, 0.02));

    rig.stop();
}

TEST_CASE("the two failures are raises, not misses", "[editor][scripting][physx]") {

    // A miss is None. Neither of these may also be None, or a script cannot tell
    // "nothing there" from "you asked the wrong question".
    const char* kCatcher = R"(
    def probe(self, direction):
        try:
            threepp.editor.raycast(threepp.Vector3(0.0, 4.0, 0.0), direction, 10.0)
            return 1.0    # answered
        except ValueError:
            return 2.0
        except RuntimeError:
            return 3.0
        except Exception:
            return 4.0

    def update(self, dt):
        self.flags[0] = self.probe(threepp.Vector3(0.0, -1.0, 0.0))
        self.flags[1] = self.probe(threepp.Vector3(0.0, 0.0, 0.0))
        self.publish(0)
        self.publish(1)
)";

    SECTION("no physics world playing") {

        Rig rig;
        addGround(rig.scene());
        auto caster = addCaster(rig.scene(), "Caster", probe("dead", kCatcher));

        rig.start(/*withPhysics=*/false);
        rig.runScriptsOnly(3);
        CHECK(rig.scripts.errorFor(caster->uuid).empty());

        auto* wellFormed = rig.marker("dead-out0");
        auto* zeroDir = rig.marker("dead-out1");
        REQUIRE(wellFormed != nullptr);
        REQUIRE(zeroDir != nullptr);

        // RuntimeError even for a perfectly good ray — there is no world to ask.
        CHECK_THAT(wellFormed->rotation.y, WithinAbs(3.f, 1e-6));
        // The world is looked for FIRST, so a malformed ray at a dead session
        // reports the session: it is the reason nothing could have been cast.
        CHECK_THAT(zeroDir->rotation.y, WithinAbs(3.f, 1e-6));

        rig.scripts.stop();
    }

    SECTION("a world, but a direction going nowhere") {

        Rig rig;
        addGround(rig.scene());
        auto caster = addCaster(rig.scene(), "Caster", probe("zero", kCatcher));

        rig.start();
        rig.run(3);
        CHECK(rig.scripts.errorFor(caster->uuid).empty());

        auto* good = rig.marker("zero-out0");
        auto* zeroDir = rig.marker("zero-out1");
        REQUIRE(good != nullptr);
        REQUIRE(zeroDir != nullptr);

        CHECK_THAT(good->rotation.y, WithinAbs(1.f, 1e-6));   // answered normally
        CHECK_THAT(zeroDir->rotation.y, WithinAbs(2.f, 1e-6));// ValueError

        rig.stop();
    }
}

// The ground-check script from doc/editor.md, verbatim. If this stops working
// the documentation is handing users a script that does not run.
TEST_CASE("the documented ground check fires only when grounded",
          "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene());
    auto hopper = addBox(rig.scene(), "Hopper", Vector3(0.f, 1.2f, 0.f),
                         PhysicsConfig::Body::Dynamic, R"(
import threepp


class Hopper:
    probe = 0.6              # metres: just past the bottom of a unit box

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)

    def fixed_update(self, dt: float):
        if self.body is None:
            return
        down = threepp.Vector3(0, -1, 0)
        hit = threepp.editor.raycast(self.body.position, down, self.probe, ignore=self.obj)
        if hit is not None and self.body.velocity.y <= 0.0:
            self.body.apply_impulse(threepp.Vector3(0, 4.0 * self.body.mass, 0))
)");

    rig.start();
    REQUIRE(rig.scripts.errorFor(hopper->uuid).empty());

    // Measured from the first time the probe could possibly have reached the
    // ground, so the drop it spawns into is not mistaken for a hop. (The probe
    // is a touch longer than the box's half-height, so "grounded" begins a few
    // centimetres before contact — which is what a ground check is for.)
    bool landed = false;
    float peak = 0.f;
    float lowestAfterPeak = 100.f;
    for (int i = 0; i < 400; ++i) {
        rig.frame(kFixed);
        if (hopper->position.y < 0.65f) landed = true;
        if (!landed) continue;
        peak = std::max(peak, hopper->position.y);
        if (peak > 1.f) lowestAfterPeak = std::min(lowestAfterPeak, hopper->position.y);
    }
    CHECK(rig.scripts.errorFor(hopper->uuid).empty());
    REQUIRE(landed);

    // It left the ground, which means the probe found it — and it came back
    // down, which means the probe did NOT keep finding it in mid-air. A ray that
    // hit the caller's own collider would have fired every substep and shot it
    // out of the scene.
    INFO("peak " << peak << ", lowest after " << lowestAfterPeak);
    CHECK(peak > 1.f);
    CHECK(peak < 3.f);
    CHECK(lowestAfterPeak < 0.75f);

    rig.stop();
}
