// on_collision_enter / on_collision_exit: the two optional script methods fired
// when the body governing a script's object starts and stops touching another.
//
// The sessions are driven exactly as the editor drives them (physics, sensors,
// scripts, in that order), because the whole feature lives in the gap between
// those calls: PhysX reports a contact from inside physics.update(), and the
// script hears about it from the script session's own sweep afterwards. What is
// asserted here is what survives that gap — the edges (both of them, even when a
// touch begins and ends inside one frame), their order, what `contact` carries,
// who `other` is, and what happens when there is no body to watch.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // The default PhysxWorld timestep. A frame of exactly this length is exactly
    // one substep, which is what makes "one delivery window" a countable thing.
    constexpr float kFixed = 1.f / 60.f;

    // A dynamic body, dropped from `y`. A unit box and a half-metre sphere both
    // come to rest with their centre at y = 0.5 on the ground below.
    std::shared_ptr<Mesh> addBody(Scene& scene, const char* name, const std::string& script,
                                  float y = 1.2f, float restitution = 0.05f,
                                  Primitive shape = Primitive::Box) {

        auto body = ObjectFactory::createPrimitive(shape, scene);
        body->name = name;
        body->position.set(0.f, y, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Auto;
        physics.mass = 2.f;
        physics.restitution = restitution;
        physics.write(*body);

        if (!script.empty()) {
            ScriptConfig config;
            config.source = script;
            config.write(*body);
        }

        scene.add(body);
        return body;
    }

    // A wide static slab whose top face is at y = 0.
    std::shared_ptr<Mesh> addGround(Scene& scene, const char* name, const std::string& script = "",
                                    float restitution = 0.05f) {

        auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
        ground->name = name;
        ground->scale.set(20.f, 0.2f, 20.f);
        ground->position.set(0.f, -0.1f, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Static;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.restitution = restitution;
        physics.write(*ground);

        if (!script.empty()) {
            ScriptConfig config;
            config.source = script;
            config.write(*ground);
        }

        scene.add(ground);
        return ground;
    }

    // An object with a script and deliberately NO physics.
    std::shared_ptr<Mesh> addWeightless(Scene& scene, const char* name, const std::string& script) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(5.f, 5.f, 0.f);

        ScriptConfig config;
        config.source = script;
        config.write(*box);

        scene.add(box);
        return box;
    }

    // A contact sensor on the same body, sampling every substep.
    void instrument(Object3D& object) {

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Contact;
        sensor.rateHz = 0.f;
        sensor.write(object);
    }

    // The editor's registration order, and its stop order — the REVERSE of
    // registration, scripts first, so the script session gives its contact
    // watches back to a world that is still alive to take them.
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
            scripts.stop();
            sensors.stop();
            physics.stop();
        }

        [[nodiscard]] std::size_t linesContaining(const std::string& text) const {
            std::size_t n = 0;
            for (const auto& line : log) {
                if (line.find(text) != std::string::npos) ++n;
            }
            return n;
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

    // The ledger every test below reads. A script's only channel back into a test
    // is the scene, so it keeps two nodes of its own:
    //
    //   "<tag>-count"  position = (enters, exits, out-of-order exits)
    //                  scale    = (last normal.y, last |impulse|, windows that
    //                              carried BOTH an enter and an exit)
    //                  rotation = the last contact point
    //   "<tag>-who"    renamed to "<tag>-who|<type>|<name>" on every enter
    //
    // `extra` is appended as further methods of the same class.
    std::string ledger(const std::string& tag, const std::string& extra = "") {

        std::string source = R"(
import threepp

class Watcher:
    def start(self, obj):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)
        self.count = threepp.Object3D()
        self.count.name = "TAG-count"
        obj.parent.add(self.count)
        self.who = threepp.Object3D()
        self.who.name = "TAG-who"
        obj.parent.add(self.who)
        self.enters = 0
        self.exits = 0
        self.bad = 0
        self.open = 0
        self.both = 0
        self.seen_enters = 0
        self.seen_exits = 0
        self.ny = 0.0
        self.imp = 0.0
        self.publish()

    def on_collision_enter(self, contact):
        self.enters += 1
        self.open += 1
        self.ny = contact.normal.y
        i = contact.impulse
        self.imp = (i.x * i.x + i.y * i.y + i.z * i.z) ** 0.5
        self.count.rotation.set(contact.point.x, contact.point.y, contact.point.z)
        other = contact.other
        if other is None:
            self.who.name = "TAG-who|None|"
        else:
            self.who.name = "TAG-who|" + type(other).__name__ + "|" + other.name
        self.publish()

    def on_collision_exit(self, contact):
        self.exits += 1
        # An exit with nothing open is an edge out of order, or one whose enter
        # was collapsed away.
        if self.open <= 0:
            self.bad += 1
        else:
            self.open -= 1
        self.publish()

    def update(self, dt):
        # Callbacks are delivered before this, so the deltas since the previous
        # frame are exactly what THIS delivery window carried.
        de = self.enters - self.seen_enters
        dx = self.exits - self.seen_exits
        if de >= 1 and dx >= 1:
            self.both += 1
        self.seen_enters = self.enters
        self.seen_exits = self.exits
        self.publish()

    def publish(self):
        self.count.position.set(float(self.enters), float(self.exits), float(self.bad))
        self.count.scale.set(self.ny, self.imp, float(self.both))
EXTRA
)";
        // Substituted rather than formatted: the source is Python and full of
        // braces.
        for (std::string::size_type at = source.find("TAG"); at != std::string::npos;
             at = source.find("TAG", at)) {
            source.replace(at, 3, tag);
            at += tag.size();
        }
        source.replace(source.find("EXTRA"), 5, extra);
        return source;
    }

}// namespace


TEST_CASE("a landing body reports exactly one enter", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground");
    auto box = addBody(rig.scene(), "Faller", ledger("drop"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());

    // Long enough to fall 0.7 m, land, and settle. Settling is the point: PhysX
    // re-reports a resting manifold every substep until the pair sleeps, and
    // every one of those is a TOUCH_PERSISTS that must NOT read as a new touch.
    rig.run(200);
    CHECK(rig.scripts.errorFor(box->uuid).empty());

    auto* count = rig.marker("drop-count");
    REQUIRE(count != nullptr);

    CHECK(static_cast<int>(count->position.x) == 1);// one enter
    CHECK(static_cast<int>(count->position.y) == 0);// still resting on it
    CHECK(static_cast<int>(count->position.z) == 0);// nothing out of order

    // The ground is pushing UP on the box, whichever way round PhysX happened to
    // put the pair, and it took a real impulse to stop 2 kg falling.
    CHECK(count->scale.x > 0.9f);
    CHECK(count->scale.y > 0.f);
    // The contact point is on the ground's surface, under the box.
    CHECK_THAT(count->rotation.y, WithinAbs(0.f, 0.1));
    CHECK(std::abs(count->rotation.x) < 1.f);
    CHECK(std::abs(count->rotation.z) < 1.f);

    rig.stop();
}

TEST_CASE("a body that leaves reports the exit", "[editor][scripting][physx]") {

    // The push comes from the script itself, so the whole loop — report,
    // delivery, callback, physics call — runs through the paths this feature
    // adds rather than around them.
    Rig rig;
    addGround(rig.scene(), "Ground");
    auto jumper = addBody(rig.scene(), "Jumper", ledger("jump", R"(
    def fixed_update(self, dt):
        if self.body is None:
            return
        if self.enters >= 1 and self.exits == 0 and self.body.position.y < 0.6:
            self.body.apply_impulse(threepp.Vector3(0.0, 40.0, 0.0))
)"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(jumper->uuid).empty());
    rig.run(120);
    CHECK(rig.scripts.errorFor(jumper->uuid).empty());

    auto* count = rig.marker("jump-count");
    REQUIRE(count != nullptr);
    CHECK(static_cast<int>(count->position.x) == 1);// landed once
    CHECK(static_cast<int>(count->position.y) == 1);// and left again
    CHECK(static_cast<int>(count->position.z) == 0);// enter before exit, always
    CHECK(jumper->position.y > 2.f);               // genuinely airborne

    rig.stop();
}

TEST_CASE("a touch that begins and ends inside one frame still delivers both edges",
          "[editor][scripting][physx]") {

    // The rule the queue exists for. Contacts are reported from inside the solver
    // and delivered a sweep later; a bouncing ball at four substeps per frame
    // regularly touches and leaves between two deliveries. A state flag ("am I
    // touching?") sampled per frame would see no change at all and report
    // NOTHING. The queue keeps both edges, in order.
    Rig rig;
    addGround(rig.scene(), "Ground", "", /*restitution=*/0.95f);
    auto ball = addBody(rig.scene(), "Bouncer", ledger("bounce"), /*y=*/2.f,
                        /*restitution=*/0.95f, Primitive::Sphere);

    rig.start();
    REQUIRE(rig.scripts.errorFor(ball->uuid).empty());

    // Four substeps per frame — PhysxWorld's maxSubSteps, so nothing is dropped
    // and the delivery window is as wide as it ever gets.
    rig.run(150, 4.f * kFixed);
    CHECK(rig.scripts.errorFor(ball->uuid).empty());

    auto* count = rig.marker("bounce-count");
    REQUIRE(count != nullptr);

    const int enters = static_cast<int>(count->position.x);
    const int exits = static_cast<int>(count->position.y);

    CHECK(enters >= 3);        // it bounced
    CHECK(exits >= 2);         // and left again
    CHECK(enters - exits <= 1);// paired up, at most one still open
    // Never an exit with nothing open: within a window the enter always came
    // first, which is the ordering half of the rule.
    CHECK(static_cast<int>(count->position.z) == 0);
    // And at least one whole touch lived and died between two deliveries.
    CHECK(static_cast<int>(count->scale.z) >= 1);

    rig.stop();
}

TEST_CASE("both sides of a collision get their own callback", "[editor][scripting][physx]") {

    Rig rig;
    auto ground = addGround(rig.scene(), "Floor", ledger("floor"));
    auto box = addBody(rig.scene(), "Crate", ledger("crate"));

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());
    REQUIRE(rig.scripts.errorFor(ground->uuid).empty());
    REQUIRE(rig.scripts.instanceCount() == 2);

    rig.run(120);
    CHECK(rig.scripts.errorFor(box->uuid).empty());
    CHECK(rig.scripts.errorFor(ground->uuid).empty());

    auto* crate = rig.marker("crate-count");
    auto* floor = rig.marker("floor-count");
    REQUIRE(crate != nullptr);
    REQUIRE(floor != nullptr);

    // One touch, two scripts, one enter each — including the STATIC body, which
    // has no pose to write back and would be invisible to PhysxWorld's own
    // binding list.
    CHECK(static_cast<int>(crate->position.x) == 1);
    CHECK(static_cast<int>(floor->position.x) == 1);

    // Each names the other, as its concrete type rather than as Object3D.
    CHECK(rig.tagged("crate-who|") == "crate-who|Mesh|Floor");
    CHECK(rig.tagged("floor-who|") == "floor-who|Mesh|Crate");

    // The normals are opposite: each is oriented into the body whose script is
    // reading it, whichever side of the pair PhysX put it on.
    CHECK(crate->scale.x > 0.9f);
    CHECK(floor->scale.x < -0.9f);

    rig.stop();
}

TEST_CASE("a raising on_collision_enter is reported once and disabled",
          "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground");
    auto bad = addBody(rig.scene(), "Bad", R"(
import threepp

class Bad:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "BadMarker"
        obj.parent.add(self.marker)

    def on_collision_enter(self, contact):
        self.marker.position.x += 1.0
        raise ValueError("no")

    def on_collision_exit(self, contact):
        self.marker.position.z += 1.0

    def update(self, dt):
        self.marker.position.y += 1.0
)");
    auto good = addBody(rig.scene(), "Good", ledger("good"));
    good->position.set(3.f, 1.2f, 0.f);

    rig.start();
    REQUIRE(rig.scripts.errorFor(bad->uuid).empty());
    rig.run(120);

    auto* marker = rig.marker("BadMarker");
    REQUIRE(marker != nullptr);
    // It ran once and was then disabled WHOLE: no second enter, no exit, and no
    // more update() either — the instance is out for the session, not the
    // method.
    CHECK_THAT(marker->position.x, WithinAbs(1.f, 1e-6));
    CHECK_THAT(marker->position.z, WithinAbs(0.f, 1e-6));
    CHECK(marker->position.y > 0.f);
    CHECK(marker->position.y < 120.f);

    // One report, not one per contact, and it names the method.
    CHECK(rig.linesContaining("script error") == 1);
    CHECK(rig.linesContaining("(on_collision_enter)") == 1);
    CHECK(!rig.scripts.errorFor(bad->uuid).empty());

    // The other script on the other body is untouched by any of it.
    CHECK(rig.scripts.errorFor(good->uuid).empty());
    auto* count = rig.marker("good-count");
    REQUIRE(count != nullptr);
    CHECK(static_cast<int>(count->position.x) == 1);

    rig.stop();
}

TEST_CASE("collision callbacks on an object with no body say so once",
          "[editor][scripting][physx]") {

    SECTION("a physics world, but nothing physical on this object") {

        Rig rig;
        addGround(rig.scene(), "Ground");
        auto orphan = addWeightless(rig.scene(), "Orphan", ledger("orphan"));

        rig.start();
        REQUIRE(rig.scripts.errorFor(orphan->uuid).empty());
        rig.run(120);

        auto* count = rig.marker("orphan-count");
        REQUIRE(count != nullptr);
        CHECK(static_cast<int>(count->position.x) == 0);
        CHECK(static_cast<int>(count->position.y) == 0);
        // One line, naming the object, at Play — not a method that silently never
        // runs, and not one line per frame.
        CHECK(rig.linesContaining("collision callbacks need a physics body") == 1);
        CHECK(rig.linesContaining("Orphan") == 1);
        // Everything else about the script still works.
        CHECK(rig.scripts.errorFor(orphan->uuid).empty());

        rig.stop();
    }

    SECTION("no physics session at all") {

        Rig rig;
        auto orphan = addWeightless(rig.scene(), "Orphan", ledger("orphan"));

        rig.start(/*withPhysics=*/false);
        rig.runScriptsOnly(30);

        CHECK(rig.linesContaining("collision callbacks need a playing physics world") == 1);
        auto* count = rig.marker("orphan-count");
        REQUIRE(count != nullptr);
        CHECK(static_cast<int>(count->position.x) == 0);

        rig.scripts.stop();
    }
}

TEST_CASE("a ContactSensor on the same body keeps measuring",
          "[editor][scripting][sensors][physx]") {

    // Contact reporting is one opt-in bit per actor, shared by every watcher of
    // that body. Both consumers have to come out of one Play alive: the script
    // gets its edges, and the sensor's readings keep arriving for the panel and
    // the CSV.
    Rig rig;
    addGround(rig.scene(), "Ground");
    auto box = addBody(rig.scene(), "Instrumented", ledger("both"));
    instrument(*box);

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());
    REQUIRE(rig.sensors.liveCount() == 1);
    REQUIRE(rig.sensors.entries().size() == 1);
    const auto& entry = *rig.sensors.entries().front();
    REQUIRE(entry.contact != nullptr);

    // Peak force rather than the final value: PhysX stops re-reporting a resting
    // manifold once the pair sleeps, so the force channel is meant to go quiet
    // while the latch stays.
    float peakForce = 0.f;
    for (int i = 0; i < 150; ++i) {
        rig.frame(kFixed);
        peakForce = std::max(peakForce, entry.contactForce);
    }

    auto* count = rig.marker("both-count");
    REQUIRE(count != nullptr);
    CHECK(static_cast<int>(count->position.x) == 1);// the script got its edge

    const auto samples = entry.samples;
    CHECK(samples > 0);                       // measurements drained
    CHECK(entry.contactReadings.total() > 0); // and retained for readers
    CHECK(entry.inContact);                   // the latch agrees with the script
    CHECK(peakForce > 0.f);

    // Still advancing, with the script watching the same actor.
    rig.run(30);
    CHECK(entry.samples > samples);
    CHECK(static_cast<int>(count->position.x) == 1);

    rig.stop();
}

TEST_CASE("play, stop, play again re-registers cleanly", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground");
    auto box = addBody(rig.scene(), "Replay", ledger("replay"));

    rig.start();
    rig.run(120);

    auto* first = rig.marker("replay-count");
    REQUIRE(first != nullptr);
    CHECK(static_cast<int>(first->position.x) == 1);

    // Physics stops FIRST, so the world (and every contact watch in it) is gone
    // before the script session is asked to give its registrations back.
    rig.stop();

    // The box is left resting where it landed, so the second session's very first
    // substep finds the touch already there: exactly one enter again, never two,
    // and never one inherited from the run before.
    rig.start();
    rig.run(120);
    CHECK(rig.scripts.errorFor(box->uuid).empty());

    // The first run's markers are still in the scene (the editor's snapshot
    // restore is what removes them, and there is none here), so count instead of
    // looking one up by name.
    int counted = 0;
    float latestEnters = -1.f;
    float latestExits = -1.f;
    rig.scene().traverse([&](Object3D& object) {
        if (object.name == "replay-count") {
            ++counted;
            latestEnters = object.position.x;
            latestExits = object.position.y;
        }
    });
    CHECK(counted == 2);
    CHECK(static_cast<int>(latestEnters) == 1);
    CHECK(static_cast<int>(latestExits) == 0);
    // And the first run's ledger did not gain a thing from the second.
    CHECK(static_cast<int>(first->position.x) == 1);

    rig.stop();
}

// The Impact script from doc/editor.md, verbatim. If this stops working the
// documentation is handing users a script that does not run.
TEST_CASE("the documented collision script recolours on impact", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground", "", /*restitution=*/0.9f);
    auto ball = addBody(rig.scene(), "Impact", R"(
import threepp


class Impact:
    threshold = 2.0          # N*s; below this it is a nudge, not a hit

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.material = obj.material
        self.rest = self.material.color.get_hex()

    def on_collision_enter(self, contact: threepp.editor.Collision):
        i = contact.impulse
        strength = (i.x * i.x + i.y * i.y + i.z * i.z) ** 0.5
        if strength >= self.threshold:
            print(self.obj.name, "hit", contact.other.name if contact.other else "?",
                  "at", strength)
        self.material.color.set_hex(0xff3020)

    def on_collision_exit(self, contact: threepp.editor.Collision):
        self.material.color.set_hex(self.rest)
)",
                        /*y=*/2.f, /*restitution=*/0.9f, Primitive::Sphere);

    auto* material = dynamic_cast<MaterialWithColor*>(ball->material().get());
    REQUIRE(material != nullptr);
    const auto rest = material->color.getHex();

    rig.start();
    REQUIRE(rig.scripts.errorFor(ball->uuid).empty());

    bool wentRed = false;
    bool cameBack = false;
    for (int i = 0; i < 300; ++i) {
        rig.frame(kFixed);
        if (material->color.getHex() == 0xff3020) wentRed = true;
        else if (wentRed && material->color.getHex() == rest) cameBack = true;
    }

    CHECK(rig.scripts.errorFor(ball->uuid).empty());
    CHECK(wentRed);  // it landed
    CHECK(cameBack); // and bounced off again

    rig.stop();
}

TEST_CASE("a script session destroyed mid-play takes its contact watches with it",
          "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    PhysicsPlaySession physics;

    auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
    ground->name = "Ground";
    ground->scale.set(20.f, 0.2f, 20.f);
    ground->position.set(0.f, -0.1f, 0.f);
    PhysicsConfig groundConfig;
    groundConfig.enabled = true;
    groundConfig.body = PhysicsConfig::Body::Static;
    groundConfig.shape = PhysicsConfig::Shape::Box;
    groundConfig.write(*ground);
    scene.add(ground);

    auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
    box->name = "Box";
    box->position.set(0.f, 1.2f, 0.f);
    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Dynamic;
    config.shape = PhysicsConfig::Shape::Box;
    config.write(*box);
    ScriptConfig script;
    script.source = ledger("orphaned");
    script.write(*box);
    scene.add(box);

    physics.start(scene);
    {
        ScriptPlaySession scripts;
        scripts.start(scene);
        physics.update(kFixed);
        scripts.update(kFixed);
    }// destroyed without stop(), exactly as the editor tearing down mid-Play does

    // The world outlives it, and the box has yet to land — so the contact that
    // WOULD have been reported happens now, into a watcher that no longer exists.
    // Stepping through it is the whole assertion.
    for (int i = 0; i < 200; ++i) physics.update(kFixed);
    CHECK(box->position.y < 1.f);

    physics.stop();
}
