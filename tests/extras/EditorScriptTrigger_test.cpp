// on_trigger_enter / on_trigger_exit: the two optional script methods fired when
// a body enters and leaves a TRIGGER VOLUME — a shape cooked with PhysX's
// eTRIGGER_SHAPE instead of eSIMULATION_SHAPE, so it overlaps rather than
// collides.
//
// The sessions are driven exactly as the editor drives them (physics, sensors,
// scripts, in that order), because the feature lives in the gap between those
// calls: PhysX reports an overlap from inside physics.update() and the scripts
// hear about it from their own sweep afterwards. What is asserted here is what
// survives that gap — both edges, their order, who `other` is on EACH side of
// one overlap (the volume and the body that walked in both get called), and the
// thing that makes it a trigger at all: the body FELL STRAIGHT THROUGH.
//
// The interplay is asserted rather than assumed: a trigger overlap generates no
// contacts, so on_collision_enter must NOT fire for it while it still fires for
// an ordinary landing on the same frame budget.

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

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

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

    void attachScript(Object3D& object, const std::string& source) {

        if (source.empty()) return;
        ScriptConfig config;
        config.source = source;
        config.write(object);
    }

    // A box-shaped trigger volume: things pass through it, and it reports them.
    std::shared_ptr<Mesh> addVolume(Scene& scene, const char* name, const std::string& script,
                                    const Vector3& position, const Vector3& size,
                                    PhysicsConfig::Shape shape = PhysicsConfig::Shape::Box) {

        auto volume = ObjectFactory::createPrimitive(Primitive::Box, scene);
        volume->name = name;
        volume->position.copy(position);
        volume->scale.copy(size);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Static;
        physics.shape = shape;
        physics.trigger = true;
        physics.write(*volume);

        attachScript(*volume, script);
        scene.add(volume);
        return volume;
    }

    // A dynamic body, dropped from `y`.
    std::shared_ptr<Mesh> addBody(Scene& scene, const char* name, const std::string& script,
                                  const Vector3& position, float restitution = 0.05f,
                                  Primitive shape = Primitive::Box, float scale = 1.f) {

        auto body = ObjectFactory::createPrimitive(shape, scene);
        body->name = name;
        body->position.copy(position);
        body->scale.set(scale, scale, scale);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Auto;
        physics.mass = 2.f;
        physics.restitution = restitution;
        physics.write(*body);

        attachScript(*body, script);
        scene.add(body);
        return body;
    }

    // A wide static slab whose top face is at y = 0. An ordinary collider — the
    // control every "the trigger did NOT collide" assertion is read against.
    std::shared_ptr<Mesh> addGround(Scene& scene, const char* name, float restitution = 0.05f) {

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

        scene.add(ground);
        return ground;
    }

    // An object with a script and deliberately NO physics.
    std::shared_ptr<Mesh> addWeightless(Scene& scene, const char* name, const std::string& script) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(20.f, 20.f, 0.f);
        attachScript(*box, script);
        scene.add(box);
        return box;
    }

    // The editor's registration order, and its stop order (physics first, so the
    // world — and every trigger watch in it — is gone before the script session
    // is asked to give its registrations back). Both sessions log into ONE list:
    // the trimesh fallback is the physics session's line and the missing-body one
    // is the script session's, and a test reads them the same way.
    struct Rig {

        SceneDocument document;
        PhysicsPlaySession physics;
        PhysxSensorPlaySession sensors;
        ScriptPlaySession scripts;
        std::vector<std::string> log;

        Rig() {
            sensors.setPhysics(&physics);
            scripts.setLogger([this](const std::string& line) { log.push_back(line); });
            physics.setLogger([this](const std::string& line) { log.push_back(line); });
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
    //                  scale    = (windows carrying BOTH edges, collision enters,
    //                              0)
    //   "<tag>-who"    renamed to "<tag>-who|<type>|<name>" on every enter
    //
    // `extra` is appended as further methods of the same class.
    std::string ledger(const std::string& tag, const std::string& extra = "") {

        std::string source = R"(
import threepp

class Watcher:
    def start(self, obj):
        self.obj = obj
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
        self.hits = 0
        self.seen_enters = 0
        self.seen_exits = 0
        self.publish()

    def on_trigger_enter(self, other):
        self.enters += 1
        self.open += 1
        if other is None:
            self.who.name = "TAG-who|None|"
        else:
            self.who.name = "TAG-who|" + type(other).__name__ + "|" + other.name
        self.publish()

    def on_trigger_exit(self, other):
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
        self.count.scale.set(float(self.both), float(self.hits), 0.0)
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

    // Counts contacts into the same ledger, so one test can assert that a
    // trigger overlap produced NONE of them while an ordinary landing did.
    const char* kCountsContacts = R"(
    def on_collision_enter(self, contact):
        self.hits += 1
        self.publish()
)";

}// namespace


TEST_CASE("a body falls through a trigger volume, and both sides are told",
          "[editor][scripting][physx]") {

    Rig rig;
    // Nothing under it: what the volume does NOT do is the assertion.
    auto zone = addVolume(rig.scene(), "Zone", ledger("zone"), {0.f, 0.f, 0.f}, {4.f, 0.5f, 4.f});
    auto faller = addBody(rig.scene(), "Faller", ledger("box"), {0.f, 3.f, 0.f});

    rig.start();
    REQUIRE(rig.scripts.errorFor(zone->uuid).empty());
    REQUIRE(rig.scripts.errorFor(faller->uuid).empty());

    // Every frame strictly lower than the last, for the whole fall: a trigger
    // resolves nothing, so there is no frame where the volume slowed it, caught
    // it, or bounced it. This is the feature, not a side effect of it.
    // One warm-up frame first: PhysxWorld interpolates a binding between the two
    // most recent substeps, and at exactly one substep per frame the alpha is
    // always zero — so the first frame still shows the authored pose.
    rig.frame(kFixed);

    bool monotonic = true;
    float previous = faller->position.y;
    for (int i = 0; i < 240; ++i) {
        rig.frame(kFixed);
        if (faller->position.y >= previous) monotonic = false;
        previous = faller->position.y;
    }
    CHECK(monotonic);
    CHECK(faller->position.y < -5.f);// long past where a collider would have held it

    auto* zoneCount = rig.marker("zone-count");
    auto* boxCount = rig.marker("box-count");
    REQUIRE(zoneCount != nullptr);
    REQUIRE(boxCount != nullptr);

    // One crossing, two scripts, one enter and one exit each — including the
    // volume, which is STATIC and would be invisible to PhysxWorld's own
    // binding list.
    CHECK(static_cast<int>(zoneCount->position.x) == 1);
    CHECK(static_cast<int>(zoneCount->position.y) == 1);
    CHECK(static_cast<int>(zoneCount->position.z) == 0);
    CHECK(static_cast<int>(boxCount->position.x) == 1);
    CHECK(static_cast<int>(boxCount->position.y) == 1);
    CHECK(static_cast<int>(boxCount->position.z) == 0);

    // Each names the other, as its concrete type rather than as Object3D. The
    // entering body is not itself a trigger, and it is told all the same.
    CHECK(rig.tagged("zone-who|") == "zone-who|Mesh|Faller");
    CHECK(rig.tagged("box-who|") == "box-who|Mesh|Zone");

    CHECK(rig.scripts.errorFor(zone->uuid).empty());
    CHECK(rig.scripts.errorFor(faller->uuid).empty());

    rig.stop();
}

TEST_CASE("a body resting inside a trigger enters once and never leaves",
          "[editor][scripting][physx]") {

    // Also the sleeping case, which is the one worth pinning: PhysX stops
    // reporting for a pair whose actor has gone to sleep, and there is no
    // PERSISTS event for a trigger anyway. Neither may read as a departure.
    Rig rig;
    addGround(rig.scene(), "Ground");
    auto zone = addVolume(rig.scene(), "Zone", ledger("zone"), {0.f, 1.f, 0.f}, {4.f, 2.f, 4.f});
    auto box = addBody(rig.scene(), "Sitter", ledger("box"), {0.f, 1.2f, 0.f});

    rig.start();
    REQUIRE(rig.scripts.errorFor(zone->uuid).empty());

    // Long enough to land, settle, and fall asleep on the floor inside it.
    rig.run(400);

    auto* zoneCount = rig.marker("zone-count");
    auto* boxCount = rig.marker("box-count");
    REQUIRE(zoneCount != nullptr);
    REQUIRE(boxCount != nullptr);

    CHECK(static_cast<int>(zoneCount->position.x) == 1);// one enter
    CHECK(static_cast<int>(zoneCount->position.y) == 0);// still inside
    CHECK(static_cast<int>(boxCount->position.x) == 1);
    CHECK(static_cast<int>(boxCount->position.y) == 0);

    // It rests ON THE GROUND, inside the volume: the trigger let it pass and the
    // ordinary collider under it did not.
    CHECK_THAT(box->position.y, WithinAbs(0.5f, 0.05f));

    rig.stop();
}

TEST_CASE("a crossing that begins and ends inside one frame still delivers both edges",
          "[editor][scripting][physx]") {

    // The rule the queue exists for. Overlaps are reported from inside the
    // solver and delivered a sweep later; a small ball crossing a thin volume at
    // four substeps per frame regularly enters and leaves between two
    // deliveries. A state flag ("am I inside?") sampled per frame would see no
    // change at all and report NOTHING. The queue keeps both edges, in order.
    // A small ball through a thin gate, fast: the crossing is ~1.5 substeps of
    // travel, comfortably inside one four-substep frame, and it repeats on every
    // bounce so the phase against the frame boundary keeps moving.
    Rig rig;
    addGround(rig.scene(), "Ground", /*restitution=*/0.95f);
    auto zone = addVolume(rig.scene(), "Gate", ledger("gate"), {0.f, 2.f, 0.f}, {4.f, 0.1f, 4.f});
    auto ball = addBody(rig.scene(), "Ball", "", {0.f, 9.f, 0.f}, /*restitution=*/0.95f,
                        Primitive::Sphere, /*scale=*/0.2f);

    rig.start();
    REQUIRE(rig.scripts.errorFor(zone->uuid).empty());

    // Four substeps per frame — PhysxWorld's maxSubSteps, so nothing is dropped
    // and the delivery window is as wide as it ever gets.
    rig.run(150, 4.f * kFixed);
    CHECK(rig.scripts.errorFor(zone->uuid).empty());

    auto* count = rig.marker("gate-count");
    REQUIRE(count != nullptr);

    const int enters = static_cast<int>(count->position.x);
    const int exits = static_cast<int>(count->position.y);

    CHECK(enters >= 3);        // it fell through, bounced, came back up through it
    CHECK(exits >= 2);         // and left every time
    CHECK(enters - exits <= 1);// paired up, at most one crossing still open
    // Never an exit with nothing open: within a window the enter always came
    // first, which is the ordering half of the rule.
    CHECK(static_cast<int>(count->position.z) == 0);
    // And at least one whole crossing lived and died between two deliveries.
    CHECK(static_cast<int>(count->scale.x) >= 1);

    // Still bouncing under the height it was dropped from: the gate never held
    // it, it just kept passing through.
    CHECK(ball->position.y < 9.f);

    rig.stop();
}

TEST_CASE("a trigger overlap generates no contacts at all", "[editor][scripting][physx]") {

    // The interplay bullet, asserted rather than asserted-in-prose. Both scripts
    // define on_collision_enter as well; only the one that touched the GROUND
    // may see it fire.
    Rig rig;
    addGround(rig.scene(), "Ground");
    auto zone = addVolume(rig.scene(), "Zone", ledger("zone", kCountsContacts),
                          {0.f, 1.f, 0.f}, {4.f, 2.f, 4.f});
    auto box = addBody(rig.scene(), "Sitter", ledger("box", kCountsContacts), {0.f, 1.2f, 0.f});

    rig.start();
    rig.run(200);

    auto* zoneCount = rig.marker("zone-count");
    auto* boxCount = rig.marker("box-count");
    REQUIRE(zoneCount != nullptr);
    REQUIRE(boxCount != nullptr);

    // The volume was entered and reported nothing else. A trigger shape carries
    // eTRIGGER_SHAPE instead of eSIMULATION_SHAPE, so the pair produces no
    // manifold for anything to report.
    CHECK(static_cast<int>(zoneCount->position.x) == 1);// on_trigger_enter fired
    CHECK(static_cast<int>(zoneCount->scale.y) == 0);   // on_collision_enter did not

    // The falling box got BOTH, from two different bodies: the trigger it
    // crossed and the ground it landed on. Which is the positive control — the
    // collision path is wired and working in this very session.
    CHECK(static_cast<int>(boxCount->position.x) == 1);
    CHECK(static_cast<int>(boxCount->scale.y) >= 1);

    CHECK(rig.scripts.errorFor(zone->uuid).empty());
    CHECK(rig.scripts.errorFor(box->uuid).empty());

    rig.stop();
}

TEST_CASE("a TriMesh trigger falls back to a convex hull and still reports",
          "[editor][scripting][physx]") {

    // PhysX rejects eTRIGGER_SHAPE on triangle-mesh geometry outright — a
    // trigger tests points against a volume, and a triangle soup has no inside.
    // A trigger that silently does not exist is worse than an approximate one,
    // so the cook substitutes the convex hull and says so once.
    Rig rig;
    auto zone = addVolume(rig.scene(), "Zone", ledger("zone"), {0.f, 0.f, 0.f}, {4.f, 0.5f, 4.f},
                          PhysicsConfig::Shape::TriMesh);
    auto faller = addBody(rig.scene(), "Faller", ledger("box"), {0.f, 3.f, 0.f});

    rig.start();

    // One line, naming the object and the reason, at Play.
    CHECK(rig.linesContaining("no triangle-mesh trigger") == 1);
    CHECK(rig.linesContaining("\"Zone\"") == 1);
    // And no refusal: the substituted hull IS a legal trigger shape.
    CHECK(rig.linesContaining("will not make a trigger") == 0);

    rig.run(240);

    auto* zoneCount = rig.marker("zone-count");
    auto* boxCount = rig.marker("box-count");
    REQUIRE(zoneCount != nullptr);
    REQUIRE(boxCount != nullptr);

    // The volume is there, it reports, and it is still not a collider.
    CHECK(static_cast<int>(zoneCount->position.x) == 1);
    CHECK(static_cast<int>(zoneCount->position.y) == 1);
    CHECK(static_cast<int>(boxCount->position.x) == 1);
    CHECK(faller->position.y < -5.f);

    rig.stop();
}

TEST_CASE("a raising on_trigger_enter is reported once and disabled",
          "[editor][scripting][physx]") {

    Rig rig;
    auto bad = addVolume(rig.scene(), "BadZone", R"(
import threepp

class Bad:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "BadMarker"
        obj.parent.add(self.marker)

    def on_trigger_enter(self, other):
        self.marker.position.x += 1.0
        raise ValueError("no")

    def on_trigger_exit(self, other):
        self.marker.position.z += 1.0

    def update(self, dt):
        self.marker.position.y += 1.0
)",
                          {0.f, 0.f, 0.f}, {4.f, 0.5f, 4.f});
    addBody(rig.scene(), "Faller", "", {0.f, 3.f, 0.f});

    auto good = addVolume(rig.scene(), "GoodZone", ledger("good"), {10.f, 0.f, 0.f},
                          {4.f, 0.5f, 4.f});
    addBody(rig.scene(), "OtherFaller", "", {10.f, 3.f, 0.f});

    rig.start();
    REQUIRE(rig.scripts.errorFor(bad->uuid).empty());
    rig.run(240);

    auto* marker = rig.marker("BadMarker");
    REQUIRE(marker != nullptr);
    // It ran once and was then disabled WHOLE: no exit, and no more update()
    // either — the instance is out for the session, not the method.
    CHECK_THAT(marker->position.x, WithinAbs(1.f, 1e-6));
    CHECK_THAT(marker->position.z, WithinAbs(0.f, 1e-6));
    CHECK(marker->position.y > 0.f);
    CHECK(marker->position.y < 240.f);

    // One report, not one per overlap, and it names the method.
    CHECK(rig.linesContaining("script error") == 1);
    CHECK(rig.linesContaining("(on_trigger_enter)") == 1);
    CHECK(!rig.scripts.errorFor(bad->uuid).empty());

    // The other volume, a scene away, is untouched by any of it.
    CHECK(rig.scripts.errorFor(good->uuid).empty());
    auto* count = rig.marker("good-count");
    REQUIRE(count != nullptr);
    CHECK(static_cast<int>(count->position.x) == 1);
    CHECK(static_cast<int>(count->position.y) == 1);

    rig.stop();
}

TEST_CASE("trigger callbacks on an object with no body say so once",
          "[editor][scripting][physx]") {

    SECTION("a physics world, but nothing physical on this object") {

        Rig rig;
        addVolume(rig.scene(), "Zone", "", {0.f, 0.f, 0.f}, {4.f, 0.5f, 4.f});
        auto orphan = addWeightless(rig.scene(), "Orphan", ledger("orphan"));

        rig.start();
        REQUIRE(rig.scripts.errorFor(orphan->uuid).empty());
        rig.run(120);

        auto* count = rig.marker("orphan-count");
        REQUIRE(count != nullptr);
        CHECK(static_cast<int>(count->position.x) == 0);
        CHECK(static_cast<int>(count->position.y) == 0);
        // One line, naming the object, at Play — not a method that silently
        // never runs, and not one line per frame.
        CHECK(rig.linesContaining("trigger callbacks need a physics body") == 1);
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

        CHECK(rig.linesContaining("trigger callbacks need a playing physics world") == 1);
        auto* count = rig.marker("orphan-count");
        REQUIRE(count != nullptr);
        CHECK(static_cast<int>(count->position.x) == 0);

        rig.scripts.stop();
    }
}

TEST_CASE("play, stop, play again re-registers the trigger watches cleanly",
          "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground");
    auto zone = addVolume(rig.scene(), "Zone", ledger("zone"), {0.f, 1.f, 0.f}, {4.f, 2.f, 4.f});
    auto box = addBody(rig.scene(), "Sitter", "", {0.f, 1.2f, 0.f});

    rig.start();
    rig.run(200);

    auto* first = rig.marker("zone-count");
    REQUIRE(first != nullptr);
    CHECK(static_cast<int>(first->position.x) == 1);

    // Physics stops FIRST, so the world (and every trigger watch in it) is gone
    // before the script session is asked to give its registrations back.
    rig.stop();

    // The box is left resting inside the volume, so the second session's very
    // first substep finds the overlap already there: exactly one enter again,
    // never two, and never one inherited from the run before.
    rig.start();
    rig.run(200);
    CHECK(rig.scripts.errorFor(zone->uuid).empty());

    // The first run's markers are still in the scene (the editor's snapshot
    // restore is what removes them, and there is none here), so count instead of
    // looking one up by name.
    int counted = 0;
    float latestEnters = -1.f;
    float latestExits = -1.f;
    rig.scene().traverse([&](Object3D& object) {
        if (object.name == "zone-count") {
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

TEST_CASE("a trigger volume is still visible to editor.raycast", "[editor][scripting][physx]") {

    // The documented answer to "what does the query see". A trigger shape loses
    // eSIMULATION_SHAPE but KEEPS eSCENE_QUERY_SHAPE, so a ray still finds it —
    // which is what lets a script ask where a zone is without waiting to be
    // inside it, and what a ground check has to know to pass `ignore`.
    Rig rig;
    addVolume(rig.scene(), "Zone", "", {0.f, 1.f, 0.f}, {4.f, 2.f, 4.f});
    auto prober = addWeightless(rig.scene(), "Prober", R"(
import threepp

class Prober:
    def start(self, obj):
        self.report = threepp.Object3D()
        self.report.name = "probe"
        obj.parent.add(self.report)

    def update(self, dt):
        hit = threepp.editor.raycast(threepp.Vector3(0.0, 5.0, 0.0),
                                     threepp.Vector3(0.0, -1.0, 0.0))
        if hit is not None and hit.object is not None:
            self.report.name = "probe|" + hit.object.name
            self.report.position.y = hit.point.y
)");

    rig.start();
    REQUIRE(rig.scripts.errorFor(prober->uuid).empty());
    rig.run(3);
    CHECK(rig.scripts.errorFor(prober->uuid).empty());

    CHECK(rig.tagged("probe|") == "probe|Zone");
    auto* report = rig.marker("probe|Zone");
    REQUIRE(report != nullptr);
    // The top face of a volume spanning y = 0..2.
    CHECK_THAT(report->position.y, WithinAbs(2.f, 0.05f));

    rig.stop();
}

// The GoalZone / Scoreboard pair from doc/editor.md, verbatim. Three features
// composing: a trigger volume, a script reaching another script's live instance,
// and the callbacks themselves. If this stops working the documentation is
// handing users a pair of scripts that do not run.
TEST_CASE("the documented goal-zone scripts score a ball", "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene(), "Ground");

    auto scoreboard = ObjectFactory::createPrimitive(Primitive::Box, rig.scene());
    scoreboard->name = "Scoreboard";
    scoreboard->position.set(0.f, 5.f, 0.f);
    attachScript(*scoreboard, R"(
import threepp


class Scoreboard:
    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.score = 0

    def scored(self, who: threepp.Object3D):
        self.score += 1
        print("goal by", who.name, "- score is now", self.score)
        self.obj.scale.y = 1.0 + self.score      # the bar grows
)");
    rig.scene().add(scoreboard);

    auto zone = addVolume(rig.scene(), "GoalZone", R"(
import threepp


class GoalZone:
    def start(self, obj: threepp.Object3D):
        # Resolve in start(), use later: every script instance exists by now,
        # but its own start() may not have run yet.
        self.board = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Scoreboard"))

    def on_trigger_enter(self, other: threepp.Object3D):
        if self.board is not None and other is not None:
            self.board.scored(other)
)",
                          {0.f, 1.f, 0.f}, {4.f, 2.f, 4.f});

    auto ball = addBody(rig.scene(), "Ball", "", {0.f, 3.f, 0.f}, 0.05f, Primitive::Sphere);

    rig.start();
    REQUIRE(rig.scripts.errorFor(zone->uuid).empty());
    REQUIRE(rig.scripts.errorFor(scoreboard->uuid).empty());

    rig.run(200);
    CHECK(rig.scripts.errorFor(zone->uuid).empty());
    CHECK(rig.scripts.errorFor(scoreboard->uuid).empty());

    // The ball dropped in, the zone told the scoreboard through its live
    // instance, and the scoreboard counted it ONCE — the bar it grows is the
    // only channel the test needs, and the script would grow it in the editor
    // just the same.
    CHECK_THAT(scoreboard->scale.y, WithinAbs(2.f, 1e-5));
    CHECK(ball->position.y < 1.f);// it came to rest inside the zone, on the floor

    rig.stop();
}
