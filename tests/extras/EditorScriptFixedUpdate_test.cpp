// fixed_update(dt): a script method driven by the PHYSICS clock rather than by
// the frame — once per fixed substep, with the world's constant timestep.
//
// The sessions are driven exactly as the editor drives them (physics, sensors,
// scripts, in that order), because the point of the feature is what happens
// BETWEEN those calls: fixed_update runs from inside physics.update(), so a
// frame with two substeps calls it twice and a frame with none calls it not at
// all. Everything asserted here is about that: the count, the dt, the ordering
// against update(), and what a script sees of a world that is not there.

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

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // The default PhysxWorld timestep, i.e. what every fixed_update must be
    // handed. A frame of exactly this length is exactly one substep.
    constexpr float kFixed = 1.f / 60.f;

    std::shared_ptr<Mesh> addBox(Scene& scene, const char* name, const std::string& script,
                                 bool physical = true) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(0.f, 3.f, 0.f);

        if (physical) {
            PhysicsConfig physics;
            physics.enabled = true;
            physics.body = PhysicsConfig::Body::Dynamic;
            physics.shape = PhysicsConfig::Shape::Box;
            physics.mass = 2.f;
            physics.write(*box);
        }

        ScriptConfig config;
        config.source = script;
        config.write(*box);

        scene.add(box);
        return box;
    }

    // An IMU sampling every substep, noiseless — the timestamps are what this
    // file cares about, not the numbers.
    void instrument(Object3D& object) {

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Imu;
        sensor.rateHz = 0.f;
        sensor.gyroNoiseDensity = 0.f;
        sensor.gyroRandomWalk = 0.f;
        sensor.accelNoiseDensity = 0.f;
        sensor.accelRandomWalk = 0.f;
        sensor.write(object);
    }

    // The editor's registration order, and its stop order — the REVERSE of
    // registration, scripts first, so the substep callback is removed from a
    // world that is still alive. (The stale-handle window still exists — a
    // teardown that skips the controller — and detachFromPhysics still guards
    // it; the dead-world path is exercised where EditorScriptPhysics_test
    // stops physics by hand first.)
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
    };

    // Counts its own calls into a marker's transform, the only numeric channel a
    // script has back into a test:
    //   position = (fixed calls, dt values differing from the first, that dt)
    //   scale    = (accumulated fixed dt, update calls, updates that did not
    //               follow a fixed_update)
    const char* kTicker = R"(
import threepp

class Ticker:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "Ticks"
        obj.parent.add(self.marker)
        self.n = 0
        self.odd = 0
        self.dt0 = 0.0
        self.total = 0.0
        self.frames = 0
        self.bad = 0

    def fixed_update(self, dt):
        if self.n == 0:
            self.dt0 = dt
        elif dt != self.dt0:
            self.odd += 1
        self.n += 1
        self.total += dt
        self.publish()

    def update(self, dt):
        self.frames += 1
        # fixed_update runs inside the physics step, i.e. BEFORE this frame's
        # update. With one substep per frame the two counters must match here.
        if self.n != self.frames:
            self.bad += 1
        self.publish()

    def publish(self):
        self.marker.position.set(float(self.n), float(self.odd), self.dt0)
        self.marker.scale.set(self.total, float(self.frames), float(self.bad))
)";

}// namespace


TEST_CASE("fixed_update runs on the physics clock, not the frame", "[editor][scripting][physx]") {

    // 1.68 s of play, twice, at wildly different frame rates. 1.68 / (1/60) is
    // 100.8, so both must produce 100 substeps — far enough from a boundary that
    // float accumulation cannot move it, and the leftover 0.8 stays in the
    // accumulator rather than becoming a 101st call.
    constexpr int kExpected = 100;

    std::size_t irregularFrames = 0;
    float irregularTotal = 0.f;

    {
        Rig rig;
        auto box = addBox(rig.scene(), "Jitter", kTicker);
        rig.start();
        REQUIRE(rig.scripts.errorFor(box->uuid).empty());

        // A frame pattern no accumulator can mistake for a fixed one: below the
        // timestep, above it, and well below it. All under the maxSubSteps=4
        // hitch clamp (which would need ~67 ms in one go), so nothing is dropped.
        for (int i = 0; i < 30; ++i) {
            for (const float dt : {0.016f, 0.033f, 0.007f}) {
                rig.frame(dt);
                irregularTotal += dt;
                ++irregularFrames;
            }
        }

        auto* ticks = rig.marker("Ticks");
        REQUIRE(ticks != nullptr);
        CHECK(rig.scripts.errorFor(box->uuid).empty());

        CHECK(static_cast<int>(ticks->position.x) == kExpected);
        // Every dt was the same one, and it was the world's timestep.
        CHECK(static_cast<int>(ticks->position.y) == 0);
        CHECK_THAT(ticks->position.z, WithinAbs(kFixed, 1e-7));
        // Which makes the accumulated simulated time exact rather than drifting
        // with the frame pattern.
        CHECK_THAT(ticks->scale.x, WithinAbs(kExpected * kFixed, 1e-3));
        // 90 frames, 100 substeps: the two clocks are genuinely different.
        CHECK(static_cast<int>(ticks->scale.y) == static_cast<int>(irregularFrames));
        CHECK(static_cast<int>(ticks->scale.y) != kExpected);

        rig.stop();
    }

    {
        Rig rig;
        auto box = addBox(rig.scene(), "Steady", kTicker);
        rig.start();
        REQUIRE(rig.scripts.errorFor(box->uuid).empty());

        // The same 1.68 s at a different, uniform rate.
        rig.run(120, 0.014f);

        auto* ticks = rig.marker("Ticks");
        REQUIRE(ticks != nullptr);
        CHECK(static_cast<int>(ticks->position.x) == kExpected);
        CHECK(static_cast<int>(ticks->position.y) == 0);
        CHECK(static_cast<int>(ticks->scale.y) == 120);
        CHECK_THAT(irregularTotal, WithinAbs(120 * 0.014f, 1e-3));

        rig.stop();
    }
}

TEST_CASE("fixed_update precedes the frame's update, and reads a fresh sensor",
          "[editor][scripting][sensors][physx]") {

    Rig rig;

    // One substep per frame throughout, so "one fixed_update per update" is an
    // exact statement and the IMU's timestamps advance one step at a time.
    auto box = addBox(rig.scene(), "Body", R"(
import threepp

class Loop:
    def start(self, obj):
        self.imu = threepp.editor.imu_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)
        self.n = 0
        self.frames = 0
        self.bad = 0
        self.empty = 0
        self.advanced = 0
        self.stale = 0
        self.last_t = -1.0

    def fixed_update(self, dt):
        self.n += 1
        reading = self.imu.latest() if self.imu else None
        if reading is None:
            # Only before the first substep has been sampled at all.
            self.empty += 1
        elif reading.time > self.last_t:
            self.advanced += 1
            self.last_t = reading.time
        else:
            self.stale += 1
        self.marker.position.set(float(self.n), float(self.advanced), float(self.stale))
        self.marker.scale.set(float(self.frames), float(self.bad), float(self.empty))

    def update(self, dt):
        self.frames += 1
        if self.n != self.frames:
            self.bad += 1
        self.marker.scale.set(float(self.frames), float(self.bad), float(self.empty))
)");
    instrument(*box);

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());
    REQUIRE(rig.sensors.liveCount() == 1);

    rig.run(30);
    CHECK(rig.scripts.errorFor(box->uuid).empty());

    auto* ledger = rig.marker("Ledger");
    REQUIRE(ledger != nullptr);

    // 30 frames, 30 substeps, and every update() was preceded by its
    // fixed_update rather than followed by it.
    CHECK(static_cast<int>(ledger->position.x) == 30);
    CHECK(static_cast<int>(ledger->scale.x) == 30);
    CHECK(static_cast<int>(ledger->scale.y) == 0);

    // The measurement is fresh on every call but the first: the sensor is
    // sampled at the end of each substep, so the first fixed_update of a session
    // runs before any measurement exists at all, and every one after it sees a
    // timestamp it has not seen before. Never a repeat.
    CHECK(static_cast<int>(ledger->scale.z) == 1);
    CHECK(static_cast<int>(ledger->position.y) == 29);
    CHECK(static_cast<int>(ledger->position.z) == 0);

    rig.stop();
}

TEST_CASE("two substeps in one frame share the frame's measurements",
          "[editor][scripting][sensors][physx]") {

    // The other half of the freshness story, and a limit worth pinning: the
    // SENSOR is sampled every substep, but the session hands its retained copies
    // to script handles once per FRAME (SensorPlaySession::update -> drainBodies,
    // which is where the single-drainer rule puts it). So the two fixed_updates
    // of a double-length frame read the same measurement — the newest one that
    // existed when the frame began — and the next frame delivers both of them at
    // once. A controller is unbothered (it acts on the last measurement either
    // way); a test asserting "one new sample per call" would be asserting
    // something that is not true, so this asserts what is.
    Rig rig;

    auto box = addBox(rig.scene(), "Body", R"(
import threepp

class Loop:
    def start(self, obj):
        self.imu = threepp.editor.imu_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)
        self.n = 0
        self.empty = 0
        self.advanced = 0
        self.stale = 0
        self.last_t = -1.0

    def fixed_update(self, dt):
        self.n += 1
        reading = self.imu.latest() if self.imu else None
        if reading is None:
            self.empty += 1
        elif reading.time > self.last_t:
            self.advanced += 1
            self.last_t = reading.time
        else:
            self.stale += 1
        self.marker.position.set(float(self.n), float(self.advanced), float(self.stale))
        self.marker.scale.set(float(self.empty), self.last_t, 1.0)
)");
    instrument(*box);

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());

    constexpr int kFrames = 20;
    rig.run(kFrames, 2.f * kFixed);

    auto* ledger = rig.marker("Ledger");
    REQUIRE(ledger != nullptr);

    // Two calls per frame, with the constant timestep — the frame being twice
    // as long changes the COUNT, never the dt.
    CHECK(static_cast<int>(ledger->position.x) == 2 * kFrames);
    // Nothing at all on the first frame; afterwards one new measurement per
    // frame and one repeat.
    CHECK(static_cast<int>(ledger->scale.x) == 2);
    CHECK(static_cast<int>(ledger->position.y) == kFrames - 1);
    CHECK(static_cast<int>(ledger->position.z) == kFrames - 1);
    // And it is the second substep of the previous frame, not the first.
    CHECK_THAT(ledger->scale.y, WithinAbs(2. * (kFrames - 1) * kFixed, 1e-6));

    rig.stop();
}

TEST_CASE("a raising fixed_update is reported once and disabled", "[editor][scripting][physx]") {

    Rig rig;

    auto bad = addBox(rig.scene(), "Bad", R"(
import threepp

class Bad:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "BadMarker"
        obj.parent.add(self.marker)

    def fixed_update(self, dt):
        self.marker.position.x += 1.0
        raise ValueError("no")

    def update(self, dt):
        self.marker.position.y += 1.0
)");
    auto good = addBox(rig.scene(), "Good", kTicker);

    rig.start();
    REQUIRE(rig.scripts.errorFor(bad->uuid).empty());
    rig.run(20);

    auto* badMarker = rig.marker("BadMarker");
    REQUIRE(badMarker != nullptr);
    // It ran exactly once and was then disabled — whole, not just its
    // fixed_update: update() never got a turn either, because the first substep
    // of the first frame comes before the first frame's update.
    CHECK_THAT(badMarker->position.x, WithinAbs(1.f, 1e-6));
    CHECK_THAT(badMarker->position.y, WithinAbs(0.f, 1e-6));

    // One report, not one per substep, and it names the method.
    CHECK(rig.linesContaining("script error") == 1);
    CHECK(rig.linesContaining("(fixed_update)") == 1);
    CHECK(!rig.scripts.errorFor(bad->uuid).empty());

    // The other script kept its clock: 20 frames of exactly one substep each.
    CHECK(rig.scripts.errorFor(good->uuid).empty());
    auto* ticks = rig.marker("Ticks");
    REQUIRE(ticks != nullptr);
    CHECK(static_cast<int>(ticks->position.x) == 20);

    rig.stop();
}

TEST_CASE("without a physics world fixed_update never fires", "[editor][scripting][physx]") {

    SECTION("no physics session at all") {

        Rig rig;
        auto box = addBox(rig.scene(), "Orphan", kTicker, /*physical=*/false);

        rig.start(/*withPhysics=*/false);
        REQUIRE(rig.scripts.errorFor(box->uuid).empty());
        rig.runScriptsOnly(30);

        auto* ticks = rig.marker("Ticks");
        REQUIRE(ticks != nullptr);
        // No fixed clock, so no fixed_update — and one line saying so, at Play,
        // rather than a method that silently never runs.
        CHECK(static_cast<int>(ticks->position.x) == 0);
        CHECK(rig.linesContaining("fixed_update needs a playing physics world") == 1);
        // update() is untouched by any of it.
        CHECK(static_cast<int>(ticks->scale.y) == 30);

        rig.scripts.stop();
    }

    SECTION("a physics session with nothing physical in it still has a clock") {

        // PhysicsPlaySession::start() builds a world unconditionally, even for a
        // scene carrying no PhysicsConfig at all — so the fixed clock exists and
        // fixed_update runs on it. This pins that, since the no-world message
        // above depends on it being the only way to end up without one.
        Rig rig;
        auto box = addBox(rig.scene(), "Weightless", kTicker, /*physical=*/false);

        rig.start();
        REQUIRE(rig.physics.world() != nullptr);
        REQUIRE(rig.physics.bodyCount() == 0);
        rig.run(20);

        auto* ticks = rig.marker("Ticks");
        REQUIRE(ticks != nullptr);
        CHECK(static_cast<int>(ticks->position.x) == 20);
        CHECK(rig.linesContaining("fixed_update needs a playing physics world") == 0);

        rig.stop();
    }
}

// The Thrust script from doc/editor.md, verbatim. If this stops working the
// documentation is handing users a script that does not run.
TEST_CASE("the documented fixed_update script holds its height", "[editor][scripting][physx]") {

    Rig rig;

    auto box = addBox(rig.scene(), "Thrust", R"(
import threepp


class Thrust:
    height = 3.0
    stiffness = 40.0

    def start(self, obj: threepp.Object3D):
        self.body = threepp.editor.rigid_body_from_object(obj)

    def fixed_update(self, dt: float):
        if self.body is None:
            return
        # Once per substep, at 1/60 s whatever the window is doing.
        error = self.height - self.body.position.y
        lift = self.stiffness * error - 5.0 * self.body.velocity.y
        self.body.apply_force(threepp.Vector3(0, lift * self.body.mass, 0))
)");
    box->position.set(0.f, 1.f, 0.f);

    rig.start();
    REQUIRE(rig.scripts.errorFor(box->uuid).empty());

    // Deliberately ragged frames: the same controller, the same settled height.
    // Under update() this is the run where the answer would depend on the frame
    // pattern rather than on the gains.
    for (int i = 0; i < 200; ++i) {
        for (const float dt : {0.016f, 0.033f, 0.007f}) rig.frame(dt);
    }

    CHECK(rig.scripts.errorFor(box->uuid).empty());
    CHECK_THAT(box->position.y, WithinAbs(3.f, 0.3));

    rig.stop();
}

TEST_CASE("play, stop, play again re-registers cleanly", "[editor][scripting][physx]") {

    Rig rig;
    auto box = addBox(rig.scene(), "Replay", kTicker);

    rig.start();
    rig.run(15);
    auto* first = rig.marker("Ticks");
    REQUIRE(first != nullptr);
    CHECK(static_cast<int>(first->position.x) == 15);

    // The controller's order: physics stops FIRST, so the world (and with it
    // the registration) is gone before the script session's stop() runs. Holding
    // the handle past that has to be harmless.
    rig.stop();

    // The markers from the first run are still in the scene (the editor's
    // snapshot restore is what removes them, and there is none here); the second
    // run adds its own, so find the fresh one by counting instead.
    rig.start();
    rig.run(15);
    CHECK(rig.scripts.errorFor(box->uuid).empty());

    int counted = 0;
    float latest = 0.f;
    rig.scene().traverse([&](Object3D& object) {
        if (object.name == "Ticks") {
            ++counted;
            latest = object.position.x;
        }
    });
    CHECK(counted == 2);
    // Fresh instance, fresh count — not 30.
    CHECK(static_cast<int>(latest) == 15);

    rig.stop();
}

TEST_CASE("a script session destroyed mid-play takes its callback with it",
          "[editor][scripting][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    PhysicsPlaySession physics;
    auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
    box->name = "Box";
    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Dynamic;
    config.write(*box);
    ScriptConfig script;
    script.source = kTicker;
    script.write(*box);
    scene.add(box);

    physics.start(scene);
    const float startY = box->position.y;
    {
        ScriptPlaySession scripts;
        scripts.start(scene);
        physics.update(kFixed);
        scripts.update(kFixed);
    }// destroyed without stop(), exactly as the editor tearing down mid-Play does

    // The world outlives it. Stepping now would call into a freed Impl if the
    // destructor had not unregistered.
    for (int i = 0; i < 10; ++i) physics.update(kFixed);
    // It kept falling, i.e. the world still steps — ~0.16 m of free fall in the
    // 11 substeps that ran.
    CHECK(box->position.y < startY - 0.1f);

    physics.stop();
}

TEST_CASE("pause pauses fixed_update", "[editor][scripting][physx]") {

    SceneDocument document;

    auto physics = std::make_shared<PhysicsPlaySession>();
    auto scripts = std::make_shared<ScriptPlaySession>();

    auto box = addBox(document.scene(), "Paused", kTicker);

    PlayController play;
    play.addSession(physics);
    play.addSession(scripts);

    std::string error;
    REQUIRE(play.play(document, &error));

    for (int i = 0; i < 10; ++i) play.update(kFixed);
    auto* ticks = document.scene().getObjectByName("Ticks");
    REQUIRE(ticks != nullptr);
    CHECK(static_cast<int>(ticks->position.x) == 10);

    // Paused, the controller does not update the sessions at all: physics does
    // not step, so there are no substeps to run a fixed_update on. It costs
    // nothing to get right, but only because nothing else pretends to have a
    // clock of its own.
    play.pause();
    for (int i = 0; i < 30; ++i) play.update(kFixed);
    CHECK(static_cast<int>(ticks->position.x) == 10);
    CHECK(static_cast<int>(ticks->scale.y) == 10);// update() is paused too

    play.resume();
    for (int i = 0; i < 5; ++i) play.update(kFixed);
    CHECK(static_cast<int>(ticks->position.x) == 15);

    REQUIRE(play.stop(document, &error));
}
