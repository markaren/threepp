// threepp.editor.time: the two clocks a play session runs on, published to the
// scripts riding them.
//
// The feature exists because the clocks DISAGREE. update(dt) is handed the wall
// delta of the last frame; fixed_update(dt) is handed the physics world's
// constant substep; and PhysxWorld::step takes at most maxSubSteps substeps per
// call before discarding what is left of the accumulator. So a long frame
// advances wall time in full and simulated time only partly, permanently. That
// divergence is what the middle case here pins, and everything else is about
// making both clocks readable from anywhere in a script without either being
// mistaken for the other.
//
// Driven exactly as the editor drives the sessions (physics, then scripts), for
// the same reason EditorScriptFixedUpdate_test is: what the clock reads depends
// on what happened between those two calls.

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

#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // PhysxWorld's defaults, which PhysicsPlaySession does not override: the
    // substep every fixed_update is handed, and the ceiling on how many of them
    // one step() call may take before it drops the remainder.
    constexpr float kFixed = 1.f / 60.f;
    constexpr int kMaxSubSteps = 4;

    std::shared_ptr<Mesh> addBox(Scene& scene, const char* name, const std::string& script) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(0.f, 3.f, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 2.f;
        physics.write(*box);

        ScriptConfig config;
        config.source = script;
        config.write(*box);

        scene.add(box);
        return box;
    }

    struct Rig {

        SceneDocument document;
        PhysicsPlaySession physics;
        ScriptPlaySession scripts;
        std::vector<std::string> log;

        Rig() {
            scripts.setLogger([this](const std::string& line) { log.push_back(line); });
        }

        Scene& scene() { return document.scene(); }

        void start(bool withPhysics = true) {
            if (withPhysics) physics.start(scene());
            scripts.start(scene());
        }

        void frame(float dt) {
            physics.update(dt);
            scripts.update(dt);
        }

        void run(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) frame(dt);
        }

        // No world was ever built, so nothing steps and there is no fixed clock.
        void runScriptsOnly(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) scripts.update(dt);
        }

        void stop() {
            scripts.stop();
            physics.stop();
        }

        Object3D* marker(const char* name) { return scene().getObjectByName(name); }

        // What the editor's snapshot restore does on Stop, by hand: a script's
        // markers are play-time litter, and a second Play must not find the
        // first one's. Nothing about the clock — this rig just drives the
        // sessions, it does not snapshot the document around them.
        void dropMarkers() {
            for (const char* name : {"Clock", "Flags"}) {
                if (auto* found = scene().getObjectByName(name)) {
                    if (found->parent) found->parent->remove(*found);
                }
            }
        }
    };

    // Reports the clock through two markers, the only numeric channel a script
    // has back into a test:
    //   Clock.position = (wall_time, sim_time, steps)
    //   Clock.scale    = (sim_dt, fixed_update calls, disagreements)
    //   Flags.position = (fixed_clock, playing, frame_dt)
    //
    // `self.t` is stashed in start() ON PURPOSE. threepp.editor.time is one bound
    // object whose properties read the live clock, so a reference kept for the
    // whole session must keep reporting fresh numbers; if it were a snapshot every
    // reading below would be zero and every case here would fail.
    const char* kClockScript = R"(
import threepp

class Clock:
    def start(self, obj):
        self.t = threepp.editor.time
        self.marker = threepp.Object3D()
        self.marker.name = "Clock"
        obj.parent.add(self.marker)
        self.flags = threepp.Object3D()
        self.flags.name = "Flags"
        obj.parent.add(self.flags)
        self.n = 0
        self.bad = 0
        self.publish()

    def fixed_update(self, dt):
        # This hook runs pre-simulate, so the clock must read the START of the
        # substep about to be solved: n substeps done, n * sim_dt on the clock.
        if abs(self.t.sim_time - self.n * self.t.sim_dt) > 1e-6:
            self.bad += 1
        if self.t.steps != self.n:
            self.bad += 1
        # The substep the world hands the method and the one it publishes are the
        # same number, or one of them is lying.
        if dt != self.t.sim_dt:
            self.bad += 1
        self.n += 1
        self.publish()

    def update(self, dt):
        # Same again for the frame half: update's argument IS frame_dt.
        if dt != self.t.frame_dt:
            self.bad += 1
        self.publish()

    def publish(self):
        self.marker.position.set(self.t.wall_time, self.t.sim_time, float(self.t.steps))
        self.marker.scale.set(self.t.sim_dt, float(self.n), float(self.bad))
        self.flags.position.set(1.0 if self.t.fixed_clock else 0.0,
                                1.0 if self.t.playing else 0.0,
                                self.t.frame_dt)
)";

}// namespace


TEST_CASE("editor.time reports the frame clock and the physics clock", "[editor][scripting][physx]") {

    constexpr int kFrames = 30;

    Rig rig;
    addBox(rig.scene(), "Box", kClockScript);
    rig.start();
    rig.run(kFrames);

    auto* clock = rig.marker("Clock");
    auto* flags = rig.marker("Flags");
    REQUIRE(clock);
    REQUIRE(flags);

    // A frame of exactly one substep, thirty times: the two clocks agree here,
    // which is the only regime in which they ever do.
    CHECK_THAT(clock->position.x, WithinAbs(kFrames * kFixed, 1e-4));// wall
    CHECK_THAT(clock->position.y, WithinAbs(kFrames * kFixed, 1e-4));// sim
    CHECK(clock->position.z == static_cast<float>(kFrames));         // steps
    CHECK_THAT(clock->scale.x, WithinAbs(kFixed, 1e-7));             // sim_dt
    CHECK(clock->scale.y == static_cast<float>(kFrames));            // fixed_update calls
    // Every in-script assertion: sim_time at the substep boundary, steps against
    // the script's own count, both dt arguments against their published twins.
    CHECK(clock->scale.z == 0.f);

    CHECK(flags->position.x == 1.f);// fixed_clock: a world is playing
    CHECK(flags->position.y == 1.f);// playing
    CHECK_THAT(flags->position.z, WithinAbs(kFixed, 1e-7));// frame_dt

    // And the C++ side agrees with what the script saw, since both read the one
    // clock the session publishes.
    CHECK(scripting::scriptClock().steps == static_cast<std::uint64_t>(kFrames));
    CHECK(scripting::scriptClock().fixedClock);

    rig.stop();

    // Zeroed on the way down, exactly as the draw list and the resolver are: a
    // reader outside a session gets "not playing", not the last frame's numbers.
    CHECK_FALSE(scripting::scriptClock().active);
    CHECK(scripting::scriptClock().wallTime == 0.0);
    CHECK(scripting::scriptClock().steps == 0);
}

TEST_CASE("a hitching frame advances wall time and not simulated time", "[editor][scripting][physx]") {

    // THE reason editor.time exists. One second of wall clock in a single frame:
    // PhysxWorld::step takes maxSubSteps substeps, discards the rest of the
    // accumulator so it cannot spiral, and returns. Wall time gained a full
    // second; simulated time gained four sixtieths. Nothing catches that up
    // later, so a script integrating update()'s dt toward a physics quantity is
    // permanently wrong from here on — which it can now SEE.
    Rig rig;
    addBox(rig.scene(), "Box", kClockScript);
    rig.start();
    rig.frame(1.f);

    auto* clock = rig.marker("Clock");
    REQUIRE(clock);

    CHECK_THAT(clock->position.x, WithinAbs(1.0, 1e-5));                       // wall: all of it
    CHECK_THAT(clock->position.y, WithinAbs(kMaxSubSteps * kFixed, 1e-5));     // sim: capped
    CHECK(clock->position.z == static_cast<float>(kMaxSubSteps));              // steps
    CHECK(clock->scale.y == static_cast<float>(kMaxSubSteps));                 // fixed_update calls
    CHECK(clock->scale.z == 0.f);

    // The divergence is real and large — a full second against 67 ms.
    CHECK(clock->position.x - clock->position.y > 0.9f);

    rig.stop();
}

TEST_CASE("editor.time degrades to the frame clock with no world", "[editor][scripting]") {

    // No physics session was ever started, so there is no fixed clock to report.
    // fixed_update never fires (its own contract, tested elsewhere) and the sim
    // half falls back to the frame clock rather than reading zero forever — with
    // fixed_clock False saying where the numbers came from.
    constexpr int kFrames = 12;

    Rig rig;
    addBox(rig.scene(), "Box", kClockScript);
    rig.start(false);
    rig.runScriptsOnly(kFrames);

    auto* clock = rig.marker("Clock");
    auto* flags = rig.marker("Flags");
    REQUIRE(clock);
    REQUIRE(flags);

    CHECK_THAT(clock->position.x, WithinAbs(kFrames * kFixed, 1e-4));// wall
    CHECK_THAT(clock->position.y, WithinAbs(kFrames * kFixed, 1e-4));// sim == wall
    CHECK(clock->position.z == 0.f);                                 // no substep ran
    CHECK_THAT(clock->scale.x, WithinAbs(kFixed, 1e-7));             // sim_dt == frame_dt
    CHECK(clock->scale.y == 0.f);                                    // fixed_update never fired
    CHECK(clock->scale.z == 0.f);

    CHECK(flags->position.x == 0.f);// fixed_clock: nothing simulated it
    CHECK(flags->position.y == 1.f);// still playing

    rig.stop();
}

TEST_CASE("the clock starts from zero on every play", "[editor][scripting][physx]") {

    // Two sessions over one document. The second must not inherit the first's
    // elapsed time — the clock goes up with the session, like the resolver and
    // the draw list beside it.
    Rig rig;
    addBox(rig.scene(), "Box", kClockScript);

    rig.start();
    rig.run(20);
    const float first = rig.marker("Clock")->position.x;
    rig.stop();
    CHECK(first > 0.f);
    rig.dropMarkers();

    rig.start();
    rig.run(5);
    auto* clock = rig.marker("Clock");
    REQUIRE(clock);
    CHECK_THAT(clock->position.x, WithinAbs(5 * kFixed, 1e-4));
    CHECK(clock->position.z == 5.f);
    CHECK(clock->scale.z == 0.f);
    rig.stop();
}
