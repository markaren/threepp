// threepp.editor.start_coroutine: generators a play session resumes once per
// frame, and the four things that decide whether they are usable.
//
// 1. WHICH CLOCK. wait() measures SIMULATED seconds, so a mission freezes when
//    physics is starved. The first case here proves that the only way it can be
//    proved: by hitching a frame, where wall time gains a full second and the
//    world gains four sixtieths, and watching the wait NOT fire.
// 2. WHERE THE PUMP SITS. After physics, after every script's update(), inside
//    the same GIL acquisition — so an until() predicate reads the settled frame
//    and can catch a flag another script set in the same one.
// 3. WHO OWNS A TASK. Attribution comes from the dispatch that started it, and
//    is what lets a raise inside a coroutine disable that instance whole, on the
//    same one-report path every other script error takes, with the neighbours
//    untouched.
// 4. WHEN THEY DIE. Cancel runs finally:; Stop drops everything; a second Play
//    starts from zero; and start_coroutine outside a dispatch refuses.
//
// The sessions are driven exactly as the editor drives them (physics, then
// scripts), because half of what is asserted lives in the gap between the two
// calls.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "ScriptHost.hpp"
#include "Scripting.hpp"

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

    namespace py = pybind11;

    // PhysxWorld's defaults, which PhysicsPlaySession does not override.
    constexpr float kFixed = 1.f / 60.f;
    constexpr int kMaxSubSteps = 4;

    // An object carrying a script. `physics` decides whether it also gets a
    // body — most cases here need no world at all, and say so by not building
    // one (sim time then degrades to wall time, which is fine for counting
    // frames and wrong only for the hitch case, which builds one).
    std::shared_ptr<Mesh> addScript(Scene& scene, const char* name, const std::string& source,
                                    bool physics = false, float y = 3.f) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(0.f, y, 0.f);

        if (physics) {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Dynamic;
            config.shape = PhysicsConfig::Shape::Box;
            config.mass = 2.f;
            config.write(*box);
        }

        ScriptConfig script;
        script.source = source;
        script.write(*box);

        scene.add(box);
        return box;
    }

    // A wide static slab whose top face is at y = 0 — something to land on.
    std::shared_ptr<Mesh> addGround(Scene& scene) {

        auto ground = ObjectFactory::createPrimitive(Primitive::Box, scene);
        ground->name = "Ground";
        ground->scale.set(20.f, 0.2f, 20.f);
        ground->position.set(0.f, -0.1f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Static;
        config.shape = PhysicsConfig::Shape::Box;
        config.restitution = 0.05f;
        config.write(*ground);

        scene.add(ground);
        return ground;
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

        void start(bool withPhysics = false) {
            if (withPhysics) physics.start(scene());
            scripts.start(scene());
        }

        // The editor's own order: physics first, scripts after.
        void frame(float dt = kFixed) {
            physics.update(dt);
            scripts.update(dt);
        }

        void run(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) frame(dt);
        }

        // No world was ever built, so nothing steps it.
        void runScriptsOnly(int frames, float dt = kFixed) {
            for (int i = 0; i < frames; ++i) scripts.update(dt);
        }

        void stop() {
            scripts.stop();
            physics.stop();
        }

        Object3D* marker(const char* name) { return scene().getObjectByName(name); }

        // Markers are play-time litter, exactly as they are in
        // EditorScriptClock_test: what the editor's snapshot restore drops on
        // Stop, done by hand so a second Play does not find the first's.
        void dropMarkers(std::initializer_list<const char*> names) {
            for (const char* name : names) {
                if (auto* found = scene().getObjectByName(name)) {
                    if (found->parent) found->parent->remove(*found);
                }
            }
        }

        [[nodiscard]] std::size_t errorLines() const {
            return static_cast<std::size_t>(
                    std::count_if(log.begin(), log.end(), [](const std::string& line) {
                        return line.find("script error") != std::string::npos;
                    }));
        }
    };

    // How many tasks the scheduler is holding. The registry is a plain list in
    // the threepp.editor submodule; reading it directly is what lets "Stop drops
    // everything" be an assertion rather than an inference.
    std::size_t liveTasks() {

        if (!scripting::interpreterStarted()) return 0;
        py::gil_scoped_acquire gil;
        return py::len(py::module_::import("threepp").attr("editor").attr("_tasks"));
    }

    // ------------------------------------------------------------------ 1 ---
    //
    // wait() on the SIM clock. The coroutine yields its wait on a normal frame,
    // then one frame takes a full second of WALL time — which buys 4 substeps
    // and nothing more. A wall-paced wait would have fired there; this one must
    // not, and must fire only once the world has actually simulated 0.505 s past
    // the mark.
    //
    // 0.505 rather than 0.5 on purpose: every substep boundary is a multiple of
    // 1/60, and a deadline landing exactly on one turns the firing frame into a
    // floating-point coin toss.
    const char* kWaitScript = R"(
import threepp


class Waiter:
    def start(self, obj):
        self.t = threepp.editor.time
        self.frames = 0
        self.marker = threepp.Object3D()
        self.marker.name = "Wait"
        obj.parent.add(self.marker)
        self.mark = threepp.Object3D()
        self.mark.name = "Mark"
        obj.parent.add(self.mark)
        threepp.editor.start_coroutine(self.mission())

    def update(self, dt):
        self.frames += 1

    def mission(self):
        # Runs at the FIRST pump, so the deadline is measured from that frame.
        self.mark.position.set(self.t.sim_time, self.t.wall_time, float(self.frames))
        yield threepp.editor.wait(0.505)
        # (sim, wall, frame) at the moment it fired.
        self.marker.position.set(self.t.sim_time, self.t.wall_time, float(self.frames))
)";

    // ------------------------------------------------------------------ 2 ---
    //
    // A bare yield is exactly one resume per pump, and a pump is one frame.
    const char* kTickScript = R"(
import threepp


class Ticker:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "Tick"
        obj.parent.add(self.marker)
        self.n = 0
        self.frames = 0
        self.task = threepp.editor.start_coroutine(self.forever())

    def update(self, dt):
        self.frames += 1
        # Resumes so far, frames so far, done yet. Written from update() rather
        # than from the coroutine, so the numbers are read at the same instant.
        self.marker.position.set(float(self.n), float(self.frames),
                                 1.0 if self.task.done else 0.0)

    def forever(self):
        while True:
            yield
            self.n += 1
)";

    // ------------------------------------------------------------------ 3 ---
    //
    // until(): the truthy value is sent in, and the predicate sees the frame
    // AFTER every update() has run. The flag is set by a NEIGHBOUR's update(),
    // and the catcher must see it in the same frame it was set.
    const char* kFlagScript = R"(
import threepp


class Flagger:
    def start(self, obj):
        self.frames = 0
        self.payload = None

    def update(self, dt):
        self.frames += 1
        if self.frames == 4:
            # Something with identity, so "the value arrived" is provable.
            self.payload = threepp.Vector3(7.0, 8.0, 9.0)
)";

    const char* kCatchScript = R"(
import threepp


class Catcher:
    def start(self, obj):
        self.frames = 0
        self.marker = threepp.Object3D()
        self.marker.name = "Caught"
        # scale defaults to (1, 1, 1); zeroed so "nothing was written yet" reads
        # as zero here the same way a position does.
        self.marker.scale.set(0.0, 0.0, 0.0)
        obj.parent.add(self.marker)
        self.other = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Flagger"))
        threepp.editor.start_coroutine(self.watch())

    def update(self, dt):
        self.frames += 1

    def watch(self):
        got = yield threepp.editor.until(lambda: self.other.payload)
        # The value the predicate returned, and the frame it arrived on.
        self.marker.position.copy(got)
        self.marker.scale.x = float(self.frames)
)";

    // ------------------------------------------------------------------ 4 ---
    //
    // Nesting: a yielded generator is run to completion and its return value is
    // what the parent's yield evaluates to.
    const char* kNestScript = R"(
import threepp


class Nester:
    def start(self, obj):
        self.frames = 0
        self.marker = threepp.Object3D()
        self.marker.name = "Nest"
        obj.parent.add(self.marker)
        threepp.editor.start_coroutine(self.parent())

    def update(self, dt):
        self.frames += 1

    def child(self, base):
        yield
        yield
        return base * 3.0

    def parent(self):
        # Two children in sequence, the second fed by the first.
        first = yield self.child(2.0)
        second = yield self.child(first)
        self.marker.position.set(first, second, float(self.frames))
)";

    // ------------------------------------------------------------------ 5 ---
    //
    // A raise inside a coroutine disables the owning instance WHOLE: its
    // update() stops too, its second task is dropped with it, one traceback is
    // logged, and the neighbour never notices.
    const char* kBoomScript = R"(
import threepp


class Boom:
    def start(self, obj):
        self.frames = 0
        self.other = 0
        self.marker = threepp.Object3D()
        self.marker.name = "BoomState"
        obj.parent.add(self.marker)
        threepp.editor.start_coroutine(self.bad())
        # A second task on the SAME instance, which must go down with it.
        threepp.editor.start_coroutine(self.sibling())

    def update(self, dt):
        self.frames += 1
        self.marker.position.set(float(self.frames), float(self.other), 0.0)

    def bad(self):
        yield
        yield
        raise RuntimeError("coroutine went bang")

    def sibling(self):
        while True:
            yield
            self.other += 1
)";

    const char* kSurvivorScript = R"(
import threepp


class Survivor:
    def start(self, obj):
        self.n = 0
        self.marker = threepp.Object3D()
        self.marker.name = "SurvivorState"
        obj.parent.add(self.marker)
        threepp.editor.start_coroutine(self.forever())

    def forever(self):
        while True:
            yield
            self.n += 1
            self.marker.position.x = float(self.n)
)";

    // ------------------------------------------------------------------ 6 ---
    //
    // cancel(): no further resumes, and the generator's finally: ran.
    const char* kCancelScript = R"(
import threepp


class Canceller:
    def start(self, obj):
        self.frames = 0
        self.n = 0
        self.marker = threepp.Object3D()
        self.marker.name = "Cancel"
        obj.parent.add(self.marker)
        self.task = threepp.editor.start_coroutine(self.forever())

    def update(self, dt):
        self.frames += 1
        if self.frames == 3:
            self.task.cancel()
        # (resumes, cleaned-up flag, done flag)
        self.marker.position.set(float(self.n), self.marker.position.y,
                                 1.0 if self.task.done else 0.0)

    def forever(self):
        try:
            while True:
                yield
                self.n += 1
        finally:
            # Proof the finally: block ran, written where cancel() cannot undo it.
            self.marker.position.y = 1.0
)";

    // ------------------------------------------------------------------ 8 ---
    //
    // Registration from fixed_update (mid-substep) and from a collision
    // callback. Neither runs any coroutine code where it stands: the body waits
    // for the frame's pump, like every other task.
    const char* kLateScript = R"(
import threepp


class Late:
    def start(self, obj):
        self.marker = threepp.Object3D()
        self.marker.name = "LateState"
        obj.parent.add(self.marker)
        self.fixed = 0
        self.hits = 0

    def fixed_update(self, dt):
        self.fixed += 1
        if self.fixed == 2:
            threepp.editor.start_coroutine(self.from_fixed())

    def on_collision_enter(self, contact):
        self.hits += 1
        if self.hits == 1:
            threepp.editor.start_coroutine(self.from_contact())

    def from_fixed(self):
        yield
        self.marker.position.x = 1.0

    def from_contact(self):
        yield
        self.marker.position.y = 1.0
)";

}// namespace


TEST_CASE("a coroutine's wait() runs on the simulated clock", "[editor][scripting][physx]") {

    Rig rig;
    addScript(rig.scene(), "Waiter", kWaitScript, true);
    rig.start(true);

    // One normal frame: the coroutine body runs to its `yield wait(0.505)`, so
    // the deadline is measured from the end of substep 1.
    rig.run(1);
    auto* mark = rig.marker("Mark");
    REQUIRE(mark);
    CHECK_THAT(mark->position.x, WithinAbs(kFixed, 1e-5));// sim at the mark
    CHECK(mark->position.z == 1.f);                       // frame 1

    // THE case. One second of wall clock in a single frame: PhysxWorld takes 4
    // substeps and discards the rest of the accumulator, so simulated time gains
    // 4/60 s while wall time gains a full second. A wall-paced wait would fire
    // right here.
    rig.frame(1.f);
    auto* fired = rig.marker("Wait");
    REQUIRE(fired);
    CHECK(fired->position.z == 0.f);// nothing written: it has not fired
    CHECK_THAT(scripting::scriptClock().wallTime, WithinAbs(kFixed + 1.0, 1e-5));
    CHECK(scripting::scriptClock().steps == static_cast<std::uint64_t>(1 + kMaxSubSteps));

    // Deadline is 1/60 + 0.505 = 0.52167 s, i.e. substep 32 (0.53333) is the
    // first one past it. Five are done, so 27 more frames — and the pump of the
    // 27th is where it fires.
    rig.run(26);
    CHECK(fired->position.z == 0.f);// substep 31 = 0.51667, still short
    rig.run(1);

    CHECK(fired->position.z == 29.f);                            // frame 1 + 1 + 27
    CHECK_THAT(fired->position.x, WithinAbs(32 * kFixed, 1e-4)); // sim when it fired
    // And the wall clock it did NOT run on had passed the deadline nearly a
    // second earlier: this is the whole difference, in one number.
    CHECK_THAT(fired->position.y, WithinAbs(kFixed + 1.0 + 27 * kFixed, 1e-4));
    CHECK(fired->position.y > 1.4f);

    rig.stop();
}

TEST_CASE("a bare yield resumes exactly once per frame", "[editor][scripting]") {

    constexpr int kFrames = 10;

    Rig rig;
    addScript(rig.scene(), "Ticker", kTickScript);
    rig.start();
    rig.runScriptsOnly(kFrames);

    auto* tick = rig.marker("Tick");
    REQUIRE(tick);
    // The marker is written from update(), BEFORE that frame's pump, so it
    // reports what the pumps of the EARLIER frames did: nine of them, of which
    // the first only ran the body up to its first yield. Hence frames - 2.
    CHECK(tick->position.y == static_cast<float>(kFrames));
    CHECK(tick->position.x == static_cast<float>(kFrames - 2));
    CHECK(tick->position.z == 0.f);// an endless coroutine is never done

    // One more frame, exactly one more resume. Not "roughly one per frame":
    // one.
    rig.runScriptsOnly(1);
    CHECK(tick->position.x == static_cast<float>(kFrames - 1));
    CHECK(rig.errorLines() == 0);

    rig.stop();
}

TEST_CASE("until() sends its value in and sees the settled frame", "[editor][scripting]") {

    Rig rig;
    // Authored in this order, so the flag setter's update() runs first in the
    // sweep — and the catcher's own update() has already run by then either way.
    // The pump is after BOTH, which is what makes a same-frame catch possible.
    addScript(rig.scene(), "Flagger", kFlagScript, false, 1.f);
    addScript(rig.scene(), "Catcher", kCatchScript, false, 2.f);
    rig.start();
    rig.runScriptsOnly(3);

    auto* caught = rig.marker("Caught");
    REQUIRE(caught);
    CHECK(caught->scale.x == 0.f);// the flag is set on frame 4, not before

    rig.runScriptsOnly(1);

    // Same frame the flag was set, and the predicate's truthy value arrived.
    CHECK(caught->scale.x == 4.f);
    CHECK(caught->position.x == 7.f);
    CHECK(caught->position.y == 8.f);
    CHECK(caught->position.z == 9.f);
    CHECK(rig.errorLines() == 0);

    rig.stop();
}

TEST_CASE("a yielded generator runs to completion and returns a value", "[editor][scripting]") {

    Rig rig;
    addScript(rig.scene(), "Nester", kNestScript);
    rig.start();
    rig.runScriptsOnly(8);

    auto* nest = rig.marker("Nest");
    REQUIRE(nest);
    CHECK(nest->position.x == 6.f); // the first child's return value: 2 * 3
    CHECK(nest->position.y == 18.f);// the second's, fed by the first: 6 * 3
    // Frame 1 enters parent() and the first child (pushing costs no frame);
    // the child's two yields land on 2 and 3, so it returns into the parent on
    // frame 3, which immediately enters the second child. Its yields land on 4
    // and 5, and the parent finishes on frame 5.
    CHECK(nest->position.z == 5.f);
    CHECK(rig.errorLines() == 0);

    rig.stop();
}

TEST_CASE("a raise inside a coroutine disables that instance whole", "[editor][scripting]") {

    Rig rig;
    auto boom = addScript(rig.scene(), "Boom", kBoomScript, false, 1.f);
    addScript(rig.scene(), "Survivor", kSurvivorScript, false, 2.f);
    rig.start();

    // The raise is on the coroutine's third resume, which is frame 3.
    rig.runScriptsOnly(3);

    auto* marker = rig.marker("BoomState");
    auto* survivor = rig.marker("SurvivorState");
    REQUIRE(marker);
    REQUIRE(survivor);
    const float frozenFrames = marker->position.x;
    const float frozenSibling = marker->position.y;
    CHECK(frozenFrames == 3.f);

    rig.runScriptsOnly(6);

    // Its update() is dead too — one raise disables the instance, not the
    // method — and so is its OTHER task, which never advanced again.
    CHECK(marker->position.x == frozenFrames);
    CHECK(marker->position.y == frozenSibling);
    // The neighbour never noticed, and kept being pumped.
    CHECK(survivor->position.x >= 8.f);

    // One report, with the traceback, filed against the object it belongs to.
    CHECK(rig.errorLines() == 1);
    CHECK(rig.scripts.errorCount() == 1);
    CHECK_THAT(rig.scripts.errorFor(boom->uuid), ContainsSubstring("coroutine went bang"));
    CHECK_THAT(rig.scripts.errorFor(boom->uuid), ContainsSubstring("Traceback"));

    rig.stop();
}

TEST_CASE("cancel() stops a coroutine and runs its finally", "[editor][scripting]") {

    Rig rig;
    addScript(rig.scene(), "Canceller", kCancelScript);
    rig.start();
    rig.runScriptsOnly(3);

    auto* marker = rig.marker("Cancel");
    REQUIRE(marker);
    // Cancelled from update() on frame 3, before that frame's pump.
    const float resumes = marker->position.x;
    CHECK(resumes == 1.f);
    CHECK(marker->position.y == 1.f);// the finally: block ran

    rig.runScriptsOnly(5);

    // Written by the next update(): no further resumes, and the task says so.
    CHECK(marker->position.x == resumes);
    CHECK(marker->position.z == 1.f);
    CHECK(rig.errorLines() == 0);
    // The cancelled task is reaped by the pump, so nothing is left behind.
    CHECK(liveTasks() == 0);

    rig.stop();
}

TEST_CASE("tasks are session state and die at Stop", "[editor][scripting]") {

    Rig rig;
    addScript(rig.scene(), "Ticker", kTickScript);

    rig.start();
    rig.runScriptsOnly(4);
    CHECK(liveTasks() == 1);
    CHECK(rig.marker("Tick")->position.x == 2.f);

    rig.stop();
    CHECK(liveTasks() == 0);

    // A second Play over the same document starts from nothing.
    rig.dropMarkers({"Tick"});
    rig.start();
    CHECK(liveTasks() == 1);
    rig.runScriptsOnly(3);
    CHECK(rig.marker("Tick")->position.x == 1.f);
    rig.stop();
    CHECK(liveTasks() == 0);

    // And outside any dispatch there is nobody to own a task, so the call
    // refuses rather than registering an orphan. (stop() is NOT the place to
    // test that: a stop() IS a dispatch, and attribution exists there.)
    REQUIRE(scripting::interpreterStarted());
    {
        py::gil_scoped_acquire gil;
        std::string message;
        try {
            py::exec(
                    "import threepp\n"
                    "def _gen():\n"
                    "    yield\n"
                    "threepp.editor.start_coroutine(_gen())\n");
        } catch (py::error_already_set& e) {
            message = scripting::describeError(e);
        }
        CHECK_THAT(message, ContainsSubstring("no script is being run"));
    }
    CHECK(liveTasks() == 0);
}

TEST_CASE("a coroutine can be started from fixed_update and a collision",
          "[editor][scripting][physx]") {

    Rig rig;
    addGround(rig.scene());
    addScript(rig.scene(), "Late", kLateScript, true, 0.7f);
    rig.start(true);

    // Two substeps in, fixed_update registers its task; the pump of that same
    // frame runs it up to its yield, and the frame after resumes it.
    rig.run(3);
    auto* marker = rig.marker("LateState");
    REQUIRE(marker);
    CHECK(marker->position.x == 1.f);
    CHECK(marker->position.y == 0.f);// nothing has touched anything yet

    // Then let it fall the remaining ~0.2 m and land.
    rig.run(60);
    CHECK(marker->position.y == 1.f);
    CHECK(rig.errorLines() == 0);

    rig.stop();
}
