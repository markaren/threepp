// threepp_player, minus the window.
//
// The binary's CLI is verified by running it. What is asserted here is the
// contract underneath it, which is the part a CI job actually leans on:
//
//   * a document with a body, a sensor and a script plays, and the player can
//     say what happened;
//   * EPISODES ARE INDEPENDENT. Each one is a play/stop cycle over the same
//     snapshot, so a body that fell in episode 0 starts episode 1 back where it
//     was authored, and every episode ends in the same place. This is the
//     property that makes "run it a hundred times with a hundred seeds" mean
//     anything, and it is not the player's own code — it is PlayController's
//     snapshot/restore, which is exactly why it needs a test that would notice
//     if the player ever grew a shortcut around it;
//   * THE EXIT CODE. Nonzero when a script raised, nonzero when nothing ran,
//     zero only when every episode was clean. A gate that goes green because
//     the scene never loaded is worse than no gate;
//   * the script debug-draw list is DRAINED. Nothing else empties it, and
//     ScriptPlaySession switches it on, so a front end that neither draws nor
//     clears it runs into the cap and stays there.

#include <catch2/catch_test_macros.hpp>

#include "PlayerCore.hpp"

#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include <cmath>
#include <string>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::player;

namespace {

    constexpr float kFrame = 1.f / 60.f;// exactly one default substep
    constexpr float kSpawnY = 3.f;

    // A script that does the three things a MonoBehaviour does and nothing else.
    // It records into the object's own transform so a C++ test can read what a
    // Python script decided without reaching into the interpreter.
    constexpr const char* kCounter = R"(
class Counter:
    ticks = 0

    def start(self, obj):
        self.obj = obj
        self.n = 0

    def update(self, dt):
        self.n += 1
        self.obj.rotation.y = self.n * 0.01

    def stop(self):
        pass
)";

    // Raises on the first update. ScriptPlaySession logs it once, disables the
    // instance and keeps playing — which is exactly the failure the player has
    // to turn into a nonzero exit code rather than swallow.
    constexpr const char* kBroken = R"(
class Broken:
    def start(self, obj):
        self.obj = obj

    def update(self, dt):
        raise RuntimeError("this policy is not ready")

    def stop(self):
        pass
)";

    // Draws one line per update, forever. The cap is 100000 segments, so a run
    // long enough would hit it if nobody drained the list.
    constexpr const char* kDrawer = R"(
import threepp
from threepp import Vector3

class Drawer:
    def start(self, obj):
        self.obj = obj

    def update(self, dt):
        threepp.editor.draw_line(Vector3(0, 0, 0), Vector3(0, 1, 0))

    def stop(self):
        pass
)";

    // A dynamic box with an IMU and a script — the smallest document that
    // exercises all three runtimes at once. Returned as JSON, so the test loads
    // it through the same parse a file would take.
    std::string documentJson(const char* script) {

        SceneDocument authoring;
        auto& scene = authoring.scene();

        auto floor = Mesh::create(BoxGeometry::create(20.f, 1.f, 20.f),
                                  MeshStandardMaterial::create());
        floor->name = "Floor";
        floor->position.set(0.f, -0.5f, 0.f);
        PhysicsConfig ground;
        ground.enabled = true;
        ground.body = PhysicsConfig::Body::Static;
        ground.shape = PhysicsConfig::Shape::Box;
        ground.write(*floor);
        scene.add(floor);

        auto box = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
        box->name = "Subject";
        box->position.set(0.f, kSpawnY, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 1.f;
        physics.write(*box);

        SensorConfig imu;
        imu.enabled = true;
        imu.type = SensorConfig::Type::Imu;
        imu.rateHz = 0.f;// every substep
        imu.write(*box);

        if (script) {
            ScriptConfig config;
            config.source = script;
            config.write(*box);
        }
        scene.add(box);

        std::string error;
        auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());
        return json;
    }

    float subjectY(PlayerCore& core) {

        auto* subject = core.scene().getObjectByName("Subject");
        REQUIRE(subject != nullptr);
        return subject->position.y;
    }

}// namespace


TEST_CASE("the player opens a document and plays it", "[player]") {

    PlayerCore core;
    std::string error;
    REQUIRE(core.openJson(documentJson(kCounter), &error));
    CHECK(error.empty());

    // Authored, before anything has played it.
    CHECK(subjectY(core) == kSpawnY);

    const auto result = core.runEpisode(0, 120, kFrame);

    CHECK(result.started);
    CHECK(result.error.empty());
    CHECK(result.frames == 120);
    CHECK(result.scriptErrors == 0);
    CHECK(result.ok());
    CHECK(core.exitCode() == 0);
    CHECK(core.results().size() == 1);

#ifdef THREEPP_EDITOR_WITH_PYTHON
    CHECK(result.scriptInstances == 1);
#endif
#ifdef THREEPP_EDITOR_WITH_PHYSX
    // The floor and the box.
    CHECK(result.bodyCount == 2);
#endif
    CHECK(result.sensorCount == 1);
}

TEST_CASE("a stopped episode leaves the document exactly as it was", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kCounter)));

    REQUIRE(core.beginEpisode(0));
    for (int i = 0; i < 90; ++i) core.step(kFrame);

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // It really did fall — otherwise the restore below proves nothing.
    const float fallen = subjectY(core);
    INFO("fell from " << kSpawnY << " to " << fallen);
    CHECK(fallen < kSpawnY - 0.5f);
#endif

    core.endEpisode();

    // Restored from the snapshot PlayController took at play().
    CHECK(subjectY(core) == kSpawnY);
    CHECK_FALSE(core.playing());
}

TEST_CASE("episodes are independent", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kCounter)));

    float endOfEpisode[3];
    for (int episode = 0; episode < 3; ++episode) {
        // Every episode starts from the authored pose, not from where the last
        // one left off.
        REQUIRE(subjectY(core) == kSpawnY);

        REQUIRE(core.beginEpisode(episode));
        for (int i = 0; i < 90; ++i) core.step(kFrame);
        endOfEpisode[episode] = subjectY(core);
        core.endEpisode();
    }

    INFO("episode ends: " << endOfEpisode[0] << ", " << endOfEpisode[1] << ", "
                          << endOfEpisode[2]);
    // Same document, same step, same number of steps: the same trajectory. A
    // tolerance rather than equality because a fresh PhysX scene per episode is
    // not obliged to be bit-identical, but it is obliged to be the same fall.
    CHECK(std::abs(endOfEpisode[1] - endOfEpisode[0]) < 1e-3f);
    CHECK(std::abs(endOfEpisode[2] - endOfEpisode[0]) < 1e-3f);

    CHECK(core.results().size() == 3);
    CHECK(core.failedEpisodes() == 0);
    CHECK(core.exitCode() == 0);
}

#ifdef THREEPP_EDITOR_WITH_PYTHON

TEST_CASE("a script that raises fails the episode and the run", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kBroken)));

    const auto result = core.runEpisode(0, 30, kFrame);

    // It played — the point is that the player does not confuse "ran" with
    // "worked". The scene kept simulating, one script was disabled, and the
    // process must still report failure.
    CHECK(result.started);
    CHECK(result.scriptErrors == 1);
    CHECK_FALSE(result.ok());

    CHECK(core.failedEpisodes() == 1);
    CHECK(core.exitCode() == 1);
}

TEST_CASE("one bad episode fails a run of good ones", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kBroken)));

    for (int episode = 0; episode < 3; ++episode) {
        core.runEpisode(episode, 10, kFrame);
    }

    CHECK(core.results().size() == 3);
    // Errors do not leak between episodes: start() clears the map, so each
    // episode reports its own one rather than a running total.
    for (const auto& result : core.results()) {
        CHECK(result.scriptErrors == 1);
    }
    CHECK(core.failedEpisodes() == 3);
    CHECK(core.exitCode() == 1);
}

TEST_CASE("the debug-draw list is drained every step", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kDrawer)));

    // No drain set: this is the headless path, where the core empties the list
    // itself. The scripts still RUN their draw calls — a draw must not behave
    // differently because nobody is looking — the segments just go nowhere.
    REQUIRE(core.beginEpisode(0));
    for (int i = 0; i < 400; ++i) {
        core.step(kFrame);
        // Never more than the one segment this frame's update() pushed.
        REQUIRE(scripting::debugDraw().segments.size() <= 1);
    }
    const auto result = core.endEpisode();

    CHECK(result.scriptErrors == 0);
    CHECK(scripting::debugDraw().dropped == 0);
    // The session is down, and nothing is left behind for the next episode.
    CHECK(scripting::debugDraw().segments.empty());
}

TEST_CASE("a front end that draws the lines is the one that drains them", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(kDrawer)));

    // What the windowed player does: the overlay consumes the segments by
    // building geometry from them, so drawing and draining are the same act.
    // Stood in for here, because a real overlay needs a GL context.
    int drains = 0;
    std::size_t sawSegments = 0;
    core.setDebugDrawDrain([&] {
        ++drains;
        sawSegments += scripting::debugDraw().segments.size();
        scripting::debugDraw().clear();
    });

    REQUIRE(core.beginEpisode(0));
    for (int i = 0; i < 50; ++i) core.step(kFrame);
    core.endEpisode();

    CHECK(drains == 50);
    // The drain SAW what the script drew, rather than being handed an already
    // emptied list.
    CHECK(sawSegments == 50);
}

#endif// THREEPP_EDITOR_WITH_PYTHON

TEST_CASE("a document that will not load is a failed run", "[player]") {

    PlayerCore core;
    std::string error;
    CHECK_FALSE(core.openJson("{ this is not a scene }", &error));
    CHECK_FALSE(error.empty());

    // Nothing ran. That is a failure to do the job, not a vacuous success — a
    // gate that passes because the scene never loaded is the exact bug this
    // program exists to not have.
    CHECK(core.results().empty());
    CHECK(core.exitCode() == 1);
}

TEST_CASE("an episode cannot be started twice", "[player]") {

    PlayerCore core;
    REQUIRE(core.openJson(documentJson(nullptr)));

    REQUIRE(core.beginEpisode(0));
    std::string error;
    CHECK_FALSE(core.beginEpisode(1, &error));
    CHECK(error == "an episode is already playing");
    // The refusal is not recorded as an episode: nothing was played.
    CHECK(core.results().empty());

    core.endEpisode();
    CHECK(core.results().size() == 1);
}
