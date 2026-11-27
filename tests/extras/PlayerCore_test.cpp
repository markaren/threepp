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
//     anything, and it is not the player's own code — it is the play copy
//     PlayController runs each episode on and Stop drops, which is exactly why
//     it needs a test that would notice
//     if the player ever grew a shortcut around it;
//   * THE EXIT CODE. Nonzero when a script raised, nonzero when nothing ran,
//     zero only when every episode was clean. A gate that goes green because
//     the scene never loaded is worse than no gate;
//   * the script debug-draw list is DRAINED. Nothing else empties it, and
//     ScriptPlaySession switches it on, so a front end that neither draws nor
//     clears it runs into the cap and stays there;
//   * the player runs what the editor runs: a belt conveys, a flock flies, a
//     scan is solid, and a robot that names only its URDF is articulated.

#include <catch2/catch_test_macros.hpp>

#include "PlayerCore.hpp"

#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/ConveyorConfig.hpp"
#endif
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#if defined(THREEPP_EDITOR_WITH_PHYSX) && defined(THREEPP_WITH_VULKAN)
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"
#endif

#include "threepp/extras/fauna/Flock.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

    // The smallest document with weather in it: one Group carrying the snow
    // preset in userData["particles"], and no ParticleField anywhere — the type
    // is never a document node. Authored through the factory, so what the player
    // rebuilds is exactly what the editor's add menu writes.
    std::string particleDocumentJson() {

        SceneDocument authoring;
        auto& scene = authoring.scene();

        auto field = ObjectFactory::createParticleField(scene);
        field->name = "Weather";
        scene.add(field);

        std::string error;
        auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());
        return json;
    }

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // A default conveyor running +x at 1 m/s, with a box set down a hair over
    // its upstream end. The belt surface sits at y = 0.75, so y = 1 rests on it
    // after a substep — the same drop EditorConveyor_test uses against a plain
    // PhysxWorld, authored here as a DOCUMENT so the player has to rebuild it.
    std::string conveyorDocumentJson() {

        SceneDocument authoring;
        auto& scene = authoring.scene();

        auto conveyorNode = ObjectFactory::createConveyor(scene);
        scene.add(conveyorNode);

        auto config = ConveyorConfig::read(*conveyorNode).value();
        config.speed = 1.f;
        config.reverse = false;
        config.write(*conveyorNode);
        config.syncDerived(*conveyorNode);

        auto cargo = Mesh::create(BoxGeometry::create(0.4f, 0.4f, 0.4f),
                                  MeshStandardMaterial::create());
        cargo->name = "Cargo";
        cargo->position.set(-1.2f, 1.f, 0.f);
        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 5.f;
        physics.write(*cargo);
        scene.add(cargo);

        std::string error;
        auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());
        return json;
    }
#endif

    // One flock, authored through the factory as the editor's add menu writes
    // it: a Group carrying userData["flock"]. The birds themselves are session
    // content, added to the play copy and dropped with it.
    std::string flockDocumentJson() {

        SceneDocument authoring;
        auto& scene = authoring.scene();
        scene.add(ObjectFactory::createFlock(scene));

        std::string error;
        auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());
        return json;
    }

    Flock* findFlock(Scene& scene) {

        Flock* found = nullptr;
        scene.traverse([&](Object3D& object) {
            if (!found) found = object.as<Flock>();
        });
        return found;
    }

    // EditorRobotTransplant_test's arm: a base and one revolute link, primitive
    // geometry only, so it needs no mesh files.
    constexpr const char* kArmUrdf = R"(<robot name="arm">
  <link name="base_link">
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
  </link>
  <link name="upper_link">
    <visual><geometry><box size="0.1 0.5 0.1"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="upper_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
</robot>)";

    // A robot as a script writes one: a plain Object3D whose userData names the
    // URDF and the joint values, with no subtree and no joint table
    // (python/examples/spot/editor_deploy/make_scene.py writes its robot this
    // way). A document saved before the articulation block holds the same kind
    // of node, with the subtree below it.
    std::string robotReferenceDocumentJson(const std::filesystem::path& urdf) {

        SceneDocument authoring;
        auto& scene = authoring.scene();

        auto arm = Object3D::create();
        arm->name = "Arm";
        RobotConfig config;
        config.urdf = urdf.string();
        config.joints = {0.5f};
        config.write(*arm);
        scene.add(arm);

        std::string error;
        auto json = authoring.toJson(false, &error);
        REQUIRE(error.empty());
        return json;
    }

#if defined(THREEPP_EDITOR_WITH_PHYSX) && defined(THREEPP_WITH_VULKAN)
    // A scanned slab, 2 m square and 4 cm thick, made solid through the point
    // route: the surface is built straight from the splat centres with no
    // renderer, so a headless run bakes it. A box is set down a little over it.
    // There is no floor, so a box still near the slab after two seconds was
    // caught by the scan's collider; without one it has fallen about 20 m.
    // (Set down, not dropped: a box falling 1 m onto this thin shell goes
    // through it.)
    //
    // Saved as a FILE, not handed over as text: a document references its
    // splats, and a cloud with no source file gets a .ply sidecar beside the
    // saved document, which the player's open re-imports.
    std::filesystem::path saveScanDocument() {

        const auto dir = std::filesystem::temp_directory_path() / "threepp_player_scan";
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
        std::filesystem::create_directories(dir);

        SceneDocument authoring;
        auto& scene = authoring.scene();

        SplatGenerator::Options options;
        options.count = 20000;
        options.seed = 7u;
        options.extent.set(2.f, 0.04f, 2.f);
        options.minScale = 0.01f;
        options.maxScale = 0.02f;
        options.minOpacity = 0.9f;
        options.maxOpacity = 1.f;
        auto cloud = SplatCloud::create(SplatGenerator::generate(options));
        cloud->name = "Scan";
        SplatSurfaceConfig surface;
        surface.enabled = true;
        surface.method = SplatSurfaceConfig::Points;
        surface.write(*cloud);
        scene.add(cloud);

        auto cargo = Mesh::create(BoxGeometry::create(0.2f, 0.2f, 0.2f),
                                  MeshStandardMaterial::create());
        cargo->name = "Cargo";
        cargo->position.set(0.f, 0.3f, 0.f);
        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 1.f;
        physics.write(*cargo);
        scene.add(cargo);

        const auto path = dir / "scan.json";
        std::string error;
        REQUIRE(authoring.saveAs(path, &error));
        REQUIRE(error.empty());
        return path;
    }
#endif

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

#ifdef THREEPP_EDITOR_WITH_PHYSX

TEST_CASE("an episode's seconds are simulated seconds, not stepped wall time", "[player]") {

    // The number --seconds gates on and the log reports as "s sim". At a step
    // of exactly one substep the two definitions coincide; the case that tells
    // them apart is a step LONGER than the world's catch-up cap. A 0.5 s step
    // wants 30 substeps, PhysxWorld takes its capped 4 and discards the rest of
    // the accumulator, so ten such steps are five claimed seconds of which the
    // sim saw forty substeps. `seconds` must report the forty.
    PlayerCore core;
    REQUIRE(core.openJson(documentJson(nullptr)));

    const auto matched = core.runEpisode(0, 60, kFrame);
    CHECK(matched.frames == 60);
    CHECK(std::abs(matched.seconds - 1.f) < 1e-4f);

    const auto hitched = core.runEpisode(1, 10, 0.5f);
    CHECK(hitched.frames == 10);
    CHECK(std::abs(hitched.seconds - 40.f * kFrame) < 1e-4f);

    CHECK(core.exitCode() == 0);
}

#endif

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

#ifdef THREEPP_EDITOR_WITH_PHYSX

TEST_CASE("an authored conveyor conveys under the player, as it does under Play", "[player]") {

    // The session set is the editor's, and this is the assertion that says so
    // for the one session the player used to be missing: for a while it
    // registered physics/animation/sensors/scripts and no conveyor, so an
    // authored belt was inert scenery here — the exact divergence the header
    // promises can never happen, live in the tree.
    //
    // Asserted on the CARGO rather than on conveyorCount, deliberately: a
    // registered session that built no belts would still report a conveyor.
    PlayerCore core;
    std::string error;
    REQUIRE(core.openJson(conveyorDocumentJson(), &error));
    REQUIRE(core.beginEpisode(0, &error));

    auto* cargo = core.scene().getObjectByName("Cargo");
    REQUIRE(cargo != nullptr);
    const float startX = cargo->position.x;

    // Sampled DURING the run: the belt ends at x = 1.5 and cargo that conveys
    // all the way falls off it, which is correct behaviour rather than the
    // thing being asserted.
    bool conveyed = false;
    for (int i = 0; i < 300 && !conveyed; ++i) {
        core.step(kFrame);
        conveyed = cargo->position.x > startX + 0.8f// carried a metre along travel
                   && cargo->position.y > 0.6f      // still on the belt, not fallen through
                   && std::abs(cargo->position.z) < 0.3f;// and not shoved sideways
    }
    CHECK(conveyed);

    const auto result = core.endEpisode();
    CHECK(result.conveyorCount == 1);
    CHECK(result.ok());
}

#endif// THREEPP_EDITOR_WITH_PHYSX

TEST_CASE("an authored particle field is counted even where it cannot be drawn", "[player]") {

    // ParticleField is Vulkan-only by decision, and this test is headless — so
    // the session builds no field at all and says so once. What it must NOT do
    // is report nothing: the count is what the DOCUMENT contains, and a CI log
    // that says zero for a scene full of snow is the regression the counter
    // exists for (see EpisodeResult). The episode has to run clean either way,
    // which is the other half — declining to draw is not a reason to fail.
    PlayerCore core;
    std::string log;
    core.setLogger([&](const std::string& message) { log += message + "\n"; });
    std::string error;
    REQUIRE(core.openJson(particleDocumentJson(), &error));

    const auto result = core.runEpisode(0, 30, kFrame);

    // Said out loud, once per episode: a run that drew no weather has to be
    // distinguishable from a document that had none.
    INFO(log);
    CHECK(log.find("particles: 1 field(s) need the Vulkan backend") != std::string::npos);

    CHECK(result.started);
    CHECK(result.error.empty());
    CHECK(result.particleFieldCount == 1);
    CHECK(result.ok());
    CHECK(core.exitCode() == 0);

    // The authored node is a plain Group carrying one userData entry, and it is
    // the only thing the document ever held: a ParticleField is session content,
    // dropped with the play copy.
    auto* authored = core.scene().getObjectByName("Weather");
    REQUIRE(authored != nullptr);
    CHECK(ParticleFieldConfig::isParticleField(*authored));
    std::size_t fields = 0;
    core.scene().traverse([&](Object3D& object) {
        if (object.type() == "ParticleField") ++fields;
    });
    CHECK(fields == 0);

    // A second episode starts from the same document and counts the same node.
    const auto again = core.runEpisode(1, 30, kFrame);
    CHECK(again.particleFieldCount == 1);
    CHECK(again.ok());
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

TEST_CASE("the player runs the editor's play sessions, in the editor's order", "[player]") {

    // The editor and the player both build their sessions with
    // buildPlaySessions() (PlayRuntime.hpp). This pins what that gives in
    // this build, in update order, behind the same macros the sessions are
    // gated on. Flock and splat surfaces are the two the player once lacked.
    std::vector<std::string> expected;
#ifdef THREEPP_EDITOR_WITH_PHYSX
    expected.insert(expected.end(), {"Physics", "Conveyor"});
#endif
    expected.emplace_back("Particles");
#ifdef THREEPP_EDITOR_WITH_PHYSX
    expected.emplace_back("Granular");
#ifdef THREEPP_WITH_VULKAN
    expected.emplace_back("Splat surfaces");
#endif
    expected.emplace_back("Character");
#endif
    expected.insert(expected.end(), {"Animation", "Flock"});
#ifdef THREEPP_WITH_AUDIO
    expected.emplace_back("Audio");
#endif
    expected.emplace_back("Sensors");
#ifdef THREEPP_EDITOR_WITH_PYTHON
    expected.emplace_back("ScriptPlaySession");
#endif

    PlayerCore core;
    std::vector<std::string> names;
    for (const auto& session : core.controller().sessions()) names.push_back(session->name());
    CHECK(names == expected);
}

TEST_CASE("an authored flock flies under the player, as it does under Play", "[player]") {

    // Asserted on the birds: a Flock in the play copy, ticked once per step,
    // and gone again after Stop.
    PlayerCore core;
    std::string error;
    REQUIRE(core.openJson(flockDocumentJson(), &error));
    CHECK(findFlock(core.scene()) == nullptr);

    REQUIRE(core.beginEpisode(0, &error));
    for (int i = 0; i < 30; ++i) core.step(kFrame);

    auto* flock = findFlock(core.scene());
    REQUIRE(flock != nullptr);
    CHECK(flock->birdCount() == FlockConfig{}.birdCount);
    CHECK(flock->updateCount() == 30);

    const auto result = core.endEpisode();
    CHECK(result.ok());
    CHECK(findFlock(core.scene()) == nullptr);
}

TEST_CASE("a robot that names only its URDF is articulated under the player, as in the editor", "[player]") {

    // The editor re-imports the file when it opens such a node and moves the
    // joint table onto it (rearticulateRobots). Opened without that step, the
    // node stays the empty Object3D it is on disk and the scene has no robot.
    const auto urdf = std::filesystem::temp_directory_path() / "threepp_player_arm.urdf";
    {
        std::ofstream out(urdf);
        out << kArmUrdf;
    }

    PlayerCore core;
    std::string error;
    REQUIRE(core.openJson(robotReferenceDocumentJson(urdf), &error));

    auto* node = core.scene().getObjectByName("Arm");
    REQUIRE(node != nullptr);
    auto* robot = node->as<Robot>();
    REQUIRE(robot != nullptr);
    CHECK(robot->numDOF() == 1);
    CHECK(robot->getObjectByName("upper_link") != nullptr);
    // Posed from the document's joint values, not from the file's rest pose.
    CHECK(std::abs(robot->getJointValue(0) - 0.5f) < 1e-4f);

    const auto result = core.runEpisode(0, 10, kFrame);
    CHECK(result.ok());

    std::error_code ignored;
    std::filesystem::remove(urdf, ignored);
}

#if defined(THREEPP_EDITOR_WITH_PHYSX) && defined(THREEPP_WITH_VULKAN)

TEST_CASE("an authored scan is solid under the player, as it is under Play", "[player]") {

    // Asserted on the cargo, as the conveyor case is: the box is held by the
    // scan. Without the scan's collider nothing stops it.
    PlayerCore core;
    std::string log;
    core.setLogger([&](const std::string& message) { log += message + "\n"; });
    std::string error;
    const auto path = saveScanDocument();
    REQUIRE(core.open(path, &error));
    REQUIRE(core.beginEpisode(0, &error));

    auto* cargo = core.scene().getObjectByName("Cargo");
    REQUIRE(cargo != nullptr);
    for (int i = 0; i < 120; ++i) core.step(kFrame);

    INFO(log);
    CHECK(cargo->position.y > -0.5f);
    CHECK(cargo->position.y < 0.5f);
    // The sensor-only surface the session adds to the play copy.
    CHECK(core.scene().getObjectByName("Scan Surface") != nullptr);

    const auto result = core.endEpisode();
    CHECK(result.ok());

    std::error_code ignored;
    std::filesystem::remove_all(path.parent_path(), ignored);
}

#endif// THREEPP_EDITOR_WITH_PHYSX && THREEPP_WITH_VULKAN
