// The shipped Hover Arena example, driven headlessly through the same sessions
// the editor drives.
//
// This is the acceptance test for a DOCUMENT rather than for a feature, which
// makes it a different kind of test from the ones beside it: nothing here
// builds a scene: it loads the JSON compiled into the binary, exactly as
// File ▸ Open Example does, and then asks whether the thing actually works.
// A shipped example that does not fly is worse than no example at all, and the
// failure mode it protects against is real — every editor feature it composes
// (fixed_update, raycast, the noisy IMU, trigger volumes, script_from_object,
// collision callbacks) can be changed by somebody who has never opened it.
//
// What is asserted, in order of what would embarrass us most:
//
//   * the document parses, and carries what it claims to (a dynamic drone with
//     an IMU and a script, five INVISIBLE trigger gates, a scoreboard);
//   * over 300 substeps the drone HOLDS ITS HEIGHT, hands off, with zero
//     script errors — which means the raycast altimeter, the IMU-damped
//     attitude loop and fixed_update's constant dt are all doing their jobs;
//   * a scripted straight-line flight (the keyboard provider held down, the way
//     EditorScriptLookup_test drives one) crosses a ring, and the SCOREBOARD
//     records it — proving the trigger fired, script_from_object resolved, and
//     the call landed. It is read back off the scene: the ring turns amber and
//     the beacon warms toward gold, which is the same evidence a person gets.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ExampleScenes.hpp"
#include "Scripting.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/ConveyorPlaySession.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SceneSnapshot.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/TextConfig.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // The default PhysxWorld timestep. A frame of exactly this length is one
    // substep, so a frame count is a substep count.
    constexpr float kFixed = 1.f / 60.f;

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
            physics.setLogger([this](const std::string& line) { log.push_back(line); });
        }

        Scene& scene() { return document.scene(); }

        bool open() {
            std::string error;
            const auto json = examples::json("hover-arena");
            return !json.empty() && document.openJson(json, &error);
        }

        void start(bool withPhysics = true) {
            if (withPhysics) {
                physics.start(scene());
                sensors.start(scene());
            }
            playingPhysics = withPhysics;
            scripts.start(scene());
        }

        void run(int frames) {
            for (int i = 0; i < frames; ++i) {
                if (playingPhysics) {
                    physics.update(kFixed);
                    sensors.update(kFixed);
                }
                scripts.update(kFixed);
            }
        }

        bool playingPhysics = true;

        void stop() {
            if (playingPhysics) {
                physics.stop();
                sensors.stop();
            }
            scripts.stop();
        }

        Object3D* node(const char* name) { return scene().getObjectByName(name); }

        [[nodiscard]] Vector3 worldPosition(const char* name) {
            Vector3 position;
            if (auto* object = node(name)) object->getWorldPosition(position);
            return position;
        }

        // The emissive colour a named mesh is currently showing. This is how a
        // C++ test reads what a Python script decided, without reaching into
        // the interpreter: the scripts drive materials, and materials are scene
        // state.
        [[nodiscard]] Color emissiveOf(const char* name) {
            auto* object = node(name);
            auto* withMaterials = object ? object->as<ObjectWithMaterials>() : nullptr;
            if (!withMaterials) return {};
            auto material = std::dynamic_pointer_cast<MeshStandardMaterial>(withMaterials->material());
            return material ? material->emissive : Color{};
        }
    };

}// namespace


TEST_CASE("the shipped Hover Arena document is what it claims to be", "[editor][example]") {

    const auto json = examples::json("hover-arena");
    REQUIRE_FALSE(json.empty());
    // Small enough to read, small enough to diff. The whole point of an example
    // that carries its own generator is that it does not carry a mesh dump.
    REQUIRE(json.size() < 200 * 1024);

    REQUIRE(examples::find("hover-arena") != nullptr);
    REQUIRE(examples::find("no-such-example") == nullptr);
    REQUIRE(examples::json("no-such-example").empty());

    Rig rig;
    REQUIRE(rig.open());
    // Opened from text, so the document has nowhere to save itself back to —
    // which is exactly what makes Save prompt for a path instead of writing
    // over a shipped example.
    REQUIRE_FALSE(rig.document.hasPath());
    REQUIRE_FALSE(rig.document.dirty());

    SECTION("the drone is a scripted, instrumented, dynamic body") {
        auto* drone = rig.node("Drone");
        REQUIRE(drone != nullptr);

        const auto physics = PhysicsConfig::read(*drone);
        REQUIRE(physics.has_value());
        REQUIRE(physics->enabled);
        REQUIRE(physics->body == PhysicsConfig::Body::Dynamic);

        const auto script = ScriptConfig::read(*drone);
        REQUIRE(script.has_value());
        // Inline, not a path: the document has to stand on its own.
        REQUIRE(script->path.empty());
        REQUIRE_FALSE(script->source.empty());
        REQUIRE(script->source.find("fixed_update") != std::string::npos);

        const auto imu = SensorConfig::read(*drone);
        REQUIRE(imu.has_value());
        REQUIRE(imu->type == SensorConfig::Type::Imu);

        auto* lidar = rig.node("Drone Lidar");
        REQUIRE(lidar != nullptr);
        const auto scan = SensorConfig::read(*lidar);
        REQUIRE(scan.has_value());
        REQUIRE(scan->type == SensorConfig::Type::Lidar);
    }

    SECTION("every ring is an invisible trigger over a physics-free torus") {
        for (int i = 1; i <= 5; ++i) {
            const auto label = std::to_string(i);

            auto* gate = rig.node(("Ring " + label + " Gate").c_str());
            REQUIRE(gate != nullptr);
            // Invisible AND cooked: PhysicsPlaySession walks with traverse(),
            // not traverseVisible(), so `visible` is a rendering decision only.
            // The play-session case below is what proves it.
            REQUIRE_FALSE(gate->visible);
            const auto physics = PhysicsConfig::read(*gate);
            REQUIRE(physics.has_value());
            REQUIRE(physics->trigger);
            REQUIRE(physics->body == PhysicsConfig::Body::Static);
            REQUIRE(ScriptConfig::read(*gate).has_value());

            auto* torus = rig.node(("Ring " + label + " Torus").c_str());
            REQUIRE(torus != nullptr);
            // A goal you can crash into is not a goal.
            REQUIRE_FALSE(PhysicsConfig::read(*torus).has_value());
        }
    }

    SECTION("the document says how it wants to be opened") {
        // Two flat strings on the ROOT, read by the editor's open path and by
        // nothing else — an authored vantage plus what the viewport chases.
        // Asserted here because they are part of what the example IS: without
        // them the arena is framed automatically and opens facing a wall.
        auto& root = rig.scene().userData;
        REQUIRE(root.count("editorView") == 1);
        REQUIRE(root.at("editorView").type() == typeid(std::string));
        const auto view = std::any_cast<const std::string&>(root.at("editorView"));
        // "px,py,pz@tx,ty,tz" - behind and above the drone, looking down the
        // course. The editor parses it; this only asks that it is that shape.
        REQUIRE(view.find('@') != std::string::npos);
        REQUIRE(std::count(view.begin(), view.end(), ',') == 4);

        REQUIRE(root.count("editorFollow") == 1);
        REQUIRE(std::any_cast<const std::string&>(root.at("editorFollow")) == "Drone");
        // What it names has to be in the scene, or the chase has nothing to
        // chase and the editor says so instead of doing it.
        REQUIRE(rig.node("Drone") != nullptr);
    }

    SECTION("the arena is generated content, and the rule travels with it") {
        // The generator is on the ROOT, and its output is committed as ordinary
        // saved scene content — opening the document runs nothing.
        REQUIRE(rig.scene().userData.count("generatorSource") == 1);
        REQUIRE(rig.node("Generated") != nullptr);
        REQUIRE(rig.node("Arena Floor") != nullptr);
        REQUIRE(rig.node("Pillar 1") != nullptr);
        // The generated floor carries physics, which is the only reason the
        // drone has anything to hover over.
        const auto floor = PhysicsConfig::read(*rig.node("Arena Floor"));
        REQUIRE(floor.has_value());
        REQUIRE(floor->body == PhysicsConfig::Body::Static);
    }
}

TEST_CASE("Hover Arena hovers, hands off", "[editor][example]") {

    Rig rig;
    REQUIRE(rig.open());
    rig.start();

    // Settle: the controller starts at rest with the altimeter already reading.
    rig.run(60);

    float lowest = 1e9f;
    float highest = -1e9f;
    for (int i = 0; i < 300; ++i) {
        rig.run(1);
        const float y = rig.worldPosition("Drone").y;
        lowest = std::min(lowest, y);
        highest = std::max(highest, y);
    }

    INFO("altitude band " << lowest << " .. " << highest);
    // Authored hover height is 2.2 m over a floor whose top is y = 0. The band
    // is deliberately wide: this is a NOISY loop, and pinning it tighter would
    // be pinning the noise seed rather than the controller.
    REQUIRE(lowest > 1.4f);
    REQUIRE(highest < 3.2f);

    // And it stayed where it was put: no keys are held, so drift is a bug in
    // the horizontal damping, not a style.
    const auto position = rig.worldPosition("Drone");
    REQUIRE(std::abs(position.x - 0.f) < 2.f);
    REQUIRE(std::abs(position.z - 14.f) < 2.f);

    // Not one traceback, from any of the seven instances.
    REQUIRE(rig.scripts.errorCount() == 0);
    REQUIRE(rig.scripts.instanceCount() == 7);

    // The loop really did close on a MEASUREMENT: the IMU produced samples, and
    // the controller's only other rate source is the solver's own number.
    std::size_t imuSamples = 0;
    for (const auto& entry : rig.sensors.entries()) {
        if (entry->config.type == SensorConfig::Type::Imu) imuSamples += entry->samples;
    }
    REQUIRE(imuSamples > 0);

    rig.stop();
}

TEST_CASE("with no physics world Hover Arena explains itself instead of failing",
          "[editor][example]") {

    // The degradation case, stood up the only way a PhysX build can: play the
    // SCRIPTS with no physics session under them. Every physics-shaped lookup
    // then answers None (in a build without the SDK the names are absent
    // instead, which the scripts' getattr guard covers on the same branch), and
    // what must NOT happen is sixty tracebacks a second.
    Rig rig;
    REQUIRE(rig.open());
    rig.start(/*withPhysics*/ false);
    rig.run(120);

    REQUIRE(rig.scripts.errorCount() == 0);
    REQUIRE(rig.scripts.instanceCount() == 7);
    // And the scene is untouched: nothing fell, because nothing was simulating.
    REQUIRE(rig.worldPosition("Drone").y == 2.2f);

    rig.stop();
}

TEST_CASE("flying forward takes a ring, and the scoreboard hears about it", "[editor][example]") {

    Rig rig;
    REQUIRE(rig.open());

    // Hold W, exactly as a hand would. This is the provider EditorApp installs
    // from ImGui; a test installs its own, which is what makes teleop testable
    // at all.
    scripting::keyStateProvider() = [](const std::string& key) { return key == "W"; };
    struct Release {
        ~Release() { scripting::keyStateProvider() = nullptr; }
    } release;

    rig.start();

    const auto beaconBefore = rig.emissiveOf("Beacon");
    const auto ringBefore = rig.emissiveOf("Ring 1 Torus");
    // Cyan while nothing has been taken: more green than red.
    REQUIRE(ringBefore.g > ringBefore.r);

    const float startZ = rig.worldPosition("Drone").z;
    rig.run(900);// 15 s of simulated flight down the course
    const auto position = rig.worldPosition("Drone");

    INFO("drone travelled from z=" << startZ << " to " << position.z << " at y=" << position.y);
    // Ring 1 sits at z = 6, 8 m ahead of the spawn.
    REQUIRE(position.z < startZ - 8.f);
    // Still flying, not scraping or climbing away.
    REQUIRE(position.y > 1.2f);
    REQUIRE(position.y < 4.5f);

    // The ring went amber, which only its on_trigger_enter does.
    const auto ringAfter = rig.emissiveOf("Ring 1 Torus");
    INFO("ring emissive r=" << ringAfter.r << " g=" << ringAfter.g);
    REQUIRE(ringAfter.r > ringAfter.g);

    // And the BEACON warmed, which only the scoreboard does — reached from the
    // ring's script through script_from_object. Without that hop the ring would
    // still flash and this would not move.
    const auto beaconAfter = rig.emissiveOf("Beacon");
    INFO("beacon emissive r " << beaconBefore.r << " -> " << beaconAfter.r);
    REQUIRE(beaconAfter.r > beaconBefore.r + 0.05f);

    REQUIRE(rig.scripts.errorCount() == 0);

    rig.stop();
}


// --- Timber Yard -------------------------------------------------------------
//
// The second shipped example, and a different bet from Hover Arena: nothing
// here is teleoperated. The yard runs ITSELF, out of five inline scripts and
// two coroutines, so what these cases ask is whether the machine works — logs
// off the rack one at a time, along the belt, counted at the far end, with
// nobody touching the keyboard.
//
// Frame COUNTS at a fixed dt, never wall-clock: a test that asserts "within two
// seconds" fails on a loaded machine and passes on a laptop, which this repo
// has been bitten by. Every bound below carries margin, because the point is
// that the yard works, not that it works to the substep.

namespace {

    // Hover Arena's rig plus the belt, which is the one session that example
    // does not need. Registration order is the editor's, and the stop order is
    // its reverse — physics goes down last, since everything else borrows its
    // world.
    struct YardRig {

        SceneDocument document;
        PhysicsPlaySession physics;
        ConveyorPlaySession conveyors;
        PhysxSensorPlaySession sensors;
        ScriptPlaySession scripts;
        std::vector<std::string> log;

        YardRig() {
            conveyors.setPhysics(&physics);
            sensors.setPhysics(&physics);
            scripts.setLogger([this](const std::string& line) { log.push_back(line); });
            physics.setLogger([this](const std::string& line) { log.push_back(line); });
        }

        Scene& scene() { return document.scene(); }

        bool open() {
            std::string error;
            const auto json = examples::json("timber-yard");
            return !json.empty() && document.openJson(json, &error);
        }

        // Play, as the editor plays it: the document is SNAPSHOT first, and the
        // snapshot is what Stop puts back. Without that half, "Stop restores the
        // document" is a claim about a step nothing in the test performed.
        void start() {
            REQUIRE(document.capture(snapshot));
            physics.start(scene());
            conveyors.start(scene());
            sensors.start(scene());
            scripts.start(scene());
        }

        void run(int frames) {
            for (int i = 0; i < frames; ++i) {
                physics.update(kFixed);
                conveyors.update(kFixed);
                sensors.update(kFixed);
                scripts.update(kFixed);
            }
        }

        void stop() {
            scripts.stop();
            sensors.stop();
            conveyors.stop();
            physics.stop();
            REQUIRE(document.restore(snapshot));
        }

        SceneSnapshot snapshot;

        Object3D* node(const char* name) { return scene().getObjectByName(name); }

        [[nodiscard]] Vector3 worldPosition(const char* name) {
            Vector3 position;
            if (auto* object = node(name)) object->getWorldPosition(position);
            return position;
        }

        // Step the yard while WATCHING every log body for signs of leaving the
        // belt. This is the user-visible failure this scene was rebuilt around:
        // a cut that throws a half off the side, or over the kerb rails. The
        // belt is 1.5 m wide and its deck is at y = 0.85, so anything beyond
        // |z| = 1.0 or below y = 0.6 while it is over the belt run has left.
        void runWatching(int frames) {
            for (int i = 0; i < frames; ++i) {
                run(1);
                for (int n = 1; n <= 8; ++n) {
                    const auto label = std::to_string(n);
                    for (const auto& name : {"Log " + label, "Log " + label + " Offcut"}) {
                        const auto p = worldPosition(name.c_str());
                        if (p.x < -6.8f || p.x > 1.6f) continue;// not over the belt
                        strayedZ = std::max(strayedZ, std::abs(p.z));
                        droppedY = std::min(droppedY, p.y);
                    }
                }
            }
        }

        float strayedZ = 0.f;
        float droppedY = 99.f;

        // How many logs are past x — read off the SCENE rather than out of the
        // interpreter, which is the same evidence a person watching gets.
        [[nodiscard]] int logsPast(float x) {
            int count = 0;
            for (int i = 1; i <= 8; ++i) {
                if (worldPosition(("Log " + std::to_string(i)).c_str()).x > x) ++count;
            }
            return count;
        }
    };

    // The gate closes at x = -7.3, the saw stands at -1.5 and the flap that
    // ends the belt is at 1.6. These are the lines that matter, with clearance:
    // past the gate, and past the flap onto the discharge apron.
    constexpr float kPastGate = -6.5f;
    constexpr float kDelivered = 2.2f;

}// namespace


TEST_CASE("the shipped Timber Yard document is what it claims to be", "[editor][example]") {

    const auto json = examples::json("timber-yard");
    REQUIRE_FALSE(json.empty());
    // The ceiling the author tool is written against — it lands at 172 KB.
    // Generated geometry is what blows a document up; the scenery note in
    // TimberYardAuthor.cpp carries the measurements that kept this under it.
    REQUIRE(json.size() < 600 * 1024);

    REQUIRE(examples::find("timber-yard") != nullptr);
    // Both examples ship, in this menu order.
    REQUIRE(examples::all().size() >= 2);
    REQUIRE(examples::all()[0].slug == "hover-arena");
    REQUIRE(examples::all()[1].slug == "timber-yard");

    YardRig rig;
    REQUIRE(rig.open());
    REQUIRE_FALSE(rig.document.hasPath());
    REQUIRE_FALSE(rig.document.dirty());

    SECTION("the gate is a driven, instrumented revolute joint") {
        auto* hinge = rig.node("Gate Hinge");
        REQUIRE(hinge != nullptr);

        const auto joint = JointConfig::read(*hinge);
        REQUIRE(joint.has_value());
        REQUIRE(joint->type == JointConfig::Type::Revolute);
        REQUIRE(joint->body == "Gate Post");
        // "Driven" is not a stored flag — it IS stiffness/damping > 0, which is
        // the condition the play session builds a drive under.
        REQUIRE(joint->stiffness > 0.f);
        REQUIRE(joint->damping > 0.f);

        // The encoder rides the joint node itself, so it names no joint.
        const auto encoder = SensorConfig::read(*hinge);
        REQUIRE(encoder.has_value());
        REQUIRE(encoder->type == SensorConfig::Type::Encoder);

        const auto script = ScriptConfig::read(*hinge);
        REQUIRE(script.has_value());
        REQUIRE(script->path.empty());// inline: the document stands on its own
        REQUIRE(script->source.find("start_coroutine") != std::string::npos);
    }

    SECTION("the stop bar is a breakable joint with a load cell in it") {
        auto* mount = rig.node("Stop Bar Mount");
        REQUIRE(mount != nullptr);

        const auto joint = JointConfig::read(*mount);
        REQUIRE(joint.has_value());
        REQUIRE(joint->breakForce > 0.f);
        REQUIRE(joint->breakTorque > 0.f);

        const auto load = SensorConfig::read(*mount);
        REQUIRE(load.has_value());
        REQUIRE(load->type == SensorConfig::Type::ForceTorque);

        const auto script = ScriptConfig::read(*mount);
        REQUIRE(script.has_value());
        REQUIRE(script->source.find("on_break") != std::string::npos);
    }

    SECTION("the bay is an invisible trigger and the belt is a real conveyor") {
        auto* bay = rig.node("Bay Trigger");
        REQUIRE(bay != nullptr);
        REQUIRE_FALSE(bay->visible);
        const auto physics = PhysicsConfig::read(*bay);
        REQUIRE(physics.has_value());
        REQUIRE(physics->trigger);

        auto* conveyor = rig.node("Conveyor");
        REQUIRE(conveyor != nullptr);
        const auto belt = ConveyorConfig::read(*conveyor);
        REQUIRE(belt.has_value());
        REQUIRE(belt->speed > 0.f);
        // The generated parts ship IN the document: opening it regenerates
        // nothing.
        REQUIRE(ConveyorConfig::derivedGroup(*conveyor) != nullptr);
    }

    SECTION("every log is two halves waiting for the blade") {
        for (int i = 1; i <= 8; ++i) {
            const auto label = std::to_string(i);
            for (const auto& name : {"Log " + label, "Log " + label + " Offcut"}) {
                auto* half = rig.node(name.c_str());
                REQUIRE(half != nullptr);
                const auto physics = PhysicsConfig::read(*half);
                REQUIRE(physics.has_value());
                REQUIRE(physics->body == PhysicsConfig::Body::Dynamic);
            }
            // The cut, waiting to happen: a breakable fixed joint between them.
            auto* cut = rig.node(("Log " + label + " Cut").c_str());
            REQUIRE(cut != nullptr);
            const auto joint = JointConfig::read(*cut);
            REQUIRE(joint.has_value());
            REQUIRE(joint->type == JointConfig::Type::Fixed);
            REQUIRE(joint->body == "Log " + label + " Offcut");
            REQUIRE(joint->breakForce > 0.f);
        }
    }

    SECTION("the saw is a driven body, and the sign says what the yard is") {
        // A motor, not a prop that turns: zero stiffness and a velocity target,
        // which is a drive that holds a SPEED rather than a position.
        auto* spindle = rig.node("Saw Spindle");
        REQUIRE(spindle != nullptr);
        const auto joint = JointConfig::read(*spindle);
        REQUIRE(joint.has_value());
        REQUIRE(joint->type == JointConfig::Type::Revolute);
        REQUIRE(joint->stiffness == 0.f);
        REQUIRE(joint->damping > 0.f);
        REQUIRE(joint->velocity != 0.f);

        auto* blade = rig.node("Saw Blade");
        REQUIRE(blade != nullptr);
        const auto physics = PhysicsConfig::read(*blade);
        REQUIRE(physics.has_value());
        REQUIRE(physics->body == PhysicsConfig::Body::Dynamic);

        // Real lettering, from the editor's own text authoring — and the
        // geometry is baked in, so it reads with no font present.
        auto* title = rig.node("Sign Title");
        REQUIRE(title != nullptr);
        const auto text = TextConfig::read(*title);
        REQUIRE(text.has_value());
        REQUIRE(text->text == "TIMBER YARD");
        REQUIRE(title->geometry() != nullptr);
        REQUIRE(title->geometry()->hasAttribute("position"));
    }

    SECTION("the document says how it wants to be opened") {

        auto& root = rig.scene().userData;
        REQUIRE(root.count("editorView") == 1);
        const auto view = std::any_cast<const std::string&>(root.at("editorView"));
        REQUIRE(view.find('@') != std::string::npos);
        REQUIRE(std::count(view.begin(), view.end(), ',') == 4);
        REQUIRE(root.count("editorFollow") == 1);
        // What it names has to be in the scene, or the chase has nothing to
        // chase and the editor says so instead of doing it.
        REQUIRE(rig.node(std::any_cast<const std::string&>(root.at("editorFollow")).c_str()) !=
                nullptr);
    }
}

TEST_CASE("Timber Yard runs itself, and the saw bay counts", "[editor][example]") {

    YardRig rig;
    REQUIRE(rig.open());
    rig.start();

    // Six instances, live before anything has happened: the gate, the bay, the
    // stop bar, the yard master, the wobble and the saw.
    REQUIRE(rig.scripts.instanceCount() == 6);
    REQUIRE(rig.scripts.errorCount() == 0);
    // The belt built real colliders, or nothing is carried anywhere.
    REQUIRE(rig.conveyors.conveyorCount() == 1);
    REQUIRE(rig.conveyors.beltCount() > 0);
    REQUIRE(rig.logsPast(kPastGate) == 0);

    // A frame COUNT at the fixed step, not a deadline: the yard is authored on
    // the simulated clock, so this is the same amount of yard on every machine,
    // and a loaded CI box gets the same answer as a quiet laptop.
    rig.runWatching(2400);// 40 s simulated, every substep inspected

    const int released = rig.logsPast(kPastGate);
    const int arrived = rig.logsPast(kDelivered);
    INFO("logs past the gate: " << released << ", into the bay: " << arrived);
    // The measured run delivers one log about every 7.5 s, so 40 s is five with
    // room to spare. TWO is the assertion, because two proves the whole chain
    // twice over — the gate coroutine pinned, opened, shut and RE-ARMED, the
    // belt carried, the flap let one through, the trigger fired, the counter
    // counted, and the mission asked for the next one. Everything above two is
    // margin, deliberately.
    REQUIRE(arrived >= 2);
    REQUIRE(released >= arrived);

    // THE SAW CUT THEM. Every log is two halves on a breakable fixed joint, and
    // a joint that has gone is a log that has been through the blade — so this
    // counts the cuts the same way the saw's own script does. At least as many
    // cuts as deliveries, because a log reaches the apron by going through.
    int cuts = 0;
    for (int i = 1; i <= 8; ++i) {
        const auto name = "Log " + std::to_string(i) + " Cut";
        const auto* cut = rig.physics.findJoint(rig.node(name.c_str()));
        REQUIRE(cut != nullptr);
        REQUIRE(cut->joint != nullptr);
        if (cut->joint->broken()) ++cuts;
    }
    INFO("cuts: " << cuts);
    REQUIRE(cuts >= arrived);
    REQUIRE(cuts >= 2);

    // And the halves did not double-count. The bay latches on the name, so
    // eight logs are eight however many bodies cross the volume — this is the
    // assertion that catches a sawn log arriving twice.
    REQUIRE(arrived <= 8);

    // NOTHING LEFT THE BELT. The saw parts a log by breaking a joint under a
    // blade, and the first version of that threw halves across the yard — a
    // fat blade collider means a deep overlap for the solver to push out in the
    // substep the joint happens to snap, and the halves inherit the shove. This
    // is the assertion that would catch it coming back.
    INFO("worst |z| " << rig.strayedZ << ", lowest y " << rig.droppedY);
    REQUIRE(rig.strayedZ < 1.f);
    REQUIRE(rig.droppedY > 0.6f);

    // The stop bar is still on: a mission that runs as authored never reaches
    // its break threshold, which is the whole point of a guard rail.
    const auto* mount = rig.physics.findJoint(rig.node("Stop Bar Mount"));
    REQUIRE(mount != nullptr);
    REQUIRE(mount->joint != nullptr);
    REQUIRE_FALSE(mount->joint->broken());

    // Not one traceback out of six instances.
    REQUIRE(rig.scripts.errorCount() == 0);
    REQUIRE(rig.scripts.instanceCount() == 6);

    // Both joint sensors really spoke. The gate's "am I shut" check reads the
    // encoder, so a silent one would leave the coroutine waiting for ever —
    // which the count above would have caught, but this says WHY.
    std::size_t encoderSamples = 0;
    std::size_t wrenchSamples = 0;
    for (const auto& entry : rig.sensors.entries()) {
        if (entry->config.type == SensorConfig::Type::Encoder) encoderSamples += entry->samples;
        if (entry->config.type == SensorConfig::Type::ForceTorque) wrenchSamples += entry->samples;
    }
    REQUIRE(encoderSamples > 0);
    REQUIRE(wrenchSamples > 0);

    rig.stop();

    // Stop puts the document back — the play snapshot is what makes an example
    // something you can press Play on twice. Every log is on the rack again,
    // and every instance is gone.
    REQUIRE(rig.logsPast(kPastGate) == 0);
    REQUIRE(rig.scripts.instanceCount() == 0);
}


TEST_CASE("holding SPACE overrides the interlock and the rack pours out",
          "[editor][example]") {

    // SPACE skips the hold-back clamp, so the leaf lifts with nothing pinning
    // the pack and the whole rack goes onto the belt at once. That is the
    // override doing what it says.
    //
    // What this case does NOT assert is that the stop bar breaks, and the reason
    // is measured rather than assumed. Once the saw was cutting, the flap stopped
    // being able to tell the two runs apart: an authored mission peaks at 267 N
    // there and a forced pour-out peaks LOWER, at 170 N, because a crowd of
    // half-logs arriving at a flap that is already swinging leans on it less than
    // one log shouldering it open does. Calibrated by outcome the two overlap
    // completely — both survive an 800 N*m limit and both break a 600 N*m one —
    // so there is no threshold that separates them and pinning one would be
    // pinning a coin toss. The joint keeps its break force and its load cell as
    // the guard rail they are; what demonstrates a joint giving way, eight times
    // a run, is the saw.
    scripting::keyStateProvider() = [](const std::string& key) { return key == "SPACE"; };
    struct Release {
        ~Release() { scripting::keyStateProvider() = nullptr; }
    } release;

    YardRig rig;
    REQUIRE(rig.open());
    rig.start();
    REQUIRE(rig.logsPast(kPastGate) == 0);

    rig.runWatching(900);// 15 s simulated: the pour is over long before this

    // The rack emptied itself. Under the authored sequence 15 s is two logs;
    // this is what "skips the interlock" looks like from outside.
    const int released = rig.logsPast(kPastGate);
    INFO("logs past the gate under override: " << released);
    REQUIRE(released >= 6);

    // And the pour-out is contained too: eight logs onto the belt at once, and
    // the kerb rails still hold every half of every one of them.
    INFO("worst |z| " << rig.strayedZ << ", lowest y " << rig.droppedY);
    REQUIRE(rig.strayedZ < 1.f);
    REQUIRE(rig.droppedY > 0.6f);

    // An override is not an error, and a break would not be either: six
    // instances, no tracebacks.
    REQUIRE(rig.scripts.errorCount() == 0);
    REQUIRE(rig.scripts.instanceCount() == 6);

    rig.stop();
}
