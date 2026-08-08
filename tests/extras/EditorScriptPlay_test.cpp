
#include <catch2/catch_test_macros.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // One directory for the whole run, cleaned up by the OS. Scripts are
    // written fresh per test, because reloading an edited file is the feature.
    std::filesystem::path scriptDir() {

        static const auto dir = [] {
            auto path = std::filesystem::temp_directory_path() / "threepp_editor_script_test";
            std::filesystem::create_directories(path);
            return path;
        }();
        return dir;
    }

    std::filesystem::path writeScript(const std::string& name, const std::string& source) {

        const auto path = scriptDir() / name;
        std::ofstream out(path, std::ios::trunc);
        out << source;
        out.close();
        return path;
    }

    std::shared_ptr<Group> attach(Scene& scene, const std::filesystem::path& script,
                                  const std::vector<ScriptConfig::Field>& fields = {}) {

        auto group = Group::create();
        group->name = "Spinner";
        ScriptConfig config;
        config.path = script.string();
        config.fields = fields;
        config.write(*group);
        scene.add(group);
        return group;
    }

    std::shared_ptr<Group> attachSource(Scene& scene, const std::string& source,
                                        const std::vector<ScriptConfig::Field>& fields = {}) {

        auto group = Group::create();
        group->name = "Inline";
        ScriptConfig config;
        config.source = source;
        config.fields = fields;
        config.write(*group);
        scene.add(group);
        return group;
    }

    constexpr const char* kSpinner = R"(
class Spinner:
    speed = 1.5

    def start(self, obj):
        self.obj = obj

    def update(self, dt):
        self.obj.rotation.y += self.speed * dt

    def stop(self):
        pass
)";

}// namespace


TEST_CASE("the embedded interpreter starts", "[editor][scripting]") {

    std::string error;
    REQUIRE(scripting::ensureInterpreter(&error));
    CHECK(error.empty());
    CHECK(scripting::interpreterStarted());
}

TEST_CASE("a script's exposed fields are discovered", "[editor][scripting]") {

    const auto path = writeScript("discovered.py", R"(
class Discovered:
    speed = 1.5
    count = 3
    enabled = True
    label = "hello"
    _hidden = 1
    ignored = [1, 2]

    def update(self, dt):
        pass
)");

    const auto inspection = scripting::inspect(path);
    REQUIRE(inspection.error.empty());
    CHECK(inspection.className == "Discovered");

    REQUIRE(inspection.fields.size() == 4);
    // Declaration order, so the inspector reads like the file.
    CHECK(inspection.fields[0].name == "speed");
    CHECK(inspection.fields[0].type == ScriptField::Type::Float);
    CHECK(inspection.fields[0].defaultValue == "1.5");
    CHECK(inspection.fields[1].name == "count");
    CHECK(inspection.fields[1].type == ScriptField::Type::Int);
    CHECK(inspection.fields[1].defaultValue == "3");
    // bool before int: a bool IS an int in Python, and a checkbox is not a
    // drag field.
    CHECK(inspection.fields[2].name == "enabled");
    CHECK(inspection.fields[2].type == ScriptField::Type::Bool);
    CHECK(inspection.fields[2].defaultValue == "1");
    CHECK(inspection.fields[3].name == "label");
    CHECK(inspection.fields[3].type == ScriptField::Type::String);
    CHECK(inspection.fields[3].defaultValue == "hello");
}

TEST_CASE("the class is found by file name, then by update()", "[editor][scripting]") {

    // Named after the file, even with other classes around.
    const auto named = writeScript("mover.py", R"(
class Helper:
    def update(self, dt):
        pass

class Mover:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(named).className == "Mover");

    // No matching name: the single class defining update() wins.
    const auto sole = writeScript("anything.py", R"(
class Config:
    pass

class Behaviour:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(sole).className == "Behaviour");

    // Ambiguous: reported, not guessed.
    const auto ambiguous = writeScript("ambiguous.py", R"(
class One:
    def update(self, dt):
        pass

class Two:
    def update(self, dt):
        pass
)");
    CHECK_FALSE(scripting::inspect(ambiguous).error.empty());
}

TEST_CASE("the file name matches its class through the punctuation",
          "[editor][scripting]") {

    // Python spells files snake_case and classes CamelCase, so stage_look.py
    // holding `class StageLook` is the ORDINARY way to write a script. Matching
    // the two literally makes that pair a non-match.
    //
    // Worth being precise about WHEN that bites, because it is not "always":
    // a file whose single class defines update() has always resolved through
    // the sole-updater fallback, name or no name. The file name only has to be
    // read when something else is competing for the job - and that is exactly
    // when the mismatch turns into "no class named 'stage_look'", about a class
    // sitting right there, correctly named.

    // A competitor that would otherwise win: Helper updates, WristCamera does
    // not, so the sole-updater rule picks Helper and the file name is the only
    // thing that can overrule it.
    const auto beside = writeScript("wrist_camera.py", R"(
class Helper:
    def update(self, dt):
        pass

class WristCamera:
    def start(self, obj):
        pass
)");
    CHECK(scripting::inspect(beside).className == "WristCamera");

    // Two updaters: without the file name this is ambiguous and refuses to
    // run at all. With it, the named one wins - which is the whole purpose of
    // naming a class after its file.
    const auto contested = writeScript("stage_look.py", R"(
class Helper:
    def update(self, dt):
        pass

class StageLook:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(contested).className == "StageLook");
    CHECK(scripting::inspect(contested).error.empty());

    // Hyphens too - a file name is not a Python identifier and never had to be.
    const auto kebab = writeScript("stage-look.py", R"(
class Helper:
    def update(self, dt):
        pass

class StageLook:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(kebab).className == "StageLook");

    // The fold is symmetric: a snake_case CLASS in a camelCase file matches.
    const auto reversed = writeScript("stageLook.py", R"(
class Helper:
    def update(self, dt):
        pass

class stage_look:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(reversed).className == "stage_look");

    // And the plain single-class file keeps working, which is the case the
    // fold must not disturb.
    const auto lone = writeScript("lone_worker.py", R"(
class LoneWorker:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(lone).className == "LoneWorker");
}

TEST_CASE("a setup script needs no update()", "[editor][scripting]") {

    // A script that only implements start() is a real thing - it dresses the
    // scene once and gets out of the way - and it used to be unreachable
    // unless its file name matched its class exactly, because the fallback
    // looked for update() and nothing else.
    const auto sole = writeScript("unrelated_name.py", R"(
class Dressing:
    def start(self, obj):
        obj.position.y = 3.0
)");
    CHECK(scripting::inspect(sole).className == "Dressing");
    CHECK(scripting::inspect(sole).error.empty());

    // Strictly a last resort, though: it is consulted only where the update()
    // rule found NOTHING, so it can never take a resolution away from a class
    // that does update. Here Behaviour still wins over the bare starter.
    const auto mixed = writeScript("mixed_roles.py", R"(
class Dressing:
    def start(self, obj):
        pass

class Behaviour:
    def update(self, dt):
        pass
)");
    CHECK(scripting::inspect(mixed).className == "Behaviour");

    // Two starters and no updater is as ambiguous as two updaters.
    const auto both = writeScript("two_starters.py", R"(
class One:
    def start(self, obj):
        pass

class Two:
    def start(self, obj):
        pass
)");
    CHECK_FALSE(scripting::inspect(both).error.empty());
}

TEST_CASE("an unmatched class name is reported with one that would match",
          "[editor][scripting]") {

    // The error a typo actually produces. Naming the classes it DID find, and
    // a spelling that would have worked, is the difference between a minute
    // and an afternoon - the old message said only that 'stage_look' was
    // missing, which is the one name nobody would ever type.
    const auto typo = writeScript("stage_look2.py", R"(
class StageLuke:
    def start(self, obj):
        pass

class Bystander:
    def start(self, obj):
        pass
)");
    const auto reported = scripting::inspect(typo).error;
    REQUIRE_FALSE(reported.empty());
    CHECK(reported.find("StageLuke") != std::string::npos);
    CHECK(reported.find("Bystander") != std::string::npos);
    CHECK(reported.find("StageLook2") != std::string::npos);

    // Two classes whose names differ ONLY by punctuation: the fold makes them
    // one name, and picking either would be a coin toss.
    const auto folded = writeScript("twin_names.py", R"(
class TwinNames:
    def update(self, dt):
        pass

class Twinnames:
    def update(self, dt):
        pass
)");
    CHECK_FALSE(scripting::inspect(folded).error.empty());
}

TEST_CASE("a failed URDF load tells a script why", "[editor][scripting]") {

    // The bindings used to raise "URDFLoader: failed to parse URDF XML" and
    // drop everything the parser had worked out. That is the wrong end of the
    // trade for the case this exists to serve: a ROS node handed
    // /robot_description off a topic, where the XML never touched a disk and
    // the exception text IS the whole diagnosis.
    //
    // Findings ride the object's SCALE, as elsewhere in these tests - nothing
    // else writes to it, so a start() that never ran cannot read as a pass.
    const auto path = writeScript("urdf_diag.py", R"(
import threepp


class UrdfDiag:
    def start(self, obj):
        loader = threepp.URDFLoader()

        # Malformed XML through parse(), which is the /robot_description shape.
        try:
            loader.parse(".", "<robot name='x'><link")
            obj.scale.x = -1.0
        except RuntimeError as exc:
            # The parser's own words, not a generic sentence. 2.0 for the real
            # reason, -2.0 for a raise that carried nothing.
            obj.scale.x = 2.0 if "parse the document" in str(exc) else -2.0

        # The same account is readable afterwards rather than only thrown.
        obj.scale.y = 2.0 if loader.last_error and len(loader.diagnostics) == 1 else -1.0

        # And a xacro failure carries the FILE AND LINE, which is the whole
        # difference between a fix and a guess.
        try:
            loader.parse(".", "<robot name='x' xmlns:xacro='http://www.ros.org/wiki/xacro'>"
                              "<link name='${nope}'/></robot>")
            obj.scale.z = -1.0
        except RuntimeError as exc:
            obj.scale.z = 2.0 if "nope" in str(exc) else -2.0

    def update(self, dt):
        pass
)");

    auto scene = Scene::create();
    auto probe = attach(*scene, path);

    ScriptPlaySession scripts;
    scripts.start(*scene);
    CHECK(scripts.errorFor(probe->uuid).empty());

    CHECK(probe->scale.x == 2.f);// the raise carries the parser's reason
    CHECK(probe->scale.y == 2.f);// last_error / diagnostics are readable
    CHECK(probe->scale.z == 2.f);// and a xacro failure names what it could not resolve

    scripts.stop();
}

TEST_CASE("a URDF that loads can still have something to say", "[editor][scripting]") {

    // The half `last_error` cannot express. A xacro that expands FINE still
    // warns - a redefined macro, an attribute a macro never declared - and
    // those warnings only ever went to stderr, where a script cannot see them
    // and a GUI has nowhere to put them.
    const auto path = writeScript("urdf_warn.py", R"(
import threepp


class UrdfWarn:
    def start(self, obj):
        loader = threepp.URDFLoader()
        xml = (
            "<robot name='w' xmlns:xacro='http://www.ros.org/wiki/xacro'>"
            "<xacro:macro name='thing' params='size'><link name='l${size}'/></xacro:macro>"
            "<xacro:macro name='thing' params='size'><link name='m${size}'/></xacro:macro>"
            "<xacro:thing size='1' colour='red'/>"
            "<link name='base'/>"
            "</robot>"
        )
        robot = loader.parse(".", xml)          # succeeds
        obj.scale.x = 2.0 if robot is not None else -1.0
        # Succeeded, so there is no error...
        obj.scale.y = 2.0 if loader.last_error == "" else -1.0
        # ...and yet there is plenty to say.
        warnings = [d for d in loader.diagnostics if "warning" in d]
        obj.scale.z = 2.0 if len(warnings) >= 2 else -1.0

    def update(self, dt):
        pass
)");

    auto scene = Scene::create();
    auto probe = attach(*scene, path);

    ScriptPlaySession scripts;
    scripts.start(*scene);
    CHECK(scripts.errorFor(probe->uuid).empty());

    CHECK(probe->scale.x == 2.f);// it loaded
    CHECK(probe->scale.y == 2.f);// with no error
    CHECK(probe->scale.z == 2.f);// and warnings all the same

    scripts.stop();
}

TEST_CASE("ScriptPlaySession runs start/update/stop", "[editor][scripting]") {

    const auto path = writeScript("spinner.py", kSpinner);

    auto scene = Scene::create();
    auto group = attach(*scene, path);

    ScriptPlaySession session;
    session.start(*scene);
    CHECK(session.errorCount() == 0);
    CHECK(session.instanceCount() == 1);

    for (int i = 0; i < 10; ++i) session.update(0.1f);

    // The class default, since the document overrides nothing.
    CHECK(group->rotation.y > 1.4f);
    CHECK(group->rotation.y < 1.6f);

    session.stop();
    CHECK(session.instanceCount() == 0);

    // Stopping does not put the scene back — the editor's snapshot does that,
    // and a session is explicitly allowed to leave a mess.
    CHECK(group->rotation.y > 1.4f);
}

TEST_CASE("authored field values override the class defaults", "[editor][scripting]") {

    const auto path = writeScript("spinner.py", kSpinner);

    auto scene = Scene::create();
    auto group = attach(*scene, path, {{"speed", "3"}});

    ScriptPlaySession session;
    session.start(*scene);
    for (int i = 0; i < 10; ++i) session.update(0.1f);
    session.stop();

    CHECK(group->rotation.y > 2.9f);
    CHECK(group->rotation.y < 3.1f);
}

TEST_CASE("editing the file and playing again reloads it", "[editor][scripting]") {

    const auto path = writeScript("reloaded.py", R"(
class Reloaded:
    def start(self, obj):
        obj.position.x = 1.0

    def update(self, dt):
        pass
)");

    auto scene = Scene::create();
    auto group = attach(*scene, path);

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();
    CHECK(group->position.x == 1.f);

    // Same path, different behaviour. Play must pick it up — module state is
    // purged before every load, and this is the core of the workflow.
    writeScript("reloaded.py", R"(
class Reloaded:
    def start(self, obj):
        obj.position.x = 7.0

    def update(self, dt):
        pass
)");

    session.start(*scene);
    session.stop();
    CHECK(group->position.x == 7.f);
}

TEST_CASE("all three methods are optional", "[editor][scripting]") {

    const auto onlyUpdate = writeScript("onlyupdate.py", R"(
class OnlyUpdate:
    def update(self, dt):
        pass
)");
    const auto onlyStart = writeScript("onlystart.py", R"(
class OnlyStart:
    def start(self, obj):
        obj.position.y = 2.0
)");

    auto scene = Scene::create();
    auto a = attach(*scene, onlyUpdate);
    auto b = attach(*scene, onlyStart);

    ScriptPlaySession session;
    session.start(*scene);
    session.update(0.1f);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(b->position.y == 2.f);
}

TEST_CASE("a script gets its object as the concrete leaf type", "[editor][scripting]") {

    // Mesh derives from Object3D VIRTUALLY, and pybind11 corrupts the heap when
    // an inherited std::string member is written through an Object3D-typed
    // handle. This is the regression test for handing out the concrete type.
    const auto path = writeScript("renamer.py", R"(
import threepp

class Renamer:
    def start(self, obj):
        assert isinstance(obj, threepp.Mesh), type(obj).__name__
        obj.name = "renamed-by-a-script-with-a-deliberately-long-name"
        obj.position.z = 4.0
)");

    auto scene = Scene::create();
    auto mesh = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
    mesh->name = "Box";
    ScriptConfig config;
    config.path = path.string();
    config.write(*mesh);
    scene->add(mesh);

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(mesh->name == "renamed-by-a-script-with-a-deliberately-long-name");
    CHECK(mesh->position.z == 4.f);
}

TEST_CASE("the handle shares ownership of the object", "[editor][scripting]") {

    const auto path = writeScript("keeper.py", R"(
class Keeper:
    def start(self, obj):
        self.obj = obj

    def update(self, dt):
        pass
)");

    auto scene = Scene::create();
    auto group = attach(*scene, path);

    const auto before = group.use_count();

    ScriptPlaySession session;
    session.start(*scene);
    // A script that stashes its object holds a reference to it, not a pointer
    // into a graph the editor is about to throw away.
    CHECK(group.use_count() > before);

    session.stop();
    CHECK(group.use_count() == before);
}

TEST_CASE("a script that raises does not stop the session", "[editor][scripting]") {

    const auto bad = writeScript("thrower.py", R"(
class Thrower:
    def update(self, dt):
        raise RuntimeError("every single frame")
)");
    const auto good = writeScript("spinner.py", kSpinner);

    auto scene = Scene::create();
    auto broken = attach(*scene, bad);
    broken->name = "Broken";
    auto working = attach(*scene, good);

    int logged = 0;
    ScriptPlaySession session;
    session.setLogger([&logged](const std::string&) { ++logged; });

    session.start(*scene);
    for (int i = 0; i < 30; ++i) session.update(0.1f);
    session.stop();

    // One report for the failing script, no matter how many frames it ran.
    CHECK(session.errorCount() == 1);
    CHECK_FALSE(session.errorFor(broken->uuid).empty());
    CHECK(session.errorFor(working->uuid).empty());
    // The traceback is what makes an error actionable.
    CHECK(session.errorFor(broken->uuid).find("every single frame") != std::string::npos);
    // "scripts running: 2 instances" plus exactly one error line.
    CHECK(logged == 2);

    // The healthy script kept running for all thirty frames.
    CHECK(working->rotation.y > 4.f);
}

TEST_CASE("broken scripts are reported, not fatal", "[editor][scripting]") {

    auto scene = Scene::create();

    auto a = attach(*scene, writeScript("syntax.py", "class Syntax:\n    def update(self dt):\n        pass\n"));
    auto b = attach(*scene, writeScript("noclass.py", "SPEED = 3\n"));
    auto c = attach(*scene, writeScript("raises_at_import.py", "raise ValueError('boom')\n"));
    auto d = attach(*scene, writeScript("badstart.py", R"(
class Badstart:
    def start(self, obj):
        obj.no_such_attribute()

    def update(self, dt):
        pass
)"));
    auto missing = attach(*scene, scriptDir() / "not_written.py");

    ScriptPlaySession session;
    session.start(*scene);
    session.update(0.1f);
    session.stop();

    CHECK(session.errorCount() == 5);
    for (const auto* object : {a.get(), b.get(), c.get(), d.get(), missing.get()}) {
        CHECK_FALSE(session.errorFor(object->uuid).empty());
    }
}

TEST_CASE("inline source drives an object like a file does", "[editor][scripting]") {

    auto scene = Scene::create();
    // Same class, no file anywhere: the source lives in the document.
    auto group = attachSource(*scene, kSpinner, {{"speed", "3"}});

    ScriptPlaySession session;
    session.start(*scene);
    CHECK(session.errorCount() == 0);
    CHECK(session.instanceCount() == 1);

    for (int i = 0; i < 10; ++i) session.update(0.1f);
    session.stop();

    CHECK(group->rotation.y > 2.9f);
    CHECK(group->rotation.y < 3.1f);
}

TEST_CASE("inline source is compiled fresh on every play", "[editor][scripting]") {

    auto scene = Scene::create();
    auto group = attachSource(*scene, R"(
class Inline:
    def start(self, obj):
        obj.position.x = 1.0
)");

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();
    CHECK(group->position.x == 1.f);

    // Editing inline source IS the hot reload: there is no file, no cache and
    // no stamp to compare — the next Play compiles whatever the document says.
    ScriptConfig edited;
    edited.source = R"(
class Inline:
    def start(self, obj):
        obj.position.x = 7.0
)";
    edited.write(*group);

    session.start(*scene);
    session.stop();
    CHECK(group->position.x == 7.f);
}

TEST_CASE("inline scripts get their own module per object", "[editor][scripting]") {

    // Two objects, two different inline scripts, both defining a class with the
    // same name. A shared module would leave the second overwriting the first.
    auto scene = Scene::create();
    auto a = attachSource(*scene, R"(
class Behaviour:
    def start(self, obj):
        obj.position.x = 1.0
)");
    auto b = attachSource(*scene, R"(
class Behaviour:
    def start(self, obj):
        obj.position.x = 2.0
)");

    ScriptPlaySession session;
    session.start(*scene);
    session.stop();

    CHECK(session.errorCount() == 0);
    CHECK(a->position.x == 1.f);
    CHECK(b->position.x == 2.f);
}

TEST_CASE("inline fields are discovered from the source", "[editor][scripting]") {

    const auto inspection = scripting::inspectSource(R"(
class Discovered:
    speed = 1.5
    count = 3
    enabled = True
    label = "hello"
    _hidden = 1

    def update(self, dt):
        pass
)",
                                                     "uuid-1234", "Box");
    REQUIRE(inspection.error.empty());
    CHECK(inspection.className == "Discovered");
    REQUIRE(inspection.fields.size() == 4);
    CHECK(inspection.fields[0].name == "speed");
    CHECK(inspection.fields[0].type == ScriptField::Type::Float);
    CHECK(inspection.fields[2].type == ScriptField::Type::Bool);
    CHECK(inspection.fields[3].defaultValue == "hello");
}

TEST_CASE("an inline class is found without a file name to match", "[editor][scripting]") {

    // One class, whatever it is called and whether or not it updates: there is
    // nothing to be ambiguous about.
    CHECK(scripting::inspectSource("class Anything:\n    def start(self, obj):\n        pass\n",
                                   "k", "Box")
                  .className == "Anything");

    // Several classes: the one defining update() is the behaviour.
    CHECK(scripting::inspectSource("class Config:\n    pass\n\n"
                                   "class Behaviour:\n    def update(self, dt):\n        pass\n",
                                   "k", "Box")
                  .className == "Behaviour");

    // Ambiguous, and no file name to break the tie: reported, not guessed.
    CHECK_FALSE(scripting::inspectSource("class One:\n    def update(self, dt):\n        pass\n\n"
                                         "class Two:\n    def update(self, dt):\n        pass\n",
                                         "k", "Box")
                        .error.empty());

    // No class at all.
    CHECK_FALSE(scripting::inspectSource("SPEED = 3\n", "k", "Box").error.empty());
}

TEST_CASE("broken inline source is recorded, not fatal", "[editor][scripting]") {

    auto scene = Scene::create();
    // A syntax error, a runtime failure at import, and one that raises every
    // frame — beside a healthy script that has to keep running.
    auto syntax = attachSource(*scene, "class Broken:\n    def update(self dt):\n        pass\n");
    auto imports = attachSource(*scene, "raise ValueError('boom')\n");
    auto raises = attachSource(*scene, "class Thrower:\n"
                                       "    def update(self, dt):\n"
                                       "        raise RuntimeError('every single frame')\n");
    auto working = attachSource(*scene, kSpinner);

    ScriptPlaySession session;
    session.start(*scene);
    for (int i = 0; i < 20; ++i) session.update(0.1f);
    session.stop();

    CHECK(session.errorCount() == 3);
    for (const auto* object : {syntax.get(), imports.get(), raises.get()}) {
        CHECK_FALSE(session.errorFor(object->uuid).empty());
    }
    // The traceback names the object the source belongs to, since there is no
    // file name to point at.
    CHECK(session.errorFor(raises->uuid).find("<inline:Inline>") != std::string::npos);
    CHECK(session.errorFor(working->uuid).empty());
    CHECK(working->rotation.y > 2.5f);
}

TEST_CASE("a syntax check compiles without running anything", "[editor][scripting]") {

    CHECK(scripting::checkSyntax(kSpinner, "Box").empty());

    // The line number is the whole point: it is what the editor shows in red.
    const auto error = scripting::checkSyntax("class Broken:\n    def update(self dt):\n        pass\n", "Box");
    CHECK_FALSE(error.empty());
    CHECK(error.find("line 2") != std::string::npos);

    // Valid syntax that would blow up (or fire a missile) when executed still
    // passes the check — nothing here runs.
    CHECK(scripting::checkSyntax("raise SystemExit(1)\n", "Box").empty());
}

TEST_CASE("a scene with no scripts costs nothing", "[editor][scripting]") {

    auto scene = Scene::create();
    scene->add(Group::create());

    ScriptPlaySession session;
    session.start(*scene);
    session.update(0.1f);
    session.stop();

    CHECK(session.instanceCount() == 0);
    CHECK(session.errorCount() == 0);
}

TEST_CASE("debug draw lands in the seam and dies with the session", "[editor][scripting]") {

    // The seam is scripting::debugDraw() - the bindings decompose every call to
    // line segments there, and the editor drains it into the overlay. Headless
    // there is no drainer, which is exactly what lets a test read the list.
    auto& list = scripting::debugDraw();

    auto scene = Scene::create();
    attachSource(*scene, R"(
import threepp

editor = threepp.editor


class Sketcher:
    def start(self, obj):
        # Colours: default white, hex int, and a threepp.Color - all accepted.
        editor.draw_line(threepp.Vector3(0, 0, 0), threepp.Vector3(1, 0, 0))
        editor.draw_line(threepp.Vector3(0, 0, 0), threepp.Vector3(0, 1, 0), 0xff0000)
        editor.draw_ray(threepp.Vector3(0, 2, 0), threepp.Vector3(0, -1, 0), 2.0,
                        threepp.Color(0.0, 0.0, 1.0))

    def update(self, dt):
        editor.draw_point(threepp.Vector3(5, 5, 5))          # 3 segments
        editor.draw_box(threepp.Vector3(0, 0, 0),
                        threepp.Vector3(2, 2, 2), 0x00ff00)  # 12 segments
        editor.draw_sphere(threepp.Vector3(0, 0, 0), 1.0)    # 72 segments
        # draw_axes wants an Object3D; the scene root is one that always exists.
        editor.draw_axes(editor.scene(), 1.0)                # 3 segments
)");

    // Nothing playing: the binding is a no-op, not a raise, and nothing lands.
    CHECK_FALSE(list.active);
    CHECK(list.segments.empty());

    ScriptPlaySession session;
    session.start(*scene);
    CHECK(session.errorCount() == 0);

    // start() drew three lines; the session activated the seam before phase 2.
    CHECK(list.active);
    REQUIRE(list.segments.size() == 3);

    // Default white, hex red, Color blue - per-segment colour survives.
    CHECK(list.segments[0].r == 1.f);
    CHECK(list.segments[0].g == 1.f);
    CHECK(list.segments[1].r == 1.f);
    CHECK(list.segments[1].g == 0.f);
    CHECK(list.segments[2].b == 1.f);
    CHECK(list.segments[2].r == 0.f);
    // draw_ray scaled the direction: from (0,2,0) two metres straight down,
    // origin stored first, computed endpoint second.
    CHECK(list.segments[2].ay == 2.f);
    CHECK(list.segments[2].by == 0.f);

    // One update: point 3 + box 12 + sphere 72 + axes 3 on top of start's 3.
    session.update(0.016f);
    CHECK(session.errorCount() == 0);
    CHECK(list.segments.size() == 3 + 3 + 12 + 72 + 3);

    // The editor drains between frames; headless the test plays that part.
    list.clear();
    session.update(0.016f);
    CHECK(list.segments.size() == 3 + 12 + 72 + 3);

    // Stop takes the seam down with the session: inactive AND empty, so the
    // first frame of the next Play cannot render leftovers.
    session.stop();
    CHECK_FALSE(list.active);
    CHECK(list.segments.empty());

    // And a draw after stop is the headless no-op again.
    session.start(*scene);
    session.stop();
    CHECK(list.segments.empty());
}

TEST_CASE("a script can load a texture and light the scene with it", "[editor][scripting]") {

    // The editor's module is assembled from a SUBSET of the wheel's binding TUs, and
    // bind_loaders.cpp was not in it. `scene.environment` was writable the whole time; there was
    // simply nothing in the module that could produce a Texture to assign, and no constructor
    // from data either - so the only way to light a scene with an HDRI was the File menu, which
    // a script and a headless --screenshot run cannot reach.
    //
    // This pins the wiring rather than the pixels: if bind_loaders.cpp ever falls out of
    // apps/editor/CMakeLists.txt again, or init_loaders is dropped from ScriptModule.cpp, these
    // names go missing and the failure is an AttributeError deep inside somebody's scene script.
    const std::string source = R"PY(
import threepp
from threepp import editor


class Probe:

    def start(self, obj):
        # RAISE rather than record: a missing name has to fail the session, or this test passes
        # with the bindings gone (threepp.Texture comes from bind_textures.cpp, which never left).
        missing = [n for n in ("TextureLoader", "RGBELoader", "EXRLoader", "ModelLoader")
                   if not hasattr(threepp, n)]
        if missing:
            raise RuntimeError("threepp module is missing loaders: " + ", ".join(missing))
        # The assignment the loaders exist to serve. An empty Texture is enough to prove the
        # property accepts one - the file decode is ImageLoader's business, tested elsewhere.
        editor.scene().environment = threepp.Texture()

    def update(self, dt):
        pass
)PY";

    auto scene = Scene::create();
    attachSource(*scene, source);

    ScriptPlaySession session;
    session.start(*scene);
    CHECK(session.errorCount() == 0);
    REQUIRE(session.instanceCount() == 1);
    session.update(0.016f);

    CHECK(scene->environment != nullptr);

    session.stop();
}
