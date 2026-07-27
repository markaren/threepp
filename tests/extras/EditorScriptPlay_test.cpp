
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
