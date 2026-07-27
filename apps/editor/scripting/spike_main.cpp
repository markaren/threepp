// GO/NO-GO spike for embedded Python scripting in the editor.
//
// Proves the one thing the whole feature rests on: an embedded CPython 3.14
// interpreter, served the SAME threepp bindings the editor is linked against,
// running a MonoBehaviour-shaped script against a live scene object.
//
// Superseded by tests/extras/EditorScriptPlay_test.cpp; kept only until that
// exists.

#include "Scripting.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/scenes/Scene.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace threepp;
using namespace threepp::editor;

int main() {

    const auto path = std::filesystem::temp_directory_path() / "threepp_spike_spinner.py";
    {
        std::ofstream out(path);
        out << "class Spinner:\n"
            << "    speed = 1.5\n"
            << "\n"
            << "    def start(self, obj):\n"
            << "        self.obj = obj\n"
            << "\n"
            << "    def update(self, dt):\n"
            << "        self.obj.rotation.y += self.speed * dt\n"
            << "\n"
            << "    def stop(self):\n"
            << "        pass\n";
    }

    std::string error;
    if (!scripting::ensureInterpreter(&error)) {
        std::cout << "[spike] FAIL interpreter: " << error << std::endl;
        return 1;
    }
    std::cout << "[spike] interpreter up" << std::endl;

    const auto inspection = scripting::inspect(path);
    if (!inspection.error.empty()) {
        std::cout << "[spike] FAIL inspect: " << inspection.error << std::endl;
        return 1;
    }
    std::cout << "[spike] class " << inspection.className << ", "
              << inspection.fields.size() << " field(s)" << std::endl;
    for (const auto& field : inspection.fields) {
        std::cout << "[spike]   " << field.name << " = " << field.defaultValue << std::endl;
    }

    auto scene = Scene::create();
    auto group = Group::create();
    group->name = "Spinner";
    scene->add(group);

    ScriptConfig config;
    config.path = path.string();
    config.setField("speed", "2");
    config.write(*group);

    ScriptPlaySession session;
    session.setLogger([](const std::string& message) {
        std::cout << "[spike] log: " << message << std::endl;
    });

    session.start(*scene);
    for (int i = 0; i < 10; ++i) session.update(0.1f);
    const float spun = group->rotation.y;
    session.stop();

    std::cout << "[spike] rotation.y = " << spun << " (expected 2.0)" << std::endl;

    const bool ok = spun > 1.9f && spun < 2.1f && session.errorCount() == 0;
    std::cout << "[spike] " << (ok ? "GO" : "NO-GO") << std::endl;

    std::filesystem::remove(path);
    return ok ? 0 : 1;
}
