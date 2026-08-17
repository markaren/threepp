
#include "threepp/extras/editor/ScriptWorkspace.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::string jsonEscaped(const std::string& text) {

        std::string out;
        out.reserve(text.size() + 8);
        for (const char c : text) {
            if (c == '"' || c == '\\') out.push_back('\\');
            out.push_back(c);
        }
        return out;
    }

}// namespace


std::string ScriptWorkspace::normalize(const std::string& text) {

    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == '\t') {
            out.append(4, ' ');
        } else if (c != '\r') {
            out.push_back(c);
        }
    }
    return out;
}

std::string ScriptWorkspace::settingsJson(const std::filesystem::path& stubs) {

    // generic_string(): forward slashes, so the file reads the same whoever
    // wrote it and no escaping is needed for the separators.
    const auto path = jsonEscaped(stubs.generic_string());

    std::ostringstream out;
    out << "{\n"
        << "    // Written by the threepp editor the first time it opened a script here.\n"
        << "    // It is created only when it is absent, so anything you change below is\n"
        << "    // safe - delete the file and the editor writes a fresh one.\n"
        << "\n"
        << "    // Where Pylance finds the threepp API: the stub package under the source\n"
        << "    // tree (python/threepp/threepp/__init__.pyi). Absolute, because scripts\n"
        << "    // live wherever you keep them.\n"
        << "    \"python.analysis.stubPath\": \"" << path << "\",\n"
        << "\n"
        << "    // Those stubs describe a module compiled into the editor itself, so there\n"
        << "    // is no .py file beside them and Pylance would otherwise mark every\n"
        << "    // `import threepp` as unresolved-from-source.\n"
        << "    \"python.analysis.diagnosticSeverityOverrides\": {\n"
        << "        \"reportMissingModuleSource\": \"none\"\n"
        << "    },\n"
        << "\n"
        << "    // Point this at your own interpreter to also complete what IT has - numpy,\n"
        << "    // rclpy from a RoboStack environment, and so on. Scripts still RUN in the\n"
        << "    // editor's own embedded interpreter; this only changes what VS Code knows.\n"
        << "    // \"python.defaultInterpreterPath\": \"C:/Users/you/mambaforge/envs/robostack/python.exe\",\n"
        << "\n"
        << "    // The editor stores script source LF-terminated and space-indented, and\n"
        << "    // normalizes anything else on the way back in. Matching that here keeps a\n"
        << "    // save from looking like an edit to every line in the file.\n"
        << "    \"files.eol\": \"\\n\",\n"
        << "    \"editor.insertSpaces\": true,\n"
        << "    \"editor.tabSize\": 4\n"
        << "}\n";
    return out.str();
}

ScriptWorkspace::Result ScriptWorkspace::ensure(const std::filesystem::path& dir,
                                                const std::filesystem::path& stubs) {

    Result result;
    result.file = dir / ".vscode" / "settings.json";

    std::error_code ec;
    std::filesystem::create_directories(result.file.parent_path(), ec);
    if (ec) {
        result.error = ec.message();
        return result;
    }

    // The one rule: never overwrite. A user who has dialled their own
    // interpreter, linter or formatter into this file has said something, and
    // opening another script is not a reason to take it back.
    if (std::filesystem::exists(result.file, ec)) {
        result.ok = true;
        return result;
    }

    if (!writeSource(result.file, settingsJson(stubs))) {
        result.error = "could not write " + result.file.string();
        return result;
    }

    result.ok = true;
    result.created = true;
    return result;
}

std::filesystem::path ScriptWorkspace::scratchDir() {

    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = std::filesystem::current_path();
    return dir / "threepp-editor" / "scripts";
}

std::string ScriptWorkspace::scratchName(const std::string& uuid, const std::string& label) {

    std::string out = uuid.substr(0, 8);
    out.push_back('_');

    std::string name;
    for (const char c : label) {
        const auto u = static_cast<unsigned char>(c);
        name.push_back(std::isalnum(u) || c == '-' || c == '.' ? c : '_');
        // A tab title is not a place for somebody's forty-character node name.
        if (name.size() >= 40) break;
    }
    // Nothing usable in the name (unnamed object, or a name of pure punctuation
    // that collapsed to underscores) — the uuid already identifies it.
    while (!name.empty() && name.back() == '_') name.pop_back();
    out += name.empty() ? "object" : name;

    return out + ".py";
}

std::string ScriptWorkspace::readSource(const std::filesystem::path& file, bool* ok) {

    if (ok) *ok = false;

    // Binary: text mode on Windows would translate the line endings on the way
    // in, which is the right answer by accident and the wrong one to rely on.
    std::ifstream in(file, std::ios::binary);
    if (!in) return {};

    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (ok) *ok = true;

    auto text = buffer.str();
    std::erase(text, '\r');
    return text;
}

bool ScriptWorkspace::writeSource(const std::filesystem::path& file, const std::string& text) {

    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    return out.good();
}
