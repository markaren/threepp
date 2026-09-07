#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/ScriptWorkspace.hpp"

#include <filesystem>
#include <fstream>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A directory of our own per test, removed on the way out.
    struct TempDir {

        std::filesystem::path path;

        explicit TempDir(const std::string& name)
            : path(std::filesystem::temp_directory_path() / ("threepp-workspace-test-" + name)) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path, ec);
        }

        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };

    std::string contentsOf(const std::filesystem::path& file) {

        std::ifstream in(file, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

}// namespace


TEST_CASE("normalization is what makes an external round trip lossless", "[editor]") {

    // CRLF from any Windows editor. Without this, the first save after opening
    // the file in VS Code would look like a change to every line in it.
    CHECK(ScriptWorkspace::normalize("a\r\nb\r\n") == "a\nb\n");
    // A lone CR goes too — the comparison is against LF text either way.
    CHECK(ScriptWorkspace::normalize("a\rb") == "ab");

    // Tabs become four spaces, because Python raises TabError on source that
    // mixes them and the editor's own template is space-indented.
    CHECK(ScriptWorkspace::normalize("if x:\n\tpass\n") == "if x:\n    pass\n");
    CHECK(ScriptWorkspace::normalize("\t\tx") == "        x");

    // Idempotent: normalizing what came back out of a commit changes nothing,
    // which is exactly what stops a save loop from committing forever.
    const std::string source = "class A:\r\n\tdef update(self, dt):\r\n\t\tpass\r\n";
    const auto once = ScriptWorkspace::normalize(source);
    CHECK(ScriptWorkspace::normalize(once) == once);

    // Text that needs nothing is returned unchanged, byte for byte.
    CHECK(ScriptWorkspace::normalize(once) == "class A:\n    def update(self, dt):\n        pass\n");
}

TEST_CASE("the generated settings point Pylance at the stubs", "[editor]") {

    const auto json = ScriptWorkspace::settingsJson("C:/dev/threepp/python/threepp");

    // The one line that has to be right: an ABSOLUTE stubPath, forward-slashed
    // so the file reads the same whoever wrote it.
    CHECK(json.find("\"python.analysis.stubPath\": \"C:/dev/threepp/python/threepp\"") !=
          std::string::npos);

    // The stubs have no .py beside them (the module is compiled into the
    // editor), so the unresolved-from-source warning is turned off or every
    // `import threepp` carries a squiggle.
    CHECK(json.find("reportMissingModuleSource") != std::string::npos);

    // The interpreter hint is present and COMMENTED OUT: it is an example, and
    // an uncommented one pointing at a path that does not exist would break
    // completion rather than add to it.
    const auto hint = json.find("python.defaultInterpreterPath");
    REQUIRE(hint != std::string::npos);
    const auto lineStart = json.rfind('\n', hint);
    REQUIRE(lineStart != std::string::npos);
    CHECK(json.find("//", lineStart) < hint);

#ifdef _WIN32
    // A native path is emitted forward-slashed rather than with backslashes,
    // which JSON would read as escapes.
    const auto native = ScriptWorkspace::settingsJson(
            std::filesystem::path("C:\\dev\\threepp\\python\\threepp"));
    CHECK(native.find("C:/dev/threepp/python/threepp") != std::string::npos);
#endif
}

TEST_CASE("workspace generation is idempotent and never overwrites", "[editor]") {

    const TempDir dir("ensure");

    const auto first = ScriptWorkspace::ensure(dir.path, "C:/dev/threepp/python/threepp");
    REQUIRE(first.ok);
    CHECK(first.created);
    CHECK(first.file == dir.path / ".vscode" / "settings.json");
    REQUIRE(std::filesystem::exists(first.file));
    CHECK(contentsOf(first.file).find("stubPath") != std::string::npos);

    // Whatever the user made of it survives every later open.
    const std::string sentinel = "{ \"python.defaultInterpreterPath\": \"mine\" }\n";
    ScriptWorkspace::writeSource(first.file, sentinel);

    const auto second = ScriptWorkspace::ensure(dir.path, "C:/dev/threepp/python/threepp");
    CHECK(second.ok);
    CHECK_FALSE(second.created);
    CHECK(contentsOf(first.file) == sentinel);
}

TEST_CASE("a scratch file is named after the object it came from", "[editor]") {

    const auto name = ScriptWorkspace::scratchName("3f2a9c01-8b4e-4f10-9d33-77aa1c2e5b60", "Robot Arm");
    CHECK(name == "3f2a9c01_Robot_Arm.py");

    // An unnamed object still gets a usable file name.
    CHECK(ScriptWorkspace::scratchName("0123456789abcdef", "") == "01234567_object.py");
    CHECK(ScriptWorkspace::scratchName("0123456789abcdef", "///") == "01234567_object.py");

    // Nothing that could escape the directory or upset a shell survives.
    const auto hostile = ScriptWorkspace::scratchName("aabbccddeeff", "../../etc/pa sswd\"");
    CHECK(hostile.find('/') == std::string::npos);
    CHECK(hostile.find('\\') == std::string::npos);
    CHECK(hostile.find('"') == std::string::npos);
    CHECK(hostile.rfind("aabbccdd_", 0) == 0);
    CHECK(hostile.size() < 64);

    // Two objects with the same name are still two files.
    CHECK(ScriptWorkspace::scratchName("11111111", "Box") !=
          ScriptWorkspace::scratchName("22222222", "Box"));

    CHECK(ScriptWorkspace::scratchDir().filename() == "scripts");
    CHECK(ScriptWorkspace::scratchDir().parent_path().filename() == "threepp-editor");
}

TEST_CASE("source goes out verbatim and comes back LF-terminated", "[editor]") {

    const TempDir dir("io");
    const auto file = dir.path / "spinner.py";

    const std::string source = "class Spinner:\n    speed = 1.5\n";
    REQUIRE(ScriptWorkspace::writeSource(file, source));

    bool ok = false;
    CHECK(ScriptWorkspace::readSource(file, &ok) == source);
    CHECK(ok);

    // What VS Code writes back on Windows, read as what the document holds.
    REQUIRE(ScriptWorkspace::writeSource(file, "class Spinner:\r\n    speed = 1.5\r\n"));
    CHECK(ScriptWorkspace::readSource(file) == source);

    // A file that is not there is not an empty file.
    ok = true;
    CHECK(ScriptWorkspace::readSource(dir.path / "missing.py", &ok).empty());
    CHECK_FALSE(ok);

    // An empty file is not a missing one, either.
    REQUIRE(ScriptWorkspace::writeSource(file, ""));
    ok = false;
    CHECK(ScriptWorkspace::readSource(file, &ok).empty());
    CHECK(ok);
}
