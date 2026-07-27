// The plumbing behind "Edit in VS Code": a scripts workspace, and the text
// discipline that makes a round trip through a real editor lossless.
//
// Deliberately free of ImGui, of Python and of anything platform-shaped, so the
// two halves that are pure functions — the settings.json generator and the
// normalization every save goes through — are testable without an editor at
// all (tests/extras/EditorScriptWorkspace_test.cpp). Launching the editor
// itself is the part that cannot be pure, and it lives in apps/editor.
//
// The workspace is one file, `<dir>/.vscode/settings.json`, whose only real
// job is to point Pylance at the threepp type stubs so `import threepp`
// completes. It is generated once and never again: an existing file is left
// exactly as the user left it.

#ifndef THREEPP_EDITOR_SCRIPTWORKSPACE_HPP
#define THREEPP_EDITOR_SCRIPTWORKSPACE_HPP

#include <filesystem>
#include <string>

namespace threepp::editor {

    struct ScriptWorkspace {

        // Tabs become four spaces, carriage returns are dropped.
        //
        // Both halves matter for an external round trip. Python raises TabError
        // on source that mixes tabs with spaces, and a document written by the
        // editor's own template is space-indented — so a file somebody indented
        // with tabs has to be normalized on the way in, exactly as the Script
        // Editor's Apply does. And every save from a Windows editor arrives
        // CRLF-terminated: without dropping the CRs, the first save after
        // opening the file would look like a change to every single line.
        [[nodiscard]] static std::string normalize(const std::string& text);

        // The `.vscode/settings.json` body pointing Pylance at `stubs` — the
        // directory that CONTAINS the `threepp` stub package, i.e.
        // <source>/python/threepp, since the stubs themselves are
        // python/threepp/threepp/__init__.pyi.
        //
        // JSON with comments, which is what VS Code's settings parser reads:
        // the comments are half the point, since this file is the only place
        // the user is told how to also resolve their own interpreter.
        [[nodiscard]] static std::string settingsJson(const std::filesystem::path& stubs);

        struct Result {
            bool ok = false;
            // True only when this call wrote the file. ok && !created means it
            // was already there and was left alone.
            bool created = false;
            std::filesystem::path file;
            std::string error;
        };

        // Idempotent and non-destructive: creates `<dir>/.vscode/settings.json`
        // when it is absent and touches nothing when it is not.
        static Result ensure(const std::filesystem::path& dir, const std::filesystem::path& stubs);

        // Where exported inline scripts are staged: <temp>/threepp-editor/scripts.
        // One directory for all of them, so its .vscode/settings.json is
        // generated once and serves every inline script ever opened.
        [[nodiscard]] static std::filesystem::path scratchDir();

        // "3f2a9c01_Robot Arm.py" — the uuid prefix keeps two objects with the
        // same name apart, and the name keeps the tab in VS Code readable.
        // Everything that is not a plain filename character is folded to '_'.
        [[nodiscard]] static std::string scratchName(const std::string& uuid, const std::string& label);

        // Binary read with CRs dropped (see normalize). `ok` distinguishes an
        // empty file from one that could not be opened.
        [[nodiscard]] static std::string readSource(const std::filesystem::path& file, bool* ok = nullptr);

        // Binary write, byte for byte — the source goes out exactly as the
        // document holds it, LF endings and all.
        static bool writeSource(const std::filesystem::path& file, const std::string& text);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SCRIPTWORKSPACE_HPP
