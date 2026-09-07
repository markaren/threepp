// "Edit in VS Code" — the editor's scripts, in a real code editor.
//
// Two tiers, one mechanism underneath.
//
//   FILE scripts are simply handed over. Every Play recompiles the .py from
//   source, so the file is already hot and nothing has to be watched.
//
//   INLINE scripts live in the scene and have no file at all, so one is made:
//   the source is exported to a scratch .py, VS Code is opened on it, and the
//   file is polled once a second from the frame loop. Each save comes back in
//   through the Script Editor's own Apply — the same normalization, the same
//   compile check, the same undoable commit. There is one write path for
//   script source into a document, and this is not a second one.
//
// Polling, not a watcher: ReadDirectoryChangesW/inotify mean a thread, and a
// thread means a callback arriving in the middle of a frame with the scene
// half-rebuilt. Reading a few kilobytes once a second costs nothing and lands
// where every other editor mutation lands — in a frame, on the main thread.
//
// Both tiers first generate a workspace — `.vscode/settings.json` pointing
// Pylance at the threepp stubs — in the directory being opened, so `import
// threepp` completes there. Generated once, never overwritten; see
// ScriptWorkspace.

#include "EditorApp.hpp"

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
//
#include <shellapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "shell32.lib")
#endif
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace threepp;
using namespace threepp::editor;

namespace {

    // What VS Code is asked to open: the FOLDER, then the file inside it.
    //
    // The folder is not decoration. VS Code applies `.vscode/settings.json`
    // only to a folder it has open as a workspace, so handing it the bare file
    // would open an editor with no stub path, no Pylance configuration and no
    // completion — the whole point of generating the settings in the first
    // place.
    //
    // Detached and non-blocking in every branch: the editor is mid-frame.

#ifdef _WIN32

    std::filesystem::path findVsCode() {

        wchar_t buffer[MAX_PATH];
        // `code` on PATH is code.cmd, a batch file — and running one means
        // cmd.exe, which means a console window flashing over the editor. The
        // .cmd lives in <install>/bin, so the real executable is one up.
        if (SearchPathW(nullptr, L"code", L".cmd", MAX_PATH, buffer, nullptr) == 0) return {};

        const std::filesystem::path cmd(buffer);
        std::error_code ec;
        const auto exe = cmd.parent_path().parent_path() / "Code.exe";
        if (std::filesystem::exists(exe, ec)) return exe;
        return cmd;
    }

    bool startDetached(const std::filesystem::path& program, const std::wstring& arguments) {

        if (program.extension() == ".cmd" || program.extension() == ".bat") {
            // The fallback for an install we could not find Code.exe in.
            // CREATE_NO_WINDOW is what keeps the console from flashing.
            std::wstring line = L"cmd.exe /c \"\"" + program.wstring() + L"\" " + arguments + L"\"";
            std::vector<wchar_t> mutableLine(line.begin(), line.end());
            mutableLine.push_back(L'\0');

            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            PROCESS_INFORMATION process{};
            if (!CreateProcessW(nullptr, mutableLine.data(), nullptr, nullptr, FALSE,
                                CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
                return false;
            }
            // Never waited on: this is a hand-off, not a child process we own.
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
            return true;
        }

        const auto result = ShellExecuteW(nullptr, L"open", program.wstring().c_str(),
                                          arguments.c_str(), nullptr, SW_SHOWNORMAL);
        return reinterpret_cast<INT_PTR>(result) > 32;
    }

    bool launchEditorProcess(const std::filesystem::path& dir, const std::filesystem::path& file,
                             std::string& how) {

        const std::wstring arguments = L"\"" + dir.wstring() + L"\" --goto \"" + file.wstring() + L":1\"";

        if (const auto code = findVsCode(); !code.empty() && startDetached(code, arguments)) {
            how = "VS Code";
            return true;
        }

        // No `code` on PATH: whatever the machine opens .py with. Says so at
        // the call site, because that editor knows nothing about the stubs.
        const auto result = ShellExecuteW(nullptr, L"open", file.wstring().c_str(),
                                          nullptr, nullptr, SW_SHOWNORMAL);
        if (reinterpret_cast<INT_PTR>(result) > 32) {
            how = "the default .py handler (no `code` on PATH, so no threepp completion)";
            return true;
        }
        return false;
    }

#else

    bool spawnDetached(const std::vector<std::string>& argv) {

        // Double fork: the intermediate exits immediately and is reaped here,
        // so the editor never accumulates zombies for something it does not own.
        const pid_t intermediate = fork();
        if (intermediate < 0) return false;

        if (intermediate == 0) {
            if (fork() == 0) {
                setsid();
                std::vector<char*> args;
                args.reserve(argv.size() + 1);
                for (const auto& argument : argv) args.push_back(const_cast<char*>(argument.c_str()));
                args.push_back(nullptr);
                execvp(args[0], args.data());
                _exit(127);
            }
            _exit(0);
        }

        int status = 0;
        waitpid(intermediate, &status, 0);
        return true;
    }

    bool launchEditorProcess(const std::filesystem::path& dir, const std::filesystem::path& file,
                             std::string& how) {

        if (spawnDetached({"code", dir.string(), "--goto", file.string() + ":1"})) {
            how = "VS Code";
            return true;
        }
#ifdef __APPLE__
        const char* opener = "open";
#else
        const char* opener = "xdg-open";
#endif
        if (spawnDetached({opener, file.string()})) {
            how = "the default .py handler (no `code` on PATH, so no threepp completion)";
            return true;
        }
        return false;
    }

#endif

}// namespace


std::filesystem::path EditorApp::pythonStubDir() {

    // The override first: an installed editor is nowhere near the source tree
    // it was built from, and this is the one knob that puts it right.
    if (const char* env = std::getenv("THREEPP_PYTHON_STUBS"); env && *env) return env;

#ifdef THREEPP_EDITOR_PYTHON_STUBS
    return THREEPP_EDITOR_PYTHON_STUBS;
#else
    return {};
#endif
}

void EditorApp::ensureScriptWorkspace(const std::filesystem::path& dir) {

    const auto stubs = pythonStubDir();

    std::error_code ec;
    if (stubs.empty() || !std::filesystem::exists(stubs, ec)) {
        // Worth saying out loud: the settings are still written (the path may
        // be right on the machine that opens them), but nobody should wonder
        // silently why completion is missing.
        log("script workspace: no type stubs at " +
            (stubs.empty() ? std::string("<unset>") : stubs.generic_string()) +
            " - set THREEPP_PYTHON_STUBS for `import threepp` completion");
    }

    const auto result = ScriptWorkspace::ensure(dir, stubs);
    if (!result.ok) {
        log("could not write " + result.file.generic_string() + " - " + result.error);
    } else if (result.created) {
        log("wrote " + result.file.generic_string() + " - `import threepp` completes in this folder");
    }
}

void EditorApp::launchExternalEditor(const std::filesystem::path& dir,
                                     const std::filesystem::path& file) {

    // The self-test drives everything about a session except this: opening a
    // window on the machine running the tests is not its business.
    if (options_.selfTest) {
        log("selftest: not launching an editor on " + file.filename().string());
        return;
    }

    std::string how;
    if (launchEditorProcess(dir, file, how)) {
        log("opened " + file.filename().string() + " in " + how);
    } else {
        log("could not open " + file.generic_string() + " - no `code` on PATH and no handler for .py");
    }
}

void EditorApp::openScriptFileExternally(const std::filesystem::path& file) {

    std::error_code ec;
    if (file.empty() || !std::filesystem::exists(file, ec)) {
        log("script file not found: " + file.generic_string());
        return;
    }

    const auto dir = file.parent_path();
    ensureScriptWorkspace(dir);
    launchExternalEditor(dir, file);
}

bool EditorApp::externalEditActive(const Object3D& object) const {

    return externalEdit_.active && externalEdit_.uuid == object.uuid;
}

void EditorApp::startExternalEdit(Object3D& object, ExternalEditKind kind) {

    std::string source;
    if (kind == ExternalEditKind::Generator) {
        const auto config = GeneratorConfig::read(object);
        if (!config) return;
        source = config->source;
    } else {
        const auto config = ScriptConfig::read(object).value_or(ScriptConfig{});
        if (!config.isInline()) return;
        source = config.source;
    }

    // One session at a time. Not for the reason the editor once held one tab —
    // it holds as many as you open now — but because a session owns a scratch
    // file, a command-stack transaction and a poll slot, and two of them would
    // be two writers racing to answer "what is this object's source".
    if (externalEdit_.active) {
        if (externalEdit_.uuid == object.uuid) return;
        stopExternalEdit("another object took over");
    }

    const auto dir = ScriptWorkspace::scratchDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        log("could not create " + dir.generic_string() + " - " + ec.message());
        return;
    }

    const auto label = object.name.empty() ? object.type() : object.name;
    const auto file = dir / ScriptWorkspace::scratchName(object.uuid, label);
    if (!ScriptWorkspace::writeSource(file, source)) {
        log("could not write " + file.generic_string());
        return;
    }

    ensureScriptWorkspace(dir);

    externalEdit_ = ExternalEditState{};
    externalEdit_.active = true;
    externalEdit_.kind = kind;
    externalEdit_.uuid = object.uuid;
    externalEdit_.label = label;
    externalEdit_.file = file;
    externalEdit_.synced = ScriptWorkspace::normalize(source);

    // Opened here and closed in stopExternalEdit: the first save lands as its
    // own undo entry (nothing merges into what preceded the transaction) and
    // every save after it merges into that one. The one side effect of holding
    // it open is that other edits sharing a merge key — two of this script's
    // parameters in a row, say — coalesce while the session lives, as if they
    // had been one drag.
    commands_.beginTransaction();
    externalEdit_.transaction = true;

    // A tab comes along read-only, because a session needs somewhere to
    // announce itself and somewhere to be stopped.
    openScriptEditor(object);
    if (auto* editor = scriptEditorFor(object.uuid)) editor->focus = false;

    log("editing " + label + " externally - " + file.generic_string());
    launchExternalEditor(dir, file);
}

void EditorApp::stopExternalEdit(const std::string& why) {

    if (!externalEdit_.active) return;

    if (externalEdit_.transaction) commands_.endTransaction();

    // Best effort: the file is a copy of what the document already holds, so
    // failing to delete it costs nothing but a stale file in the temp folder.
    std::error_code ec;
    std::filesystem::remove(externalEdit_.file, ec);

    const auto label = externalEdit_.label;
    externalEdit_ = ExternalEditState{};

    log("external editing of " + label + " stopped" + (why.empty() ? "" : " - " + why));
}

void EditorApp::pollExternalEdit(float dt) {

    if (!externalEdit_.active) return;

    externalEdit_.poll += dt;
    if (externalEdit_.poll < ExternalEditState::pollSeconds) return;
    externalEdit_.poll = 0.f;

    // By uuid, never by pointer: play/stop replaces the whole graph.
    auto* target = findByUuid(document_.scene(), externalEdit_.uuid);
    if (!target) {
        stopExternalEdit("the object is gone");
        return;
    }

    if (externalEdit_.kind == ExternalEditKind::Generator) {
        if (!GeneratorConfig::isGenerator(*target)) {
            stopExternalEdit("the generator was cleared");
            return;
        }
    } else {
        const auto config = ScriptConfig::read(*target).value_or(ScriptConfig{});
        if (!config.isInline()) {
            stopExternalEdit("the script was cleared");
            return;
        }
    }

    // The file's own text decides, not its write time. Two saves a millisecond
    // apart can carry the SAME last_write_time — the resolution the filesystem
    // actually updates at is nothing like the resolution it reports — and a
    // watcher that misses one of those has silently lost the user's work. A few
    // kilobytes read once a second costs nothing next to that.
    bool ok = false;
    const auto text = ScriptWorkspace::readSource(externalEdit_.file, &ok);
    if (!ok) {
        stopExternalEdit("the file was removed");
        return;
    }

    // Compared NORMALIZED, because that is the form it was committed in: a save
    // that only rewrote the line endings is not an edit, and on Windows every
    // save rewrites them.
    const auto candidate = ScriptWorkspace::normalize(text);
    if (candidate == externalEdit_.synced) {
        externalEdit_.waiting = false;
        return;
    }

    // Playing: the live scene is a copy that Stop throws away, so a commit now
    // would go with it. Nothing is consumed here — the poll after Stop sees the
    // same difference and applies it to the restored object. (Async imports are
    // parked on Stop the same way, for the same reason.)
    if (isPlaying()) {
        if (!externalEdit_.waiting) log("script saved while playing - it applies on Stop");
        externalEdit_.waiting = true;
        return;
    }

    externalEdit_.waiting = false;

    if (externalEdit_.kind == ExternalEditKind::Generator) {
        // No Script Editor tab in this path — a generator's source is not a
        // behaviour script and does not belong in the tab that owns those.
        externalEdit_.synced = applyGeneratorSource(*target, text);
        ++externalEdit_.syncs;

        std::string note;
#ifdef THREEPP_EDITOR_WITH_PYTHON
        const auto syntax = scripting::checkSyntax(text, externalEdit_.label);
        if (!syntax.empty()) note = " - with a syntax error, not run";
#endif
        // Saving IS the ask: the user opened this session on this generator, and
        // an edit-save-look loop that needs a button press in another window is
        // not a loop. Deferred by a frame like the button, for the same reason —
        // it replaces nodes, and panels are about to read them. A script that
        // does not parse is synced but not run.
        if (note.empty()) {
            pendingRegenerate_ = externalEdit_.uuid;
        }

        log("synced " + externalEdit_.label + " from " +
            externalEdit_.file.filename().string() + note);
        return;
    }

    // Through this object's own tab, not around it and not through whichever
    // one happens to be visible: the sync is the same normalization, the same
    // compile check and the same undo entry a typed edit gets. Reopened without
    // revealing if the user closed it — the session outlives its tab, and a
    // save must not yank the panel out from under them to say so.
    if (!scriptEditorFor(externalEdit_.uuid)) openScriptEditor(*target, false);

    auto* editor = scriptEditorFor(externalEdit_.uuid);
    if (!editor) return;

    editor->buffer = text;
    applyScriptEditor(*editor);

    externalEdit_.synced = editor->committed;
    ++externalEdit_.syncs;

    log("synced " + externalEdit_.label + " from " + externalEdit_.file.filename().string() +
        (editor->status.empty() ? "" : " - with a syntax error"));
}
