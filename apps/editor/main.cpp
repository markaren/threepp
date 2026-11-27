
#include "EditorApp.hpp"

#include "threepp/extras/editor/ViewSpec.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace {

    // "px,py,pz@tx,ty,tz[:label]" — a camera placement for the --screenshot
    // pass over a loaded scene. Deliberately terse: this is iterated on from a
    // shell while looking at PNGs, not typed once.
    //
    // The placement itself is editor::parseViewSpec (extras/editor/ViewSpec.hpp),
    // which is also what a document's userData["editorView"] is read with, in the
    // editor and in threepp_player: one format, one parser,
    // so a vantage can be copied from a shell into a scene and back.
    std::optional<threepp::editor::EditorApp::Options::Shot> parseShot(const std::string& text) {

        auto placement = text;
        std::string label;
        // The label is the tail after the LAST colon, which cannot be part of a
        // number and so cannot be part of the placement.
        if (const auto colon = placement.rfind(':'); colon != std::string::npos) {
            label = placement.substr(colon + 1);
            placement = placement.substr(0, colon);
        }

        threepp::editor::EditorApp::Options::Shot shot;
        if (!threepp::editor::parseViewSpec(placement, shot.position, shot.target)) {
            return std::nullopt;
        }
        shot.label = std::move(label);
        return shot;
    }

}// namespace

int main(int argc, char** argv) {

    threepp::editor::EditorApp::Options options;

#ifdef __EMSCRIPTEN__
    // A page starts in "/", among the in-memory file system's /dev and /proc.
    // The dialogs open in the working directory, so make that the user's home.
    {
        std::error_code ec;
        std::filesystem::create_directories("/home/web_user", ec);
        std::filesystem::current_path("/home/web_user", ec);
    }
#endif

    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];
        if (std::strcmp(argument, "--vulkan") == 0) {
            options.vulkan = true;
        } else if (std::strncmp(argument, "--frames=", 9) == 0) {
            options.maxFrames = std::atoi(argument + 9);
        } else if (std::strcmp(argument, "--selftest") == 0) {
            options.selfTest = true;
        } else if (std::strncmp(argument, "--selftest=", 11) == 0) {
            options.selfTest = true;
            options.selfTestFilter = argument + 11;
        } else if (std::strcmp(argument, "--play") == 0) {
            options.play = true;
        } else if (std::strncmp(argument, "--urdf=", 7) == 0) {
            options.urdf = argument + 7;
        } else if (std::strncmp(argument, "--character=", 12) == 0) {
            options.character = argument + 12;
        } else if (std::strncmp(argument, "--environment=", 14) == 0) {
            options.environment = argument + 14;
        } else if (std::strncmp(argument, "--screenshot=", 13) == 0) {
            options.screenshot = argument + 13;
        } else if (std::strncmp(argument, "--bench=", 8) == 0) {
            options.bench = std::atoi(argument + 8);
        } else if (std::strncmp(argument, "--example=", 10) == 0) {
            options.example = argument + 10;
        } else if (std::strncmp(argument, "--seconds=", 10) == 0) {
            options.settle = static_cast<float>(std::atof(argument + 10));
        } else if (std::strncmp(argument, "--keys=", 7) == 0) {
            std::istringstream held(argument + 7);
            std::string key;
            while (std::getline(held, key, ',')) {
                if (!key.empty()) options.keys.push_back(key);
            }
        } else if (std::strcmp(argument, "--hold-keys") == 0) {
            options.holdKeys = true;
        } else if (std::strncmp(argument, "--shot=", 7) == 0) {
            // px,py,pz@tx,ty,tz[:label]
            if (auto shot = parseShot(argument + 7)) {
                options.shots.push_back(*shot);
            } else {
                THREEPP_LOG(threepp::editor::kLogEditor, Error)
                        << "cannot read --shot=" << (argument + 7) << " (want px,py,pz@tx,ty,tz[:label])";
                return 2;
            }
        } else if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0) {
            std::cout << "threepp editor\n"
                      << "  usage: threepp_editor [options] [scene.json]\n"
                      << "  --vulkan         use the Vulkan backend (OpenGL is the default;\n"
                      << "                   View > Backend switches while running)\n"
                      << "  --frames=N       render N frames and exit (smoke testing)\n"
                      << "  --play           press Play as soon as the scene is open\n"
                      << "  --selftest       drive the editor through its acceptance passes,\n"
                      << "                   print each one and exit non-zero on a failure\n"
                      << "  --selftest=F     run only the sections matching F: comma-separated\n"
                      << "                   terms, an exact section name or a substring\n"
                      << "                   (--selftest=terrain,splat); a term matching no\n"
                      << "                   section lists the sections and exits non-zero\n"
                      << "  --urdf=PATH      with --selftest: also exercise the URDF import and\n"
                      << "                   round trip against this file\n"
                      << "  --character=PATH run the selftest's character pass against this\n"
                      << "                   rigged model (e.g. threepp_data's xbot.glb)\n"
                      << "  --environment=P  light the scene from a .hdr/.exr, as\n"
                      << "                   File > Set Environment does (background too)\n"
                      << "  --example=SLUG   open a scene that ships in the binary (hover-arena,\n"
                      << "                   timber-yard)\n"
                      << "  --screenshot=PNG with no scene of its own: build the spline-tube\n"
                      << "                   scenario, play it and write PNG plus one _<view>.png\n"
                      << "                   per camera. With a scene.json or --example, photograph\n"
                      << "                   THAT instead, honouring --play/--seconds/--shot\n"
                      << "  --bench=N        time N frames (after --seconds of warmup) with\n"
                      << "                   vsync OFF and print median/p95 frame time plus,\n"
                      << "                   on Vulkan, the per-pass GPU breakdown\n"
                      << "  --seconds=N      how long to play before the first shot (default 3)\n"
                      << "  --keys=W,A       hold these keys for that time, then let go, so a\n"
                      << "                   scene you are meant to DRIVE can be photographed moving\n"
                      << "  --hold-keys      keep holding them through the shots instead of letting\n"
                      << "                   go first, for a picture of a manoeuvre mid-way through\n"
                      << "  --shot=P@T[:tag] camera position@target for that pass, repeatable;\n"
                      << "                   e.g. --shot=0,18,34@0,2,-2:wide. None = auto-framed\n";
            return 0;
        } else if (argument[0] != '-') {
            options.openOnStart = argument;
        }
    }

    // View > Backend closes the editor and asks for it back on the other
    // renderer. A new EditorApp in this process rather than a relaunch, so the
    // console or debugger the editor was started from stays attached. The old
    // one is destroyed first: its window and renderer go before the next are
    // built.
    bool restarted = false;
    for (;;) {
        std::unique_ptr<threepp::editor::EditorApp> app;
        try {
            app = std::make_unique<threepp::editor::EditorApp>(options);
        } catch (const std::exception& e) {
            THREEPP_LOG(threepp::editor::kLogEditor, Error) << e.what();
            if (!restarted) return 1;
            // The backend asked for will not come up here (no Vulkan driver,
            // no capable GPU). The one the editor was just running on did, so
            // go back to it rather than leave nothing open.
            const char* asked = options.vulkan ? "Vulkan" : "OpenGL";
            options.vulkan = !options.vulkan;
            restarted = false;
            if (options.resume) {
                options.resume->notice = std::string(asked) + " did not start: " + e.what() +
                                         " - back on " + (options.vulkan ? "Vulkan" : "OpenGL");
            }
            continue;
        }

        int code = 0;
        try {
            code = app->run();
        } catch (const std::exception& e) {
            THREEPP_LOG(threepp::editor::kLogEditor, Error) << e.what();
            return 1;
        }

        auto next = app->restartRequest();
        app.reset();
        if (!next) return code;
        options = std::move(*next);
        restarted = true;
    }
}
