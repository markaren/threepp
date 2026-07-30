
#include "EditorApp.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace {

    // "px,py,pz@tx,ty,tz[:label]" — a camera placement for the --screenshot
    // pass over a loaded scene. Deliberately terse: this is iterated on from a
    // shell while looking at PNGs, not typed once.
    //
    // The placement itself is EditorApp::parseViewSpec, which is also what a
    // document's userData["editorView"] is read with: one format, one parser,
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
        if (!threepp::editor::EditorApp::parseViewSpec(placement, shot.position, shot.target)) {
            return std::nullopt;
        }
        shot.label = std::move(label);
        return shot;
    }

}// namespace

int main(int argc, char** argv) {

    threepp::editor::EditorApp::Options options;

    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];
        if (std::strcmp(argument, "--vulkan") == 0) {
            options.vulkan = true;
        } else if (std::strncmp(argument, "--frames=", 9) == 0) {
            options.maxFrames = std::atoi(argument + 9);
        } else if (std::strcmp(argument, "--selftest") == 0) {
            options.selfTest = true;
        } else if (std::strcmp(argument, "--play") == 0) {
            options.play = true;
        } else if (std::strncmp(argument, "--urdf=", 7) == 0) {
            options.urdf = argument + 7;
        } else if (std::strncmp(argument, "--screenshot=", 13) == 0) {
            options.screenshot = argument + 13;
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
        } else if (std::strncmp(argument, "--shot=", 7) == 0) {
            // px,py,pz@tx,ty,tz[:label]
            if (auto shot = parseShot(argument + 7)) {
                options.shots.push_back(*shot);
            } else {
                std::cerr << "threepp editor: cannot read --shot=" << (argument + 7)
                          << " (want px,py,pz@tx,ty,tz[:label])" << std::endl;
                return 2;
            }
        } else if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0) {
            std::cout << "threepp editor\n"
                      << "  usage: threepp_editor [options] [scene.json]\n"
                      << "  --vulkan         use the Vulkan backend (OpenGL is the default)\n"
                      << "  --frames=N       render N frames and exit (smoke testing)\n"
                      << "  --play           press Play as soon as the scene is open\n"
                      << "  --selftest       drive the editor through its acceptance passes,\n"
                      << "                   print each one and exit non-zero on a failure\n"
                      << "  --urdf=PATH      import a URDF on start\n"
                      << "  --example=SLUG   open a scene that ships in the binary (hover-arena)\n"
                      << "  --screenshot=PNG with no scene of its own: build the spline-tube\n"
                      << "                   scenario, play it and write PNG plus one _<view>.png\n"
                      << "                   per camera. With a scene.json or --example, photograph\n"
                      << "                   THAT instead, honouring --play/--seconds/--shot\n"
                      << "  --seconds=N      how long to play before the first shot (default 3)\n"
                      << "  --keys=W,A       hold these keys for that time, then let go, so a\n"
                      << "                   scene you are meant to DRIVE can be photographed moving\n"
                      << "  --shot=P@T[:tag] camera position@target for that pass, repeatable;\n"
                      << "                   e.g. --shot=0,18,34@0,2,-2:wide. None = auto-framed\n";
            return 0;
        } else if (argument[0] != '-') {
            options.openOnStart = argument;
        }
    }

    try {
        threepp::editor::EditorApp app(options);
        return app.run();
    } catch (const std::exception& e) {
        std::cerr << "threepp editor: " << e.what() << std::endl;
        return 1;
    }
}
