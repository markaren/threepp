
#include "EditorApp.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>

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
        } else if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0) {
            std::cout << "threepp editor\n"
                      << "  usage: threepp_editor [options] [scene.json]\n"
                      << "  --vulkan         use the Vulkan backend (OpenGL is the default)\n"
                      << "  --frames=N       render N frames and exit (smoke testing)\n"
                      << "  --play           press Play as soon as the scene is open\n"
                      << "  --selftest       drive the editor through its acceptance passes,\n"
                      << "                   print each one and exit non-zero on a failure\n"
                      << "  --urdf=PATH      import a URDF on start\n"
                      << "  --screenshot=PNG build the spline-tube scenario, play it, and write\n"
                      << "                   PNG plus one _<view>.png per camera, then exit\n";
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
