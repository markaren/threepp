
#include "PlayerApp.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

namespace {

    void usage() {

        std::cout
                << "threepp player - plays a scene the editor authored\n"
                << "  usage: threepp_player scene.json [options]\n"
                << "\n"
                << "  --episodes=N   play the document N times back to back. Each episode is a\n"
                << "                 full play/stop cycle, and stop restores the snapshot, so\n"
                << "                 episodes are independent (default 1)\n"
                << "  --frames=N     stop each episode after N frames\n"
                << "  --seconds=N    stop each episode after N simulated seconds. With both,\n"
                << "                 whichever comes first wins. Neither, windowed, means play\n"
                << "                 until the window is closed\n"
                << "  --vulkan       use the Vulkan backend (OpenGL is the default). In a build\n"
                << "                 without Vulkan support this warns and uses OpenGL\n"
                << "  --headless     no visible window. The GL context is still created (the\n"
                << "                 window is just not shown) so depth and lidar sensors still\n"
                << "                 scan. Steps at a fixed 1/60 s and, with no --frames or\n"
                << "                 --seconds, runs 10 simulated seconds per episode\n"
                << "  --dt=SECONDS   force a fixed simulation step instead of the wall clock\n"
                << "  --record=DIR   write the sensor CSVs under DIR. With --episodes>1 each\n"
                << "                 episode gets DIR/episode_NNN, because the sensor files are\n"
                << "                 named for the sensor and truncated on open\n"
                << "  --size=WxH     window size (default 1280x720)\n"
                << "\n"
                << "  Exit code 0 when every episode played clean, 1 when the document would\n"
                << "  not load or play, when an episode failed to stop, or when any script\n"
                << "  raised. 2 is a usage error.\n";
    }

}// namespace

int main(int argc, char** argv) {

    threepp::player::PlayerOptions options;

    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];

        if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0) {
            usage();
            return 0;
        } else if (std::strncmp(argument, "--episodes=", 11) == 0) {
            options.episodes = std::atoi(argument + 11);
        } else if (std::strncmp(argument, "--frames=", 9) == 0) {
            options.frames = std::atoi(argument + 9);
        } else if (std::strncmp(argument, "--seconds=", 10) == 0) {
            options.seconds = static_cast<float>(std::atof(argument + 10));
        } else if (std::strcmp(argument, "--vulkan") == 0) {
            options.vulkan = true;
        } else if (std::strcmp(argument, "--headless") == 0) {
            options.headless = true;
        } else if (std::strncmp(argument, "--dt=", 5) == 0) {
            options.dt = static_cast<float>(std::atof(argument + 5));
        } else if (std::strncmp(argument, "--record=", 9) == 0) {
            options.record = argument + 9;
        } else if (std::strncmp(argument, "--size=", 7) == 0) {
            const std::string value = argument + 7;
            const auto x = value.find('x');
            if (x == std::string::npos) {
                std::cerr << "threepp player: cannot read " << argument << " (want --size=WxH)"
                          << std::endl;
                return 2;
            }
            options.width = std::atoi(value.substr(0, x).c_str());
            options.height = std::atoi(value.substr(x + 1).c_str());
        } else if (argument[0] == '-') {
            std::cerr << "threepp player: unknown option " << argument << std::endl;
            usage();
            return 2;
        } else {
            options.scene = argument;
        }
    }

    if (options.scene.empty()) {
        std::cerr << "threepp player: no scene given" << std::endl;
        usage();
        return 2;
    }
    if (options.episodes < 1) {
        std::cerr << "threepp player: --episodes must be at least 1" << std::endl;
        return 2;
    }

    try {
        threepp::player::PlayerApp app(std::move(options));
        return app.run();
    } catch (const std::exception& e) {
        std::cerr << "threepp player: " << e.what() << std::endl;
        return 1;
    }
}
