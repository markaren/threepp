// THREEPP_VK_STARTUP_TIMING=1: one line per section of the renderer's bring-up
// with the time it took, for finding where a slow start goes. The counterpart
// of THREEPP_VK_EXIT_TIMING. The first mark of the process starts the clock.

#ifndef THREEPP_VULKAN_STARTUP_TIMING_HPP
#define THREEPP_VULKAN_STARTUP_TIMING_HPP

#include "EnvKnob.hpp"

#include <chrono>
#include <iostream>

namespace threepp::vulkan {

    inline void startupMark(const char* what) {
        static const bool on = env::flag("THREEPP_VK_STARTUP_TIMING");
        if (!on) return;
        using clock = std::chrono::steady_clock;
        static const clock::time_point first = clock::now();
        static clock::time_point last = first;
        const auto now = clock::now();
        std::cout << "[vk-start] " << what << " "
                  << std::chrono::duration<double, std::milli>(now - last).count() << " ms (at "
                  << std::chrono::duration<double, std::milli>(now - first).count() << ")" << std::endl;
        last = now;
    }

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_STARTUP_TIMING_HPP
