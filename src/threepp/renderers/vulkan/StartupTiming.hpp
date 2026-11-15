// THREEPP_LOG=vulkan.startup:debug: one line per section of the renderer's
// bring-up with the time it took, for finding where a slow start goes. The
// counterpart of vulkan.exit. The first mark of the process starts the clock.

#ifndef THREEPP_VULKAN_STARTUP_TIMING_HPP
#define THREEPP_VULKAN_STARTUP_TIMING_HPP

#include "EnvKnob.hpp"
#include "VulkanLog.hpp"

#include <chrono>

namespace threepp::vulkan {

    inline void startupMark(const char* what) {
        if (!kLogStartup.enabled(log::Level::Debug)) return;
        using clock = std::chrono::steady_clock;
        static const clock::time_point first = clock::now();
        static clock::time_point last = first;
        const auto now = clock::now();
        THREEPP_LOG(kLogStartup, Debug) << what << " "
                                        << std::chrono::duration<double, std::milli>(now - last).count() << " ms (at "
                                        << std::chrono::duration<double, std::milli>(now - first).count() << ")";
        last = now;
    }

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_STARTUP_TIMING_HPP
