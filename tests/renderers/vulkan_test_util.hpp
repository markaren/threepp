// Shared by the plain exit-code Vulkan tests in this folder: the code they
// return when there is no GPU to test on, and the guarded construction of the
// headless canvas and renderer that decides it.

#ifndef THREEPP_VULKAN_TEST_UTIL_HPP
#define THREEPP_VULKAN_TEST_UTIL_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cstdio>
#include <exception>
#include <memory>

namespace vktest {

    // CTest SKIP_RETURN_CODE (no Vulkan/RT GPU). tests/renderers/CMakeLists.txt
    // names the same number per test.
    inline constexpr int kSkipCode = 42;

    // A headless canvas and the Vulkan renderer on it. The renderer is declared
    // last so that it is destroyed first.
    struct HeadlessVulkan {
        std::unique_ptr<threepp::Canvas> canvas;
        std::unique_ptr<threepp::VulkanRenderer> renderer;

        explicit operator bool() const { return renderer != nullptr; }
    };

    // Empty, with the reason printed, on a machine with no Vulkan/RT GPU: the
    // caller then returns kSkipCode.
    inline HeadlessVulkan openHeadlessVulkan(const char* title, int width, int height) {

        HeadlessVulkan out;
        try {
            out.canvas = std::make_unique<threepp::Canvas>(
                    threepp::Canvas::Parameters().title(title).size(width, height).vsync(false).headless(true));
            out.renderer = std::make_unique<threepp::VulkanRenderer>(*out.canvas);
        } catch (const std::exception& e) {
            std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
            return {};
        }
        return out;
    }

}// namespace vktest

#endif//THREEPP_VULKAN_TEST_UTIL_HPP
