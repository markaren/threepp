// Shared by the plain exit-code Vulkan tests in this folder: the code they
// return when there is no GPU to test on, the guarded construction of the
// headless canvas and renderer that decides it, and the helpers most of them
// need (pass/fail lines, environment knobs, a constant environment map, sRGB
// decode, PPM files, the renderer state pixel comparisons are made in).

#ifndef THREEPP_VULKAN_TEST_UTIL_HPP
#define THREEPP_VULKAN_TEST_UTIL_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/Texture.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vktest {

    // CTest SKIP_RETURN_CODE (no Vulkan GPU). tests/renderers/CMakeLists.txt
    // names the same number per test.
    inline constexpr int kSkipCode = 42;

    // A headless canvas and the Vulkan renderer on it. The renderer is declared
    // last so that it is destroyed first.
    struct HeadlessVulkan {
        std::unique_ptr<threepp::Canvas> canvas;
        std::unique_ptr<threepp::VulkanRenderer> renderer;

        explicit operator bool() const { return renderer != nullptr; }
    };

    // Empty, with the reason printed, on a machine with no Vulkan GPU: the
    // caller then returns kSkipCode. `params` carries the test's own settings
    // (title, size, antialiasing); the canvas is always headless, without vsync.
    inline HeadlessVulkan openHeadlessVulkan(threepp::Canvas::Parameters params) {

        HeadlessVulkan out;
        try {
            out.canvas = std::make_unique<threepp::Canvas>(params.vsync(false).headless(true));
            out.renderer = std::make_unique<threepp::VulkanRenderer>(*out.canvas);
        } catch (const std::exception& e) {
            std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
            return {};
        }
        return out;
    }

    inline HeadlessVulkan openHeadlessVulkan(const char* title, int width, int height) {
        return openHeadlessVulkan(threepp::Canvas::Parameters().title(title).size(width, height));
    }

    // Native resolution, neither temporal upscaler, fixed exposure: the state
    // these tests compare pixels in. The four setters are independent, so the
    // order they used to be called in does not matter.
    inline void pinNativeOutput(threepp::VulkanRenderer& renderer) {
        renderer.setRenderScale(1.f);
        renderer.setDlss(false);
        renderer.setFsr(false);
        renderer.setAutoExposure(false);
    }

    // One line per condition, and the count a test's main returns non-zero on.
    inline int failures = 0;

    inline void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    inline void setEnv(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, /*overwrite*/ 1);
#endif
    }

    // A 64x32 equirect environment of constant linear radiance, alpha 1.
    inline std::shared_ptr<threepp::Texture> constantEnv(float r, float g, float b) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 1.f);
        for (size_t i = 0; i < data.size(); i += 4) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        threepp::Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = threepp::Texture::create(img);
        tex->format = threepp::Format::RGBA;
        tex->type = threepp::Type::Float;
        tex->colorSpace = threepp::ColorSpace::Linear;
        tex->mapping = threepp::Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    inline std::shared_ptr<threepp::Texture> constantEnv(float v) {
        return constantEnv(v, v, v);
    }

    // sRGB decode of a value in [0, 1], and of an 8-bit channel.
    inline double srgbToLinear(double c) {
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    inline double srgbByteToLinear(double byte) {
        return srgbToLinear(byte / 255.0);
    }

    // Raw-RGB PPM (P6), the layout readRGBPixels returns, byte-exact both ways.
    inline void writePpm(const std::filesystem::path& path, const std::vector<unsigned char>& rgb, int w, int h) {
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }

    // Empty when the file is missing, is not a P6 PPM, or is short.
    inline std::vector<unsigned char> readPpm(const std::filesystem::path& path, int& w, int& h) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return {};
        std::string magic;
        f >> magic;
        if (magic != "P6") return {};
        int maxv = 0;
        f >> w >> h >> maxv;
        f.get();// the single whitespace byte before the pixels
        std::vector<unsigned char> d(static_cast<size_t>(w) * h * 3);
        f.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(d.size()));
        if (!f) return {};
        return d;
    }

}// namespace vktest

#endif//THREEPP_VULKAN_TEST_UTIL_HPP
