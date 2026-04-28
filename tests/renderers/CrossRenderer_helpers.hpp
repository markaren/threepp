// Cross-renderer test helpers: shared infrastructure for GL/Wgpu/Cross tests.
//
// All utility functions, canvas singletons, and macros used by the split
// test files live here. Functions are `inline` to ensure a single instance
// across translation units (especially important for the canvas singletons).

#ifndef CROSSRENDERER_HELPERS_HPP
#define CROSSRENDERER_HELPERS_HPP

#include <catch2/catch_test_macros.hpp>

#include "threepp/threepp.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/materials/MeshToonMaterial.hpp"

#include "threepp/scenes/Fog.hpp"

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace threepp;

namespace crosstest {

    constexpr int RT_WIDTH = 64;
    constexpr int RT_HEIGHT = 64;
    constexpr int PIXEL_COUNT = RT_WIDTH * RT_HEIGHT;
    constexpr int DATA_SIZE = PIXEL_COUNT * 3;

    // Persistent GL canvas — avoids glfwTerminate between tests
    inline Canvas& glCanvas() {
        static Canvas c(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true));
        return c;
    }

    // Probe whether a WebGPU adapter can be obtained (no surface needed).
    // Also detects whether the adapter is a software/CPU rasterizer (e.g. lavapipe).
    inline bool isWgpuAvailable() {
        static int cached = -1;
        if (cached >= 0) return cached != 0;

        // Use only primary backends (Vulkan/Metal/DX12) to avoid EGL conflicts
        // with GLFW's GL context when the GL backend is also enabled.
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        WGPUInstance inst = wgpuCreateInstance(&instanceDesc);
        if (!inst) {
            cached = 0;
            return false;
        }

        struct UserData {
            bool done = false;
            WGPUAdapter adapter = nullptr;
        } ud;

        WGPURequestAdapterOptions opts{};
        opts.powerPreference = WGPUPowerPreference_LowPower;
        opts.compatibleSurface = nullptr;

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView, void* userdata1, void*) {
            auto* u = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                u->adapter = adapter;
            }
            u->done = true;
        };
        callbackInfo.userdata1 = &ud;

        wgpuInstanceRequestAdapter(inst, &opts, callbackInfo);

        for (int i = 0; i < 100 && !ud.done; i++) {
            wgpuInstanceProcessEvents(inst);
        }

        bool available = ud.adapter != nullptr;
        if (ud.adapter) wgpuAdapterRelease(ud.adapter);
        wgpuInstanceRelease(inst);
        cached = available ? 1 : 0;
        return available;
    }

    // Detect whether the Wgpu adapter is a CPU/software rasterizer (e.g. lavapipe).
    inline bool isSoftwareAdapter() {
        static int cached = -1;
        if (cached >= 0) return cached != 0;

        if (!isWgpuAvailable()) {
            cached = 0;
            return false;
        }

        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        WGPUInstance inst = wgpuCreateInstance(&instanceDesc);
        if (!inst) {
            cached = 0;
            return false;
        }

        struct UserData {
            bool done = false;
            WGPUAdapter adapter = nullptr;
        } ud;

        WGPURequestAdapterOptions opts{};
        opts.powerPreference = WGPUPowerPreference_LowPower;
        opts.compatibleSurface = nullptr;

        WGPURequestAdapterCallbackInfo cbInfo{};
        cbInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        cbInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                              WGPUStringView, void* userdata1, void*) {
            auto* u = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                u->adapter = adapter;
            }
            u->done = true;
        };
        cbInfo.userdata1 = &ud;

        wgpuInstanceRequestAdapter(inst, &opts, cbInfo);
        for (int i = 0; i < 100 && !ud.done; i++) {
            wgpuInstanceProcessEvents(inst);
        }

        bool software = false;
        if (ud.adapter) {
            WGPUAdapterInfo info{};
            wgpuAdapterGetInfo(ud.adapter, &info);
            software = (info.adapterType == WGPUAdapterType_CPU);
            // Fallback: detect software adapters by description string
            if (!software && info.description.length > 0) {
                std::string desc(info.description.data, info.description.length);
                if (desc.find("llvmpipe") != std::string::npos ||
                    desc.find("lavapipe") != std::string::npos ||
                    desc.find("SwiftShader") != std::string::npos) {
                    software = true;
                }
            }
            wgpuAdapterInfoFreeMembers(info);
            wgpuAdapterRelease(ud.adapter);
        }

        wgpuInstanceRelease(inst);
        cached = software ? 1 : 0;
        return software;
    }

    struct AvgColor {
        double r, g, b;
    };

    inline AvgColor averageColor(const std::vector<unsigned char>& pixels) {
        double r = 0, g = 0, b = 0;
        int count = static_cast<int>(pixels.size()) / 3;
        for (int i = 0; i < count; i++) {
            r += pixels[i * 3 + 0];
            g += pixels[i * 3 + 1];
            b += pixels[i * 3 + 2];
        }
        return {r / count, g / count, b / count};
    }

    inline bool allPixelsMatch(const std::vector<unsigned char>& pixels,
                        unsigned char r, unsigned char g, unsigned char b,
                        int tolerance) {
        int count = static_cast<int>(pixels.size()) / 3;
        for (int i = 0; i < count; i++) {
            if (std::abs(static_cast<int>(pixels[i * 3 + 0]) - r) > tolerance) return false;
            if (std::abs(static_cast<int>(pixels[i * 3 + 1]) - g) > tolerance) return false;
            if (std::abs(static_cast<int>(pixels[i * 3 + 2]) - b) > tolerance) return false;
        }
        return true;
    }

    inline int countNonBlack(const std::vector<unsigned char>& pixels, int threshold = 5) {
        int count = static_cast<int>(pixels.size()) / 3;
        int nonBlack = 0;
        for (int i = 0; i < count; i++) {
            if (pixels[i * 3] > threshold || pixels[i * 3 + 1] > threshold || pixels[i * 3 + 2] > threshold) {
                nonBlack++;
            }
        }
        return nonBlack;
    }

    // Render with GL, return pixel data. GLRenderer is constructed per call
    // because GLFW makes a single OpenGL context current at a time — a
    // persistent helper renderer would silently lose its context whenever a
    // test constructs its own GLRenderer on a different canvas. GL init is
    // also fast enough (<1 s for the whole gl test) that sharing wouldn't
    // pay off the way it does for Wgpu.
    inline std::vector<unsigned char> renderWithGL(Object3D& scene, Camera& camera, const Color& clearColor) {
        GLRenderer renderer(glCanvas());
        renderer.setClearColor(clearColor);
        renderer.render(scene, camera);
        return renderer.readRGBPixels();
    }

    // Persistent Wgpu canvas — same lifetime as glCanvas() to avoid GLFW re-init between tests
    inline Canvas& wgpuCanvas() {
        static Canvas c(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true));
        return c;
    }

    // Persistent Wgpu renderer + render target. Constructing a WgpuRenderer
    // requests an adapter and creates a device/queue/pipeline cache — ~250 ms
    // each on this hardware — so sharing a single instance across the ~106
    // renderWithWgpu calls is a major speedup. Tests that need to mutate
    // renderer-wide state (setSize, setPixelRatio, toneMapping, sampleCount,
    // outputColorSpace) construct their own renderer; they don't touch this
    // singleton, so cross-test state never leaks here.
    //
    // The renderer + canvas + target are intentionally leaked. Static
    // destruction ordering between WGPU dispose, GLFW termination, and the
    // canvas frame-end callback (which captures `this` from the renderer)
    // segfaults at process exit. The parity-clipping test already uses the
    // same leak pattern (see wgpuClipCross). The OS reclaims memory at exit.
    //
    // The persistent renderer gets its own dedicated canvas because each
    // WgpuRenderer takes ownership of its canvas's WGPU surface — it can't
    // share a canvas with direct-construction tests.
    inline WgpuRenderer& wgpuRenderer() {
        static WgpuRenderer* r = [] {
            auto* canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true));
            return new WgpuRenderer(*canvas);
        }();
        return *r;
    }

    // Render with Wgpu, return pixel data
    inline std::vector<unsigned char> renderWithWgpu(Object3D& scene, Camera& camera, const Color& clearColor) {
        auto& renderer = wgpuRenderer();
        // Same lifetime as the renderer — leaked, never disposed.
        static RenderTarget* target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{}).release();

        renderer.setClearColor(clearColor);
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        return pixels;
    }

    inline int maxPixelBrightness(const std::vector<unsigned char>& px) {
        int maxVal = 0;
        for (size_t i = 0; i < px.size(); i += 3) {
            int brightness = px[i] + px[i + 1] + px[i + 2];
            maxVal = std::max(maxVal, brightness);
        }
        return maxVal;
    }

    inline double avgBrightness(const std::vector<unsigned char>& px) {
        auto avg = averageColor(px);
        return (avg.r + avg.g + avg.b) / 3.0;
    }

    inline AvgColor centerPixel(const std::vector<unsigned char>& px, int w, int h) {
        int cx = w / 2, cy = h / 2;
        int i = (cy * w + cx) * 3;
        return {(double)px[i], (double)px[i + 1], (double)px[i + 2]};
    }

    inline double avgXPosition(const std::vector<unsigned char>& pixels, int width, int height) {
        double sumX = 0;
        int count = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
                    sumX += x;
                    count++;
                }
            }
        }
        return count > 0 ? sumX / count : 0.0;
    }

    // Create a procedural 2x2 texture with given RGBA pixel values
    inline std::shared_ptr<Texture> makeProceduralTexture(
            unsigned char r0, unsigned char g0, unsigned char b0,
            unsigned char r1, unsigned char g1, unsigned char b1,
            unsigned char r2, unsigned char g2, unsigned char b2,
            unsigned char r3, unsigned char g3, unsigned char b3) {
        std::vector<unsigned char> data = {
            r0, g0, b0, 255, r1, g1, b1, 255,
            r2, g2, b2, 255, r3, g3, b3, 255
        };
        return Texture::create(Image(std::move(data), 2, 2));
    }

    // Create a uniform-color 1x1 texture
    inline std::shared_ptr<Texture> makeUniformTexture(unsigned char r, unsigned char g, unsigned char b) {
        std::vector<unsigned char> data = {r, g, b, 255};
        return Texture::create(Image(std::move(data), 1, 1));
    }

    // Create a stepped gradient texture for toon shading (width x 1 pixels)
    inline std::shared_ptr<Texture> makeGradientTexture(const std::vector<unsigned char>& steps) {
        std::vector<unsigned char> data;
        data.reserve(steps.size() * 4);
        for (auto v : steps) {
            data.push_back(v);
            data.push_back(v);
            data.push_back(v);
            data.push_back(255);
        }
        auto tex = Texture::create(Image(std::move(data), static_cast<int>(steps.size()), 1));
        tex->magFilter = Filter::Nearest;
        tex->minFilter = Filter::Nearest;
        return tex;
    }

    // Count pixels with intermediate brightness (between thresholds) — for MSAA edge detection
    inline int countIntermediatePixels(const std::vector<unsigned char>& pixels, int lo = 15, int hi = 240) {
        int count = 0;
        for (size_t i = 0; i < pixels.size(); i += 3) {
            int brightness = pixels[i] + pixels[i + 1] + pixels[i + 2];
            if (brightness > lo * 3 && brightness < hi * 3) {
                count++;
            }
        }
        return count;
    }

    // Count dim-but-not-black pixels — for shadow coverage measurement
    inline int countDarkPixels(const std::vector<unsigned char>& pixels, int darkThreshold = 40, int blackThreshold = 5) {
        int count = 0;
        for (size_t i = 0; i < pixels.size(); i += 3) {
            int brightness = pixels[i] + pixels[i + 1] + pixels[i + 2];
            if (brightness > blackThreshold * 3 && brightness < darkThreshold * 3) {
                count++;
            }
        }
        return count;
    }

    // Brightness variance — measures how much pixel brightness varies across the image
    inline double brightnessVariance(const std::vector<unsigned char>& pixels) {
        int count = static_cast<int>(pixels.size()) / 3;
        double mean = avgBrightness(pixels);
        double variance = 0;
        for (int i = 0; i < count; i++) {
            double b = (pixels[i * 3] + pixels[i * 3 + 1] + pixels[i * 3 + 2]) / 3.0;
            variance += (b - mean) * (b - mean);
        }
        return variance / count;
    }

}// namespace crosstest

using namespace crosstest;

#define REQUIRE_WGPU() do { if (!isWgpuAvailable()) SKIP("No GPU backend available for Wgpu"); } while(0)
#define SKIP_ON_SOFTWARE_ADAPTER() do { if (isSoftwareAdapter()) SKIP("Test skipped on software adapter (e.g. lavapipe)"); } while(0)

#endif//CROSSRENDERER_HELPERS_HPP
