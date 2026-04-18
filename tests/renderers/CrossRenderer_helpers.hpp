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

#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshNormalMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/LineDashedMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/geometries/RingGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/ShadowMaterial.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/textures/CubeTexture.hpp"

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

        // The GL canvas must exist first to establish a GL context,
        // which wgpu-native's GL backend needs.
        (void)glCanvas();

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

        (void)glCanvas();

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

    // Render with GL, return pixel data
    inline std::vector<unsigned char> renderWithGL(Object3D& scene, Camera& camera, const Color& clearColor) {
        GLRenderer renderer(glCanvas());
        renderer.setClearColor(clearColor);

        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    }

    // Render with Wgpu, return pixel data
    inline std::vector<unsigned char> renderWithWgpu(Object3D& scene, Camera& camera, const Color& clearColor) {
        // Wgpu canvas is created on demand (only when Wgpu is available)
        static Canvas* wgpuCanvasPtr = nullptr;
        if (!wgpuCanvasPtr) {
            wgpuCanvasPtr = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
        }

        WgpuRenderer renderer(*wgpuCanvasPtr);
        renderer.setClearColor(clearColor);

        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
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
