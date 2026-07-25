// GL renderer test helpers: shared pixel-measurement and fixture infrastructure.
//
// All utility functions and canvas singletons used by the GL test files live
// here. Functions are `inline` to ensure a single instance across translation
// units (especially important for the canvas singleton).

#ifndef THREEPP_GL_TEST_HELPERS_HPP
#define THREEPP_GL_TEST_HELPERS_HPP

#include <catch2/catch_test_macros.hpp>

#include "threepp/threepp.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/materials/MeshToonMaterial.hpp"

#include "threepp/scenes/Fog.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace threepp;

namespace gltest {

    constexpr int RT_WIDTH = 64;
    constexpr int RT_HEIGHT = 64;
    constexpr int PIXEL_COUNT = RT_WIDTH * RT_HEIGHT;
    constexpr int DATA_SIZE = PIXEL_COUNT * 3;

    // Persistent GL canvas — avoids glfwTerminate between tests
    inline Canvas& glCanvas() {
        static Canvas c(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true));
        return c;
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
    // fast enough (<1 s for the whole gl test) that sharing wouldn't pay off.
    inline std::vector<unsigned char> renderWithGL(Object3D& scene, Camera& camera, const Color& clearColor) {
        GLRenderer renderer(glCanvas());
        renderer.setClearColor(clearColor);
        renderer.render(scene, camera);
        return renderer.readRGBPixels();
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
        return {static_cast<double>(px[i]), static_cast<double>(px[i + 1]), static_cast<double>(px[i + 2])};
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

}// namespace gltest

using namespace gltest;

#endif//THREEPP_GL_TEST_HELPERS_HPP
