// VulkanEmitterSpecular_test: the specular light of an emissive mesh on a glossy
// floor. Two claims, one per defect found on 2026-10-12 with an emissive torus
// knot turning over a floor of roughness 0.22.
//
// IMAGE. The coherent emitter NEE stands in for a light with a few lighting
// points (8 strata for a lone light). Through a lobe that is narrow against the
// gaps between them the floor showed one highlight per point where the light's
// mirrored image belongs. Such a light's specular now stays with the reflection
// ray, per light (shaders/emissive_lights.glsl, kEmRaySpacing0), which is also
// what draws a glow-only emissive (setEmissiveCastsLight(false)). So the floor
// under the knot must look the same whether the knot casts light or only glows.
// Measured on an RTX 4060 Laptop GPU (2026-10-12), floor region, 8-bit levels:
// without the hand-off mean 12.13 and worst 8x8 block 98.07, with it 1.91 and 9.69.
// An RTX 4070 (same day) gave 1.905 and 9.69 with it.
//
// STRATA. emPlanBuild gives every light one stratum and shares the spare ones
// out by power: n[l] = 1 + floor(extra * power[l] / areaPower). For a lone light
// the share is extra * P / P: an integer, with the floor taken on it. The host
// sums a moving emitter's power from its world-space triangle areas on every
// frame, so P differs from frame to frame in its last bits, and on some bit
// patterns the float quotient is one ulp under the integer. Those frames had one
// stratum fewer (7 for 8), which moves the CDF position of every stratum and
// with it every lighting point (kEmShareSnap).
// Measured there: largest change per frame 15 to 21 times the median without
// kEmShareSnap (jumps into the frames where float32 7 * P / P < 7), 1.5 times with it.
//
// The frame clock is pinned (setSimTime), so the temporal blend weights and
// with them the size of a jump do not depend on how fast the machine renders.
//
// Plain exit-code program, not Catch2, the same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 960, kH = 540;
    // The floor below the knot: where its reflection and highlights fall.
    constexpr int kY0 = 300, kY1 = kH, kX0 = 240, kX1 = 720;

    std::shared_ptr<Texture> constantEnv(float le) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, le);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // Mean absolute difference of the floor region, in 8-bit levels.
    double floorDiff(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
        double sum = 0.0;
        size_t n = 0;
        for (int y = kY0; y < kY1; ++y) {
            const size_t row = static_cast<size_t>(y) * kW;
            for (int x = kX0; x < kX1; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const size_t i = (row + x) * 3 + c;
                    sum += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
                    ++n;
                }
            }
        }
        return sum / static_cast<double>(n);
    }

    // The largest mean absolute difference of an 8 x 8 block of the floor region.
    double floorBlockDiff(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
        double worst = 0.0;
        for (int by = kY0; by + 8 <= kY1; by += 8) {
            for (int bx = kX0; bx + 8 <= kX1; bx += 8) {
                double sum = 0.0;
                for (int y = by; y < by + 8; ++y) {
                    for (int x = bx; x < bx + 8; ++x) {
                        for (int c = 0; c < 3; ++c) {
                            const size_t i = (static_cast<size_t>(y) * kW + x) * 3 + c;
                            sum += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
                        }
                    }
                }
                worst = std::max(worst, sum / (8.0 * 8.0 * 3.0));
            }
        }
        return worst;
    }

    // Mean level of the floor region.
    double floorMean(const std::vector<unsigned char>& a) {
        double sum = 0.0;
        size_t n = 0;
        for (int y = kY0; y < kY1; ++y) {
            for (int x = kX0; x < kX1; ++x) {
                for (int c = 0; c < 3; ++c) {
                    sum += a[(static_cast<size_t>(y) * kW + x) * 3 + c];
                    ++n;
                }
            }
        }
        return sum / static_cast<double>(n);
    }

    // The floor and the knot over it. Its own material objects per scene: the
    // two cases differ in the floor's roughness and the knot's intensity.
    struct Stage {
        Scene scene;
        std::shared_ptr<Mesh> knot;

        Stage(const std::shared_ptr<Texture>& env, const std::shared_ptr<BufferGeometry>& knotGeometry,
              float floorRoughness, float emissiveIntensity) {
            scene.background = Color(0x05060a);
            scene.environment = env;

            const auto floorMat = MeshStandardMaterial::create();
            floorMat->color = Color(0x0c0d12);
            floorMat->metalness = 0.2f;
            floorMat->roughness = floorRoughness;
            auto floor = Mesh::create(PlaneGeometry::create(200.f, 200.f), floorMat);
            floor->rotateX(-math::PI / 2.f);
            scene.add(floor);

            const auto knotMat = MeshStandardMaterial::create();
            knotMat->color = Color(0x050608);
            knotMat->metalness = 0.f;
            knotMat->roughness = 0.4f;
            knotMat->emissive = Color(0x21e0c8);
            knotMat->emissiveIntensity = emissiveIntensity;
            knot = Mesh::create(knotGeometry, knotMat);
            knot->position.set(0, 1.1f, 0);
            // Its own geometry at every distance: a simplified level would be a
            // second thing that changes under the readbacks.
            knot->autoLod = false;
            scene.add(knot);
        }
    };

}// namespace

int main(int argc, char** argv) {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string dumpDir;// --dump <dir>: the pictures the checks compare, as PNGs
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dumpDir = argv[++i];
    }

    auto gpu = vktest::openHeadlessVulkan("VulkanEmitterSpecular_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMappingExposure = 1.0f;

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 500.f);
    camera.position.set(0, 2.4f, 6.f);
    camera.lookAt(Vector3{0, 0.6f, 0});
    camera.updateMatrixWorld();

    const auto env = constantEnv(0.02f);
    const auto knotGeometry = TorusKnotGeometry::create(0.52f, 0.2f, 240, 36);

    int failures = 0;
    const auto check = [&](const char* what, double got, double limit, bool atLeast = false) {
        const bool ok = atLeast ? got >= limit : got <= limit;
        std::printf("  %-52s %7.3f (%s %.3f) %s\n", what, got, atLeast ? "at least" : "limit", limit, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    int frameNo = 0;
    const auto frame = [&](Scene& scene) {
        renderer.setSimTime(static_cast<double>(++frameNo) / 60.0);
        canvas.animateOnce([&] { renderer.render(scene, camera); });
    };
    const auto dump = [&](const char* name) {
        if (!dumpDir.empty()) renderer.writeFramebuffer(dumpDir + "/" + name + ".png");
    };

    // Both stages for the whole run: the renderer keys what it caches on object
    // addresses, and a stage freed before the next is built hands the second
    // one's meshes and materials the first one's addresses.
    Stage image(env, knotGeometry, 0.22f, 3.f);
    Stage strata(env, knotGeometry, 0.55f, 2.4f);

    try {
        // ── IMAGE ───────────────────────────────────────────────────────────
        {
            constexpr int kSettleFrames = 60;
            const auto settle = [&] {
                for (int i = 0; i < kSettleFrames; ++i) frame(image.scene);
                return renderer.readRGBPixels();
            };
            const auto lit = settle();
            if (lit.size() < static_cast<size_t>(kW) * kH * 3) {
                std::printf("readback too small: %zu\n", lit.size());
                return 1;
            }
            dump("image_lit");
            const auto again = settle();
            renderer.setEmissiveCastsLight(*image.knot, false);
            const auto glow = settle();
            dump("image_glow");

            std::printf("image: a still knot over a roughness-0.22 floor, casting light against glow-only\n");
            // The reflection is there at all.
            check("mean level of the floor under the glow-only knot", floorMean(glow), 8.0, true);
            std::printf("  null (the lit scene read back twice): mean %.3f, worst 8x8 block %.2f\n",
                        floorDiff(again, lit), floorBlockDiff(again, lit));
            // Limits at about twice the measured value and a sixth of the defect.
            check("floor, casting light against glow-only: mean", floorDiff(lit, glow), 4.0);
            check("the same: worst 8x8 block", floorBlockDiff(lit, glow), 24.0);
        }

        // ── STRATA ──────────────────────────────────────────────────────────
        {
            constexpr int kWarmFrames = 30, kFrames = 240;
            std::vector<unsigned char> prev;
            std::vector<double> diffs;
            double mean = 0.0;
            for (int k = 0; k < kWarmFrames + kFrames; ++k) {
                const float t = static_cast<float>(k) / 60.f;
                strata.knot->rotation.y = 0.7f * t;
                strata.knot->rotation.x = 0.4f * t;
                frame(strata.scene);
                auto img = renderer.readRGBPixels();
                if (k < kWarmFrames) continue;
                if (!prev.empty()) diffs.push_back(floorDiff(img, prev));
                mean = floorMean(img);
                prev = std::move(img);
            }
            dump("strata_last");

            std::vector<double> sorted = diffs;
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2], maximum = sorted.back();
            std::printf("strata: a turning knot over a roughness-0.55 floor, change of the floor per frame\n");
            std::printf("  median %.3f, maximum %.3f; frames that changed by more than 3x the median:", median, maximum);
            int jumps = 0;
            for (size_t i = 0; i < diffs.size(); ++i) {
                if (diffs[i] > 3.0 * median) {
                    std::printf(" %zu(%.2f)", i + 1, diffs[i]);
                    ++jumps;
                }
            }
            std::printf("%s\n", jumps ? "" : " none");
            // The knot lights the floor at all, and it turns.
            check("mean level of the floor", mean, 4.0, true);
            check("median change per frame", median, 0.02, true);
            check("largest change per frame against the median", maximum / std::max(median, 1e-3), 3.0);
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("emitter specular: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
