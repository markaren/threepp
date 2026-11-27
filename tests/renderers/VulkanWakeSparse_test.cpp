// VulkanWakeSparse_test: the wake field's sparse step leaves the field a step
// over every texel leaves, texel for texel.
//
// A producer's patch is stepped over the 8 x 8 tiles whose water can change
// (wake_tiles.comp): those within reach of the water the last step left,
// those a source reaches, and those under ripples steep enough to break; the
// tiles it wrote last step and leaves out now are set to zero.
// THREEPP_WAKE_DENSE lists every tile. One scene runs three times, each with
// a new ocean: sparse, dense, and sparse again (the same steps twice, the
// null). Every producer's layer is read back at checkpoints, state and flow,
// and all three runs must agree exactly (as floats, so -0 == +0).
//
// What decides a tile is all in the scene:
//   * a ship whose thickness forces her patch's ripples (Waves::Thickness:
//     the flow her hull makes carries the water round her, wakeHullFlow),
//     with her bow wave breaking and two propellers, at 10 m/s on a circle in
//     a patch of 60 m (6 cm texels), so her patch moves by whole texels every
//     step, and her circle crosses the kayak's patch, where her sources lay
//     foam too;
//   * a planing boat on a jet that jumps 400 m ahead at one step: everything
//     her patch held is left behind at once and every tile of it is cleared;
//   * a kayak's paddles (VesselWash::put) at walking pace;
//   * a patch that is on and holds nothing;
//   * the ripples, breaking where their slope passes breakSlope0;
//   * steps of 1/60 s, some of 1/20 s, and one stall the field clamps to
//     0.25 s (the hull's flow carries the water farther in it).
// Checked by breaking wake_tiles.comp on purpose (2026-11-27): without the
// ripples' test the three vessels' patches differ, without the sources' test
// the boat's and the kayak's, without the clears the boat's from her jump on.
// With `reach` 0 (no margin round a tile) the runs still agree: in this scene
// neither the hull's flow nor a rounded tap carries water over the edge of a
// tile that nothing else lists.
// It prints the wake step's GPU time in each run, and the share of each
// layer's tiles that holds water at the end. `--dump DIR` writes the sparse
// run's last readout of each layer to DIR/layer<L>.f32: its state (4 floats a
// texel), then its flow (2), row-major.
//
// Plain exit-code program, not Catch2, the same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/extras/marine/VesselWash.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using marine::VesselWash;
using vktest::kSkipCode;

namespace {

    constexpr int      kW = 64, kH = 64;
    constexpr uint32_t kRes      = 1024;
    constexpr uint32_t kRipples  = 512;
    constexpr uint32_t kPatches  = 4;
    constexpr int      kFrames   = 360;
    constexpr int      kStall    = 150;// the step of 0.4 s
    constexpr int      kJump     = 200;// the boat jumps at this one
    constexpr std::array<int, 8> kCheckpoints = {30, 90, kStall, kStall + 1, kJump, kJump + 1, 260, kFrames - 1};

    // A level vessel at heading a: her bow along world (cos a, 0, -sin a), her
    // starboard side along (sin a, 0, cos a) (VesselWash_test's frames).
    struct Vessel {
        float   heading = 0.f;
        Vector3 origin;
        Vector3 velocity;

        [[nodiscard]] Vector3 forward() const { return {std::cos(heading), 0.f, -std::sin(heading)}; }
        [[nodiscard]] Vector3 starboard() const { return {std::sin(heading), 0.f, std::cos(heading)}; }
        [[nodiscard]] Vector3 world(float x, float y, float z) const {
            return origin + forward() * x + Vector3(0.f, y, 0.f) + starboard() * z;
        }
    };

    struct Layer {
        std::vector<float> state, velocity;
    };
    // [checkpoint][layer]
    using Readout = std::vector<std::vector<Layer>>;

    struct Run {
        Readout readout;
        double  wakeMs = 0.0;// median over the frames after the first 60
        bool    ok     = true;
    };

    double dtOf(int frame) {
        if (frame == kStall) return 0.4;
        return frame % 37 == 0 ? 1.0 / 20.0 : 1.0 / 60.0;
    }

    Run runScene(Canvas& canvas, VulkanRenderer& renderer, std::vector<std::shared_ptr<Ocean>>& keep,
                 const char* name) {
        Run run;
        Scene scene;
        PerspectiveCamera camera(50.f, float(kW) / kH, 0.5f, 3000.f);
        camera.position.set(0.f, 400.f, 300.f);
        camera.lookAt(Vector3{0.f, 0.f, 0.f});

        Ocean::Options opt;
        opt.waveScale = 0.5f;
        auto ocean = Ocean::create(opt);
        ocean->wakeField.resolution       = kRes;
        ocean->wakeField.patches          = kPatches;
        ocean->wakeField.rippleResolution = kRipples;
        ocean->wakeField.rippleHeights    = true;
        ocean->wakeField.rippleBreaking   = true;
        scene.add(ocean);
        keep.push_back(ocean);// (alive to the end: the renderer keys its state by the mesh)

        // The ship: her thickness from a trapezoid of her footprint.
        VesselWash::Hull ship;
        ship.length     = 40.f;
        ship.halfLength = 20.f;
        ship.halfBeam   = 5.f;
        ship.draft      = 3.f;
        ship.mass       = 1.2e6f;
        ship.prismatic  = 0.7f;
        ship.waves      = VesselWash::Waves::Thickness;
        VesselWash shipWash(*ocean, 0, ship, 60.f);// a tight patch: 6 cm texels, which the hull's flow crosses fast
        // The boat: her weight from her footprint, on a jet.
        VesselWash::Hull boat;
        boat.length     = 9.f;
        boat.halfLength = 4.5f;
        boat.halfBeam   = 1.5f;
        boat.mass       = 4000.f;
        VesselWash boatWash(*ocean, 1, boat);
        // The kayak, and her paddles.
        VesselWash::Hull kayak;
        kayak.length     = 4.f;
        kayak.halfLength = 2.f;
        kayak.halfBeam   = 0.35f;
        kayak.mass       = 100.f;
        VesselWash kayakWash(*ocean, 2, kayak);
        // On, with nothing in it.
        ocean->wakePatches[3].centerX = 300.f;
        ocean->wakePatches[3].centerZ = 300.f;
        ocean->wakePatches[3].size    = 64.f;

        Vessel shipV, boatV, kayakV;
        boatV.origin  = Vector3(-200.f, 0.f, 150.f);
        boatV.heading = 0.1f;
        kayakV.origin  = Vector3(60.f, 0.f, -90.f);
        kayakV.heading = 2.0f;
        const double R = 120.0, U = 10.0;// the ship's circle (m) and speed (m/s)

        std::vector<double> wakeMs;
        double t = 1.0;
        size_t next = 0;
        for (int f = 0; f < kFrames; ++f) {
            const double dt = dtOf(f);
            t += dt;
            const float fdt = static_cast<float>(dt);

            const double ang = U * (t - 1.0) / R;
            shipV.origin   = Vector3(float(R * std::cos(ang)), 0.f, float(-R * std::sin(ang)));
            shipV.heading  = float(ang + 0.5 * 3.14159265358979);
            shipV.velocity = shipV.forward() * float(U);

            boatV.velocity = boatV.forward() * 14.f;
            boatV.origin += boatV.velocity * fdt;
            if (f == kJump) boatV.origin += boatV.forward() * 400.f;

            kayakV.velocity = kayakV.forward() * 1.5f;
            kayakV.origin += kayakV.velocity * fdt;

            ocean->clearWakeSources();
            const Vector3 thrust = shipV.forward() * 150000.f;
            shipWash.update(fdt, shipV.origin, shipV.forward(), shipV.velocity,
                            {{"prop_s", shipV.world(-18.f, -2.5f, 2.5f), thrust, 1.2f, false},
                             {"prop_p", shipV.world(-18.f, -2.5f, -2.5f), thrust, 1.2f, false}});
            boatWash.update(fdt, boatV.origin, boatV.forward(), boatV.velocity,
                            {{"jet", boatV.world(-4.5f, -0.3f, 0.f), boatV.forward() * 8000.f, 0.15f, true}});
            kayakWash.update(fdt, kayakV.origin, kayakV.forward(), kayakV.velocity);
            // a stroke a second, a side at a time
            const float side = std::fmod(float(t), 2.f) < 1.f ? 1.f : -1.f;
            if (std::fmod(float(t), 1.f) < 0.6f) {
                kayakWash.put("paddle", kayakV.world(0.3f, 0.f, 0.9f * side), kayakV.forward() * -1.2f, 0.15f, 0.25f,
                              0.3f, 0.4f, 0.f, 40.f);
            }

            renderer.setSimTime(t);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
            if (f >= 60) wakeMs.push_back(renderer.lastFrameTimings().oceanWakeMs);

            if (next < kCheckpoints.size() && f == kCheckpoints[next]) {
                std::vector<Layer> layers(kPatches);
                for (uint32_t l = 0; l < kPatches; ++l) {
                    uint32_t n = 0;
                    if (!renderer.readWakeField(*ocean, l, layers[l].state, layers[l].velocity, n) || n != kRes) {
                        std::printf("  %s: layer %u not read back at step %d\n", name, l, f);
                        run.ok = false;
                    }
                }
                run.readout.push_back(std::move(layers));
                ++next;
            }
        }
        scene.remove(*ocean);
        // (the last two are still in flight when the run ends)
        if (!wakeMs.empty()) {
            std::nth_element(wakeMs.begin(), wakeMs.begin() + wakeMs.size() / 2, wakeMs.end());
            run.wakeMs = wakeMs[wakeMs.size() / 2];
        }
        return run;
    }

    bool same(float a, float b) {
        return a == b || (std::isnan(a) && std::isnan(b));
    }

    // Texels of `b` that differ from `a`, at every checkpoint and layer.
    int compare(const Run& a, const Run& b, const char* what) {
        int bad = 0;
        for (size_t c = 0; c < a.readout.size() && c < b.readout.size(); ++c) {
            for (uint32_t l = 0; l < kPatches; ++l) {
                const Layer& x = a.readout[c][l];
                const Layer& y = b.readout[c][l];
                size_t diff = 0;
                double worst = 0.0;
                for (size_t i = 0; i < x.state.size() && i < y.state.size(); ++i) {
                    if (!same(x.state[i], y.state[i])) {
                        ++diff;
                        worst = std::max(worst, double(std::abs(x.state[i] - y.state[i])));
                    }
                }
                for (size_t i = 0; i < x.velocity.size() && i < y.velocity.size(); ++i) {
                    if (!same(x.velocity[i], y.velocity[i])) {
                        ++diff;
                        worst = std::max(worst, double(std::abs(x.velocity[i] - y.velocity[i])));
                    }
                }
                if (x.state.size() != y.state.size() || x.velocity.size() != y.velocity.size()) diff = 1;
                if (diff > 0) {
                    std::printf("  %s, step %d, layer %u: %zu values differ (largest by %.3g)\n", what,
                                kCheckpoints[c], l, diff, worst);
                    ++bad;
                }
            }
        }
        if (a.readout.size() != b.readout.size()) ++bad;
        std::printf("%-44s %s\n", what, bad == 0 ? "identical" : "FAIL");
        return bad;
    }

}// namespace

int main(int argc, char** argv) {

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dump;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dump") dump = argv[i + 1];
    }

    auto gpu = vktest::openHeadlessVulkan("VulkanWakeSparse_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;
    vktest::pinNativeOutput(renderer);

    int failures = 0;
    try {
        std::vector<std::shared_ptr<Ocean>> keep;
        vktest::setEnv("THREEPP_WAKE_DENSE", "");// (empty: the default, off)
        const Run sparse = runScene(canvas, renderer, keep, "sparse");
        vktest::setEnv("THREEPP_WAKE_DENSE", "1");
        const Run dense = runScene(canvas, renderer, keep, "dense");
        vktest::setEnv("THREEPP_WAKE_DENSE", "");
        const Run again = runScene(canvas, renderer, keep, "sparse again");
        if (!sparse.ok || !dense.ok || !again.ok) ++failures;

        // The field is not trivially still: each vessel's patch holds water.
        const auto& last = sparse.readout.back();
        std::printf("tiles holding water at step %d:", kCheckpoints.back());
        for (uint32_t l = 0; l < kPatches; ++l) {
            if (last[l].state.size() != size_t(kRes) * kRes * 4u || last[l].velocity.size() != size_t(kRes) * kRes * 2u) {
                std::printf("  layer %u (FAIL: not read back)", l);
                ++failures;
                continue;
            }
            const uint32_t side = kRes / 8u;
            size_t wet = 0;
            for (uint32_t ty = 0; ty < side; ++ty) {
                for (uint32_t tx = 0; tx < side; ++tx) {
                    bool any = false;
                    for (uint32_t y = 0; y < 8 && !any; ++y) {
                        for (uint32_t x = 0; x < 8 && !any; ++x) {
                            const size_t i = size_t(ty * 8 + y) * kRes + tx * 8 + x;
                            for (int c = 0; c < 4; ++c) any = any || last[l].state[4 * i + c] != 0.f;
                            any = any || last[l].velocity[2 * i] != 0.f || last[l].velocity[2 * i + 1] != 0.f;
                        }
                    }
                    wet += any ? 1u : 0u;
                }
            }
            std::printf("  layer %u %.1f %%", l, 100.0 * double(wet) / double(side * side));
            if (l < 3 && wet == 0) {
                std::printf(" (FAIL: a vessel's patch holds nothing)");
                ++failures;
            }
            if (l == 3 && wet != 0) {
                std::printf(" (FAIL: the empty patch holds water)");
                ++failures;
            }
        }
        std::printf("\n");

        if (!dump.empty()) {
            std::filesystem::create_directories(dump);
            for (uint32_t l = 0; l < kPatches; ++l) {
                std::ofstream f(std::filesystem::path(dump) / ("layer" + std::to_string(l) + ".f32"), std::ios::binary);
                f.write(reinterpret_cast<const char*>(last[l].state.data()), std::streamsize(last[l].state.size() * 4));
                f.write(reinterpret_cast<const char*>(last[l].velocity.data()), std::streamsize(last[l].velocity.size() * 4));
            }
            std::printf("the sparse run's last readout is in %s\n", dump.c_str());
        }

        failures += compare(sparse, dense, "sparse against dense");
        failures += compare(sparse, again, "sparse against sparse again (the null)");
        std::printf("wake step (GPU, median): sparse %.3f ms, dense %.3f ms, sparse again %.3f ms\n", sparse.wakeMs,
                    dense.wakeMs, again.wakeMs);
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("wake sparse: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
