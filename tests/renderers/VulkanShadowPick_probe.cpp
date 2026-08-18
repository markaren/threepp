// A/B probe for the analyticDirectSplit light-pick rewrite (streaming
// reservoir replacing the stored-CDF walk).
//
// WHY THIS EXISTS: the committed golden scenes (VulkanGolden_test) contain
// ZERO analytic lights — they are IBL- and emissive-lit — so analyticDirectSplit
// early-outs at `wSum <= 1e-8` and its pick logic never executes. Their passing
// says nothing about this change. This scene is built to hit the path hard:
// a sun plus 6 point lights over shadow casters, denoise + ReSTIR DI on, so
// every lit pixel runs the multi-candidate pick and 1-2 shadow rays.
//
// Usage:
//   VulkanShadowPick_probe --write ref.ppm      (build A, e.g. stashed shader)
//   VulkanShadowPick_probe --compare ref.ppm    (build B, prints PSNR/bands)
//
// Interpretation: the two shaders draw DIFFERENT light picks from the same
// blue noise (same distribution, different realization), so a per-pixel diff
// is EXPECTED to be non-zero in penumbrae. What must hold is that the
// converged image agrees: the denoised means match to within the estimator's
// own frame-to-frame noise. The harness therefore also reports a
// same-build consecutive-frame delta as the noise floor to judge against —
// an A/B delta at or below that floor is indistinguishable from re-running
// the same shader twice.

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // Overridable so the same binary can be run at a resolution where the
    // shade pass dominates the frame — the register-pressure risk of any
    // occupancy change only shows up under real load.
    inline int envInt(const char* k, int dflt) {
        const char* v = std::getenv(k);
        return (v && *v) ? std::atoi(v) : dflt;
    }
    const int kW = envInt("PROBE_W", 480), kH = envInt("PROBE_H", 320);
    constexpr int kSkipCode = 42;
    constexpr int kWarmFrames = 96;// let the temporal accumulators converge

    void writePPM(const std::string& path, const std::vector<unsigned char>& px, int w, int h) {
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size()));
    }

    bool readPPM(const std::string& path, std::vector<unsigned char>& px, int& w, int& h) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::string magic;
        int maxv = 0;
        f >> magic >> w >> h >> maxv;
        f.get();
        if (magic != "P6") return false;
        px.resize(size_t(w) * h * 3);
        f.read(reinterpret_cast<char*>(px.data()), std::streamsize(px.size()));
        return bool(f);
    }

    // Sun + 6 point lights over shadow casters on a floor: multi-candidate
    // picks everywhere, hard and soft shadow overlap, and a sun share that
    // varies across the frame (so the sun-dominance gate is exercised both
    // ways — it reads the weight the rewrite now tracks in `sunW`).
    void buildScene(Scene& scene, PerspectiveCamera& cam) {
        scene.background = Color(0.02f, 0.025f, 0.035f);

        auto floorMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.55f, 0.55f, 0.58f)).roughness(0.8f).metalness(0.f));
        auto floor = Mesh::create(BoxGeometry::create(30.f, 0.4f, 30.f), floorMat);
        floor->position.set(0.f, -0.2f, 0.f);
        floor->receiveShadow = true;
        scene.add(floor);

        for (int i = 0; i < 9; ++i) {
            const float a = float(i) * 0.7f;
            auto m = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                          .color(Color(0.8f, 0.75f, 0.7f))
                                                          .roughness(0.25f + 0.06f * float(i % 4))
                                                          .metalness(i % 3 == 0 ? 1.f : 0.f));
            auto box = Mesh::create(BoxGeometry::create(0.8f, 1.6f + 0.3f * float(i % 3), 0.8f), m);
            box->position.set(std::cos(a) * (2.f + 0.35f * float(i)), 0.8f, std::sin(a) * (2.f + 0.35f * float(i)));
            box->castShadow = true;
            box->receiveShadow = true;
            scene.add(box);
        }

        // Dir light 0 = the sun (EnvSunPolicy) — the sun-dominance gate reads
        // exactly this light's weight share.
        auto sun = DirectionalLight::create(Color(1.f, 0.96f, 0.88f), 2.2f);
        sun->position.set(4.f, 6.f, 3.f);
        sun->castShadow = true;
        scene.add(sun);

        // Point lights: cluster candidates, so cCnt > 1 on most lit pixels and
        // the streaming pick has real competition to resolve.
        const Color tints[6] = {Color(1.f, 0.35f, 0.2f), Color(0.2f, 0.5f, 1.f), Color(0.3f, 1.f, 0.4f),
                                Color(1.f, 0.85f, 0.3f), Color(0.9f, 0.3f, 0.9f), Color(0.3f, 0.9f, 0.9f)};
        for (int i = 0; i < 6; ++i) {
            const float a = float(i) * 1.05f;
            auto p = PointLight::create(tints[i], 12.f, 14.f);
            p->position.set(std::cos(a) * 3.6f, 1.9f + 0.5f * float(i % 3), std::sin(a) * 3.6f);
            p->castShadow = true;
            scene.add(p);
        }

        cam.position.set(0.f, 3.4f, 8.2f);
        cam.lookAt(Vector3(0.f, 0.9f, 0.f));
    }

}// namespace

int main(int argc, char** argv) {
    std::string writePath, comparePath;
    bool timeMode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--write") == 0 && i + 1 < argc) writePath = argv[++i];
        else if (std::strcmp(argv[i], "--compare") == 0 && i + 1 < argc) comparePath = argv[++i];
        else if (std::strcmp(argv[i], "--time") == 0) timeMode = true;
    }
    if (writePath.empty() && comparePath.empty() && !timeMode) {
        std::printf("usage: --write <ref.ppm> | --compare <ref.ppm>\n");
        return 2;
    }

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanShadowPick_probe").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    // Same pinning as the goldens: plain TAA resolve, no upscaler variance.
    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    renderer.setClearColor(Color(0.f, 0.f, 0.f));

    Scene scene;
    PerspectiveCamera cam(50.f, float(kW) / float(kH), 0.1f, 200.f);
    buildScene(scene, cam);

    // Deterministic frame clock: the temporal accumulators must land in the
    // same state in both builds, or the diff measures TAA phase, not the pick.
    std::vector<unsigned char> prev;
    for (int i = 0; i < kWarmFrames; ++i) {
        renderer.setSimTime(double(i) / 60.0);
        canvas.animateOnce([&] { renderer.render(scene, cam); });
        if (i == kWarmFrames - 2) prev = renderer.readRGBPixels();
    }
    if (timeMode) {
        // Median of per-frame wall clock after the warm-up above. Reported so
        // the register-count delta can be judged against what it costs.
        std::vector<double> ms;
        for (int i = 0; i < 300; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            renderer.setSimTime(double(kWarmFrames + i) / 60.0);
            canvas.animateOnce([&] { renderer.render(scene, cam); });
            const auto t1 = std::chrono::high_resolution_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        std::printf("[time] median %.3f ms  p10 %.3f  p90 %.3f  (n=%zu)\n",
                    ms[ms.size() / 2], ms[ms.size() / 10], ms[ms.size() * 9 / 10], ms.size());
        return 0;
    }

    auto px = renderer.readRGBPixels();

    // Same-build consecutive-frame delta = the estimator's own noise floor.
    const auto floorDiff = capture::imageDiff(prev, px);
    std::printf("[noise-floor] consecutive frames, same build: PSNR=%5.1f dB maxD=%3d hot=%6.3f%%\n",
                floorDiff.psnr, floorDiff.maxD, floorDiff.hotPct);

    if (!writePath.empty()) {
        writePPM(writePath, px, kW, kH);
        std::printf("wrote %s (%dx%d)\n", writePath.c_str(), kW, kH);
        return 0;
    }

    std::vector<unsigned char> ref;
    int rw = 0, rh = 0;
    if (!readPPM(comparePath, ref, rw, rh) || rw != kW || rh != kH) {
        std::printf("[fail] could not read reference %s\n", comparePath.c_str());
        return 1;
    }
    const auto d = capture::imageDiff(px, ref);
    std::printf("[A-vs-B]      old shader vs new:          PSNR=%5.1f dB maxD=%3d hot=%6.3f%%\n",
                d.psnr, d.maxD, d.hotPct);
    capture::diffStats(px, ref, kW, kH, 3, "A-vs-B");
    std::printf("VERDICT: A/B PSNR %.1f dB vs noise floor %.1f dB -> %s\n",
                d.psnr, floorDiff.psnr,
                d.psnr >= floorDiff.psnr - 3.0 ? "indistinguishable from re-running the same shader"
                                               : "LOOK REQUIRED: differs beyond the estimator's own noise");
    return 0;
}
