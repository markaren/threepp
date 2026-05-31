// Classical white-furnace test for the Vulkan path tracer (env-only).
// Port of the (removed) examples/wgpu/wgpu_furnace_env_test.cpp.
//
// A diffuse-white sphere (albedo=1, metalness=0, roughness=1) sits alone in a
// constant-radiance environment (scene.environment = (1,1,1) everywhere). For
// a Lambertian surface with albedo 1 lit by uniform Le=1 over the hemisphere:
//
//     L_o = (ρ/π) · ∫ L_i cos θ dω = (1/π) · Le · π = Le = 1.0
//
// So every pixel covering the sphere must read exactly 1.0. Any deviation is a
// direct BRDF / throughput / sampling bug (energy gain or loss). The white
// furnace is the canonical correctness check for a path tracer's BSDF — now
// the more important since the Vulkan PT is the project's sole path tracer and
// ground-truth source.
//
// Tonemap=None + outputColorSpace=Linear → readback bytes are raw linear, so
// 1.0 ↔ 255. Readback uses the renderer's scene-capture buffer
// (setSceneCaptureEnabled / readSceneRGBPixels): the post-TAA, *pre-overlay*
// swapchain image, so the measurement is never contaminated by the ImGui HUD.
//
// VulkanRenderer differences vs the WGPU original:
//   • It IS the path tracer — no `usePathTracer` toggle, no separate
//     `pathTracer()` object; settings live directly on the renderer.
//   • No `frameCount()` accessor — we track accumulated frames ourselves
//     (reset on camera move / material change, mirroring the renderer's own
//     accumulation reset).
//   • Clean readback via scene capture instead of an offscreen RenderTarget +
//     fullscreen preview quad.
//   • Runtime material edits propagate via Material::needsUpdate() (the
//     renderer re-uploads on a material->version() bump) — Vulkan has no
//     markDirty() and needs none.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace threepp;

namespace {

    // Tiny equirect HDR texture with every texel = (1,1,1,1). scene.environment
    // feeds the IBL sampling path at full strength, so final L_env = 1.0.
    std::shared_ptr<Texture> makeConstantEnv() {
        constexpr int W = 8, H = 4;// 2:1 equirect aspect
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 1.f);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    struct Stats {
        float meanR = 0.f, meanG = 0.f, meanB = 0.f;
        float mean = 0.f;
        float stddev = 0.f;
        int samples = 0;
        // Center-crop stats: pixels within 15% of min(w,h) radius from image
        // center. That circle is always inside the sphere's projection, so it's
        // uncontaminated by the background env pixels around the perimeter.
        float cropMean = 0.f;
        float cropStddev = 0.f;
        int cropSamples = 0;
        // Single center pixel luminance — the purest single-point measurement.
        float centerLum = -1.f;
    };

    // px is tightly-packed RGB (3 bytes/pixel). All pixels with luminance >
    // bgThreshold feed the (background-contaminated) all-pixel mean; the
    // center-crop circle feeds the clean sphere-only mean.
    Stats computeStats(const std::vector<unsigned char>& px, int w, int h, float bgThreshold = 0.01f) {
        Stats s{};
        if (px.empty() || w <= 0 || h <= 0) return s;
        if (px.size() < static_cast<size_t>(w) * h * 3) return s;

        const float cropR = 0.15f * static_cast<float>(std::min(w, h));
        const float cropR2 = cropR * cropR;
        const float cx = (w - 1) * 0.5f;
        const float cy = (h - 1) * 0.5f;

        // Center pixel.
        {
            const int px_x = w / 2;
            const int px_y = h / 2;
            const size_t idx = (static_cast<size_t>(px_y) * w + px_x) * 3;
            if (idx + 2 < px.size()) {
                const double r = px[idx + 0] / 255.0;
                const double g = px[idx + 1] / 255.0;
                const double b = px[idx + 2] / 255.0;
                s.centerLum = static_cast<float>(0.2126 * r + 0.7152 * g + 0.0722 * b);
            }
        }

        double sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0;
        size_t kept = 0;
        double csl = 0, csl2 = 0;
        size_t ckept = 0;

        const size_t n = px.size() / 3;
        for (size_t i = 0; i < n; ++i) {
            const double r = px[3 * i + 0] / 255.0;
            const double g = px[3 * i + 1] / 255.0;
            const double b = px[3 * i + 2] / 255.0;
            const double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            if (l >= bgThreshold) {
                sr += r; sg += g; sb += b; sl += l; sl2 += l * l;
                ++kept;
            }

            const float px_x = static_cast<float>(i % w);
            const float px_y = static_cast<float>(i / w);
            const float dx = px_x - cx;
            const float dy = px_y - cy;
            if (dx * dx + dy * dy <= cropR2) {
                csl += l; csl2 += l * l;
                ++ckept;
            }
        }

        if (kept > 0) {
            s.meanR = static_cast<float>(sr / kept);
            s.meanG = static_cast<float>(sg / kept);
            s.meanB = static_cast<float>(sb / kept);
            s.mean = static_cast<float>(sl / kept);
            const double var = sl2 / kept - (sl / kept) * (sl / kept);
            s.stddev = static_cast<float>(std::sqrt(std::max(0.0, var)));
            s.samples = static_cast<int>(kept);
        }
        if (ckept > 0) {
            s.cropMean = static_cast<float>(csl / ckept);
            const double cvar = csl2 / ckept - (csl / ckept) * (csl / ckept);
            s.cropStddev = static_cast<float>(std::sqrt(std::max(0.0, cvar)));
            s.cropSamples = static_cast<int>(ckept);
        }
        return s;
    }

}// namespace

int main() {
    Canvas canvas("Vulkan PT - White Furnace (Env-only)", {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    // Raw linear readback: 1.0 ↔ 255, no tone curve, no sRGB encode.
    renderer.outputColorSpace = ColorSpace::Linear;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    // Measurement config: denoiser OFF (it biases/blurs the per-pixel mean),
    // firefly clamp OFF (a furnace must not clamp anything), env-only so a few
    // bounces are plenty.
    renderer.setDenoise(false);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(0.f);// 0 → 1e30 sentinel: no cap during validation
    renderer.setMaxBounces(4);

    // Clean, overlay-free readback of the post-TAA scene image.
    renderer.setSceneCaptureEnabled(true);

    // ── Scene: constant-(1,1,1) env + white diffuse sphere ──
    Scene scene;
    auto envTex = makeConstantEnv();
    scene.environment = envTex;
    scene.background = envTex;// show env as backdrop so "1.0" everywhere is obvious

    auto sphereMat = MeshStandardMaterial::create({{"color", Color(1.f, 1.f, 1.f)},
                                                   {"roughness", 1.f},
                                                   {"metalness", 0.f}});
    auto sphere = Mesh::create(SphereGeometry::create(1.f, 64, 64), sphereMat);
    scene.add(sphere);

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.01f, 100.f);
    camera.position.set(0.f, 0.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();

    // ── UI / measurement state ──
    bool denoiserOn = renderer.denoise();
    bool restirOn = renderer.restirDIEnabled();
    bool restirGiOn = renderer.restirGIEnabled();
    bool hybridOn = renderer.hybridEnabled();
    bool taaOn = renderer.taaEnabled();
    int spp = renderer.samplesPerPixel();
    int maxBounces = renderer.maxBounces();
    int measureEveryN = 32;
    float roughness = 1.0f;
    float metalness = 0.0f;
    int presetIdx = 0;// 0 rough dielectric, 1 mirror metal, 2 rough metal, 3 glossy dielectric

    // We track our own accumulated-frame counter because VulkanRenderer has no
    // frameCount(); it advances each rendered frame and resets whenever the
    // image must re-converge (camera move, material edit, config toggle).
    int accumFrames = 0;
    Stats stats{};
    int statsFrame = -1;
    bool fresh = true;
    bool pendingReset = false;// deferred: resetAccumulation() must run outside render()
    std::vector<unsigned char> pixels;

    auto resetMeasurement = [&] {
        accumFrames = 0;
        stats = Stats{};
        statsFrame = -1;
        fresh = true;
    };
    auto requestReset = [&] {
        pendingReset = true;
        resetMeasurement();
    };
    auto applyMaterial = [&] {
        sphereMat->roughness = roughness;
        sphereMat->metalness = metalness;
        sphereMat->needsUpdate();// renderer re-uploads on the version bump
        requestReset();
    };

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({390, 0}, ImGuiCond_Once);
        ImGui::Begin("White Furnace - Env-Only");

        ImGui::Text("Frames accumulated: %d", accumFrames);
        if (!fresh) ImGui::Text("(measured @ frame %d, fg pixels=%d)", statsFrame, stats.samples);
        ImGui::Separator();

        ImGui::Text("Target: every sphere pixel = 1.000");
        if (fresh) {
            ImGui::TextDisabled("(accumulating... stats update every %d frames)", measureEveryN);
        } else {
            constexpr float tgt = 1.0f;

            // ── Center pixel (single unambiguous sphere sample) ──
            if (stats.centerLum >= 0.f) {
                const float cDev = 100.f * (stats.centerLum - tgt) / tgt;
                const ImVec4 cc = (std::abs(cDev) < 1.f)   ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
                                  : (std::abs(cDev) < 5.f) ? ImVec4(1.f, 1.f, 0.4f, 1.f)
                                                           : ImVec4(1.f, 0.4f, 0.4f, 1.f);
                ImGui::Text("Center pixel lum: %.4f", stats.centerLum);
                ImGui::TextColored(cc, "  Deviation: %+.2f%%", cDev);
            }

            ImGui::Separator();

            // ── Center-crop (15%% radius, sphere-only, uncontaminated) ──
            ImGui::Text("Center-crop mean (sphere-only, n=%d):", stats.cropSamples);
            if (stats.cropSamples > 0) {
                const float crDev = 100.f * (stats.cropMean - tgt) / tgt;
                const ImVec4 crc = (std::abs(crDev) < 1.f)   ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
                                   : (std::abs(crDev) < 5.f) ? ImVec4(1.f, 1.f, 0.4f, 1.f)
                                                             : ImVec4(1.f, 0.4f, 0.4f, 1.f);
                ImGui::Text("  Lum %.4f  stddev %.4f", stats.cropMean, stats.cropStddev);
                ImGui::TextColored(crc, "  Deviation: %+.2f%%", crDev);
            }

            ImGui::Separator();

            // ── All-pixel mean (contaminated by bg env at 1.0; reference only) ──
            const float devPct = 100.f * (stats.mean - tgt) / tgt;
            ImGui::TextDisabled("All-pixel mean (bg-contaminated, n=%d): %.4f  (%+.2f%%)",
                                stats.samples, stats.mean, devPct);
            ImGui::TextDisabled("Mean RGB: (%.4f, %.4f, %.4f)", stats.meanR, stats.meanG, stats.meanB);
            ImGui::TextDisabled("Stddev: %.4f", stats.stddev);
        }
        ImGui::Separator();

        // ── Render-path toggles (energy conservation should hold for all) ──
        if (ImGui::Checkbox("Denoiser", &denoiserOn)) {
            renderer.setDenoise(denoiserOn);
            requestReset();
        }
        if (ImGui::Checkbox("ReSTIR DI", &restirOn)) {
            renderer.setRestirDIEnabled(restirOn);
            requestReset();
        }
        if (ImGui::Checkbox("ReSTIR GI", &restirGiOn)) {
            renderer.setRestirGIEnabled(restirGiOn);
            requestReset();
        }
        if (ImGui::Checkbox("Hybrid raster G-buffer", &hybridOn)) {
            renderer.setHybridEnabled(hybridOn);
            requestReset();
        }
        if (!hybridOn) {
            if (ImGui::Checkbox("TAA (standalone)", &taaOn)) {
                renderer.setTaaEnabled(taaOn);
                requestReset();
            }
        }
        if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 8)) {
            renderer.setMaxBounces(maxBounces);
            requestReset();
        }
        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 8)) {
            renderer.setSamplesPerPixel(spp);
            requestReset();
        }

        ImGui::Separator();
        ImGui::Text("Material preset");
        const char* presets[] = {"rough dielectric (1.0 / 0.0)",
                                 "mirror metal (0.02 / 1.0)",
                                 "rough metal (1.0 / 1.0)",
                                 "glossy dielectric (0.1 / 0.0)"};
        if (ImGui::Combo("##preset", &presetIdx, presets, IM_ARRAYSIZE(presets))) {
            switch (presetIdx) {
                case 0: roughness = 1.0f;  metalness = 0.0f; break;
                case 1: roughness = 0.02f; metalness = 1.0f; break;
                case 2: roughness = 1.0f;  metalness = 1.0f; break;
                case 3: roughness = 0.1f;  metalness = 0.0f; break;
            }
            applyMaterial();
        }
        ImGui::SliderFloat("roughness", &roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("metalness", &metalness, 0.0f, 1.0f);
        if (ImGui::Button("Apply material")) applyMaterial();

        ImGui::Separator();
        ImGui::SliderInt("Measure every N frames", &measureEveryN, 1, 256);
        if (ImGui::Button("Reset accumulation")) requestReset();

        ImGui::Separator();
        ImGui::TextDisabled("Method");
        ImGui::TextDisabled("  Env = constant (1,1,1) equirect");
        ImGui::TextDisabled("  Sphere: albedo=1, rough=1, metal=0");
        ImGui::TextDisabled("  Tonemap=None, Encoding=Linear, clamp off");
        ImGui::TextDisabled("Interpretation");
        ImGui::TextDisabled("  mean != 1.0 -> BRDF/throughput bug");
        ImGui::TextDisabled("  (one-bounce; metals/glossy stress other lobes)");

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = []() -> bool { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
        resetMeasurement();
    });

    Vector3 lastCamPos = camera.position;
    Vector3 lastCamTgt = controls.target;

    canvas.animate([&] {
        // Deferred accumulation reset (resetAccumulation issues a
        // vkDeviceWaitIdle and must not run inside render()).
        if (pendingReset) {
            renderer.resetAccumulation();
            pendingReset = false;
        }

        controls.update();

        // Camera motion resets the renderer's own accumulation; keep our frame
        // counter and stats in sync so we never measure a half-converged frame.
        if (!camera.position.equals(lastCamPos) || !controls.target.equals(lastCamTgt)) {
            lastCamPos = camera.position;
            lastCamTgt = controls.target;
            resetMeasurement();
        }

        renderer.render(scene, camera);
        ++accumFrames;

        // Clean (overlay-free) scene readback; measure periodically.
        pixels = renderer.readSceneRGBPixels();
        const auto sz = canvas.size();
        const int w = sz.width();
        const int h = sz.height();
        // The scene-capture buffer is at the swapchain extent, which equals
        // canvas.size() at the default pixel ratio. If they ever diverge (HiDPI
        // scaling), skip the measurement rather than read with a wrong stride.
        const bool sizeMatches = pixels.size() == static_cast<size_t>(w) * h * 3;
        if (sizeMatches && accumFrames >= measureEveryN &&
            (accumFrames % measureEveryN == 0) && accumFrames != statsFrame) {
            stats = computeStats(pixels, w, h);
            statsFrame = accumFrames;
            fresh = false;
        }

        ui.render();
    });

    return 0;
}
