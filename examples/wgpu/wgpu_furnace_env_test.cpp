// Classical white-furnace test (Option A — env-only).
//
// A diffuse-white sphere (albedo=1, metalness=0, roughness=1) sits alone in
// a constant-radiance environment (scene.environment = (1,1,1) everywhere).
// For a Lambertian surface with albedo 1 lit by uniform Le=1 over the full
// upper hemisphere, the reflected radiance is:
//
//     L_o = (ρ/π) * ∫ L_i cos θ dω = (1/π) * Le * π = Le = 1.0
//
// Every pixel covering the sphere must read exactly 1.0 — in one bounce,
// no convergence delay, no geometric form-factor concerns. Any deviation
// is a direct BRDF / throughput / sampling bug.
//
// Readback is 8-bit LDR so Le > 1 would clip; we keep Le = 1.
//
// The pathtracer renders to an offscreen RenderTarget; pixels are read
// back (readRGBPixels only works on a RenderTarget) and displayed via a
// fullscreen preview quad.
//
// See wgpu_furnace_test.cpp for the harder enclosed-box variant.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstring>

using namespace threepp;

namespace {

    // Tiny equirect HDR texture with every texel = (1,1,1,1).
    // scene.environment feeds the IBL sampling path at full strength,
    // so final L_env = 1.0 (per-texel value).
    std::shared_ptr<Texture> makeConstantEnv() {
        constexpr int W = 8, H = 4;// 2:1 equirect aspect
        std::vector<float> data(W * H * 4);
        for (int i = 0; i < W * H; ++i) {
            data[4 * i + 0] = 1.f;
            data[4 * i + 1] = 1.f;
            data[4 * i + 2] = 1.f;
            data[4 * i + 3] = 1.f;
        }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->encoding = Encoding::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    struct Stats {
        float meanR = 0.f, meanG = 0.f, meanB = 0.f;
        float mean = 0.f;
        float stddev = 0.f;
        int samples = 0;
        // Center-crop stats: pixels within 15% of min(w,h) radius from image center.
        // This circle is always inside the sphere's projection, so it's uncontaminated
        // by background env pixels (which sit around the sphere's perimeter).
        float cropMean = 0.f;
        float cropStddev = 0.f;
        int cropSamples = 0;
        // Single center pixel luminance — the purest single-point measurement.
        float centerLum = -1.f;
    };

    // All pixels with luminance > bgThreshold (includes sphere + env background at 1.0).
    // bgMean is contaminated upward by background; use cropMean for clean sphere measurement.
    Stats computeStats(const std::vector<unsigned char>& px, int w, int h, float bgThreshold = 0.01f) {
        Stats s{};
        if (px.empty() || w <= 0 || h <= 0) return s;

        // Center-crop radius: 15% of min dimension. Sphere projected radius is ~80%
        // of half-height, so r=0.15 is well inside and never clips background pixels.
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

            // All-pixel stats (contaminated by background).
            if (l >= bgThreshold) {
                sr += r; sg += g; sb += b; sl += l; sl2 += l * l;
                ++kept;
            }

            // Center-crop stats (sphere-only, no background contamination).
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
            s.mean  = static_cast<float>(sl / kept);
            const double var = sl2 / kept - (sl / kept) * (sl / kept);
            s.stddev = static_cast<float>(std::sqrt(std::max(0.0, var)));
            s.samples = static_cast<int>(kept);
        }
        if (ckept > 0) {
            s.cropMean   = static_cast<float>(csl / ckept);
            const double cvar = csl2 / ckept - (csl / ckept) * (csl / ckept);
            s.cropStddev = static_cast<float>(std::sqrt(std::max(0.0, cvar)));
            s.cropSamples = static_cast<int>(ckept);
        }
        return s;
    }

}// namespace

int main() {
    Canvas canvas("WgpuPathTracer — White Furnace (Env-only)", {{"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputEncoding = Encoding::Linear;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setMaxBounces(4);// env-only — 1 bounce suffices, extra headroom is free
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setReSTIREnabled(true);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setFireflyClamp(0);// 0 → 1e30 sentinel: no cap during furnace validation
    pathTracer.setExposure(1.f);

    // ── Offscreen target + preview quad ──
    auto [cw, ch] = canvas.size();
    auto rt = RenderTarget::create(static_cast<unsigned>(cw),
                                   static_cast<unsigned>(ch),
                                   RenderTarget::Options{});
    auto previewTex = DataTexture::create(3, cw, ch);
    previewTex->format = Format::RGBA;
    previewTex->magFilter = Filter::Linear;
    previewTex->minFilter = Filter::Linear;

    auto previewScene = Scene::create();
    auto previewCam = OrthographicCamera::create(-1, 1, 1, -1, 0, 1);
    auto previewMat = MeshBasicMaterial::create();
    previewMat->map = previewTex;
    previewScene->add(Mesh::create(PlaneGeometry::create(2, 2), previewMat));

    // ── Scene: constant-(1,1,1) env + white diffuse sphere ──
    Scene scene;
    auto envTex = makeConstantEnv();
    scene.environment = envTex;
    scene.background = envTex;// also show env as backdrop so "1.0" is obvious

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

    // ── State ──
    bool restirOn = true;
    int maxBounces = pathTracer.maxBounces();
    int measureEveryN = 16;
    float roughness = 1.0f;
    float metalness = 0.0f;
    int presetIdx = 0;// 0 = rough dielectric, 1 = mirror metal, 2 = rough metal, 3 = glossy dielectric
    Stats stats{};
    int statsFrame = -1;
    bool fresh = true;
    std::vector<unsigned char> pixels;
    std::vector<unsigned char> flipped;

    auto resetStats = [&] {
        stats = Stats{};
        statsFrame = -1;
        fresh = true;
    };
    auto resetAll = [&] {
        pathTracer.resetAccumulation();
        resetStats();
    };
    // Material uniforms are baked into matBuffer during atlas build; toggling
    // MeshStandardMaterial props at runtime does not propagate until we force
    // a topology rebuild via markDirty().
    auto applyMaterial = [&] {
        sphereMat->roughness = roughness;
        sphereMat->metalness = metalness;
        sphereMat->needsUpdate();
        pathTracer.markDirty();
        resetAll();
    };

    // ── ImGui ──
    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10});
        ImGui::SetNextWindowSize({380, 0});
        ImGui::Begin("White Furnace — Env-Only");

        ImGui::Text("Frames accumulated: %d", pathTracer.frameCount());
        if (!fresh) ImGui::Text("(measured @ frame %d, fg pixels=%d)", statsFrame, stats.samples);
        ImGui::Separator();

        ImGui::Text("Target: every sphere pixel = 1.000");
        if (fresh) {
            ImGui::TextDisabled("(accumulating... stats update every %d frames)", measureEveryN);
        } else {
            constexpr float tgt = 1.0f;

            // ── Center pixel (single unambiguous sphere measurement) ──
            if (stats.centerLum >= 0.f) {
                const float cDev = 100.f * (stats.centerLum - tgt) / tgt;
                ImVec4 cc = (std::abs(cDev) < 1.f)  ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
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
                ImVec4 crc = (std::abs(crDev) < 1.f)  ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
                             : (std::abs(crDev) < 5.f) ? ImVec4(1.f, 1.f, 0.4f, 1.f)
                                                       : ImVec4(1.f, 0.4f, 0.4f, 1.f);
                ImGui::Text("  Lum %.4f  stddev %.4f", stats.cropMean, stats.cropStddev);
                ImGui::TextColored(crc, "  Deviation: %+.2f%%", crDev);
            }

            ImGui::Separator();

            // ── All-pixel mean (contaminated by bg env at 1.0, shown for reference) ──
            const float devPct = 100.f * (stats.mean - tgt) / tgt;
            ImGui::TextDisabled("All-pixel mean (bg-contaminated, n=%d): %.4f  (%+.2f%%)",
                                stats.samples, stats.mean, devPct);
            ImGui::TextDisabled("Mean RGB: (%.4f, %.4f, %.4f)", stats.meanR, stats.meanG, stats.meanB);
            ImGui::TextDisabled("Stddev: %.4f", stats.stddev);
        }
        ImGui::Separator();

        bool dirty = false;
        if (ImGui::Checkbox("ReSTIR DI", &restirOn)) {
            pathTracer.setReSTIREnabled(restirOn);
            dirty = true;
        }
        if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 8)) {
            pathTracer.setMaxBounces(maxBounces);
            dirty = true;
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
        ImGui::SliderFloat("roughness", &roughness, 0.000f, 1.0f);
        ImGui::SliderFloat("metalness", &metalness, 0.0f, 1.0f);
        if (ImGui::Button("Apply material")) applyMaterial();
        ImGui::Separator();

        ImGui::SliderInt("Measure every N frames", &measureEveryN, 1, 256);
        if (ImGui::Button("Reset accumulation")) dirty = true;
        if (dirty) resetAll();

        ImGui::Separator();
        ImGui::TextDisabled("Method");
        ImGui::TextDisabled("  Env = constant (1,1,1) equirect");
        ImGui::TextDisabled("  Sphere: albedo=1, rough=1, metal=0");
        ImGui::TextDisabled("  Tonemap=None, Encoding=Linear");
        ImGui::TextDisabled("Interpretation");
        ImGui::TextDisabled("  mean ≠ 1.0 → BRDF/throughput bug");
        ImGui::TextDisabled("  (unlike the enclosed box, this is one-bounce)");

        ImGui::End();
    });

    IOCapture io;
    io.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&io);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        pathTracer.setSize({ns.width(), ns.height()});
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
        rt = RenderTarget::create(static_cast<unsigned>(ns.width()),
                                  static_cast<unsigned>(ns.height()),
                                  RenderTarget::Options{});
        previewTex = DataTexture::create(3, ns.width(), ns.height());
        previewTex->format = Format::RGB;
        previewTex->magFilter = Filter::Linear;
        previewTex->minFilter = Filter::Linear;
        previewMat->map = previewTex;
        previewMat->needsUpdate();
        resetStats();
    });

    Vector3 lastCamPos = camera.position;
    Vector3 lastCamTgt = controls.target;

    canvas.animate([&] {
        controls.update();

        if (!camera.position.equals(lastCamPos) || !controls.target.equals(lastCamTgt)) {
            lastCamPos = camera.position;
            lastCamTgt = controls.target;
            resetStats();
        }

        renderer.setRenderTarget(rt.get());
        pathTracer.render(scene, camera);

        pixels = renderer.readRGBPixels();
        const auto sz = canvas.size();
        const int w = sz.width();
        const int h = sz.height();

        if (!pixels.empty()) {
            const int rowBytes = w * 3;
            flipped.resize(pixels.size());
            for (int row = 0; row < h; ++row) {
                std::memcpy(&flipped[row * rowBytes],
                            &pixels[(h - 1 - row) * rowBytes],
                            rowBytes);
            }
            previewTex->image().setData(flipped);
            previewTex->needsUpdate();

            const int fc = pathTracer.frameCount();
            if (fc > 0 && fc >= measureEveryN && (fc % measureEveryN == 0) && fc != statsFrame) {
                stats = computeStats(pixels, w, h);
                statsFrame = fc;
                fresh = false;
            }
        }

        renderer.setRenderTarget(nullptr);
        renderer.render(*previewScene, *previewCam);

        ui.render();
    });
}
