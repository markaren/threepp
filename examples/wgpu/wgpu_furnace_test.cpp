// Enclosed-box energy-conservation test for WgpuPathTracer (Option B).
//
// Setup: a fully enclosed room of pure-white Lambertian walls (albedo = 1)
// containing a single emissive sphere (zero albedo, Le = 1). No environment
// light, no other emitters.
//
// For a closed diffuse cavity with wall albedo ρ and emitter fractional area
// f_e (ρ_e = 0), the integrating-sphere balance gives wall radiance
//     L_wall = Le · f_e / (1 - ρ · (1 - f_e)).
// With ρ = 1 this reduces to L_wall = Le exactly, independent of geometry.
// So the mean foreground radiance should converge to Le = 1.0.
//
// NOTE — this converges slowly: ρ = 1 makes the Neumann series barely decay
// (only the emitter's absorption damps it). In practice you need high max
// bounces (≥ 48) and many frames before the mean approaches 1.0. A single
// per-bounce ~4–5 % leak compounds across 24 bounces to ~30 % shortfall.
//
// For a one-bounce, no-convergence sanity test, see wgpu_furnace_env_test.cpp
// (Option A — env-only). If Option A passes but this one doesn't, the bug is
// compounded per bounce (missing (1-F) term, misweighted lobe-split, etc.).
//
// Readback is 8-bit LDR so emitter Le > 1 would clip. Keep Le ≤ 1.
//
// The pathtracer renders to an offscreen RenderTarget; pixels are read back
// (readRGBPixels only works on a RenderTarget) and displayed via a preview
// quad.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstring>

using namespace threepp;

namespace {

    constexpr float ROOM = 4.f;// box edge length

    void buildFurnaceRoom(Scene& scene) {
        auto mat = MeshStandardMaterial::create({{"color", Color(1.f, 1.f, 1.f)},
                                                 {"roughness", 1.f},
                                                 {"metalness", 0.f}});
        mat->side = Side::Back;// we're inside the box — render inner faces
        auto box = Mesh::create(BoxGeometry::create(ROOM, ROOM, ROOM), mat);
        box->position.set(0.f, ROOM / 2.f, 0.f);
        scene.add(box);
    }

    std::shared_ptr<Mesh> makeEmitter(float intensity) {
        auto mat = MeshStandardMaterial::create({{"color", Color(0.f, 0.f, 0.f)},// zero albedo
                                                 {"emissive", Color(1.f, 1.f, 1.f)},
                                                 {"emissiveIntensity", intensity},
                                                 {"roughness", 1.f},
                                                 {"metalness", 0.f}});
        auto m = Mesh::create(SphereGeometry::create(0.35f, 32, 32), mat);
        m->position.set(0.f, ROOM * 0.6f, 0.f);
        return m;
    }

    struct Stats {
        float meanR = 0.f, meanG = 0.f, meanB = 0.f;
        float mean = 0.f;  // luminance
        float stddev = 0.f;// luminance stddev
        int samples = 0;   // foreground pixels counted
    };

    // Compute stats over ALL pixels. A bgThreshold filter here is a trap: dim
    // pixels that gradually cross the threshold as accumulation progresses get
    // folded into the average mid-run, dragging the reported mean downward
    // even though the image itself isn't changing. For an enclosed cavity the
    // whole frame is foreground anyway.
    Stats computeStats(const std::vector<unsigned char>& px) {
        Stats s{};
        if (px.empty()) return s;
        const size_t n = px.size() / 3;
        double sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0;
        size_t kept = 0;
        for (size_t i = 0; i < n; ++i) {
            const double r = px[3 * i + 0] / 255.0;
            const double g = px[3 * i + 1] / 255.0;
            const double b = px[3 * i + 2] / 255.0;
            const double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sr += r;
            sg += g;
            sb += b;
            sl += l;
            sl2 += l * l;
            ++kept;
        }
        if (kept == 0) return s;
        s.meanR = static_cast<float>(sr / kept);
        s.meanG = static_cast<float>(sg / kept);
        s.meanB = static_cast<float>(sb / kept);
        s.mean = static_cast<float>(sl / kept);
        const double var = sl2 / kept - (sl / kept) * (sl / kept);
        s.stddev = static_cast<float>(std::sqrt(std::max(0.0, var)));
        s.samples = static_cast<int>(kept);
        return s;
    }

}// namespace

int main() {
    Canvas canvas("WgpuPathTracer — White Furnace Energy Test", {{"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::Linear;// raw linear readback
    renderer.toneMapping = ToneMapping::None;  // no tonemap squish
    renderer.toneMappingExposure = 1.0f;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setMaxBounces(64);          // albedo=1 needs many bounces to converge
    pathTracer.setDenoiserEnabled(false);  // raw signal — no spatial smoothing
    pathTracer.setReSTIREnabled(true);
    pathTracer.setFoveatedRendering(false);// uniform quality across screen
    pathTracer.setFireflyClamp(0.f);       // disable clamp — unbiased HDR
    pathTracer.setExposure(1.f);

    // ── Offscreen target + preview (readRGBPixels only works on a RenderTarget) ──
    auto [cw, ch] = canvas.size();
    auto rt = RenderTarget::create(static_cast<unsigned>(cw),
                                   static_cast<unsigned>(ch),
                                   RenderTarget::Options{});
    auto previewTex = DataTexture::create(3, cw, ch);
    previewTex->format = Format::RGB;
    previewTex->magFilter = Filter::Linear;
    previewTex->minFilter = Filter::Linear;

    auto previewScene = Scene::create();
    auto previewCam = OrthographicCamera::create(-1, 1, 1, -1, 0, 1);
    auto previewMat = MeshBasicMaterial::create();
    previewMat->map = previewTex;
    previewScene->add(Mesh::create(PlaneGeometry::create(2, 2), previewMat));

    // ── Scene ──
    Scene scene;
    scene.background = Color(0.f, 0.f, 0.f);// no env, no sky
    buildFurnaceRoom(scene);
    float emitterIntensity = 1.f;// must be ≤ 1.0 to avoid LDR clip in readback
    auto emitter = makeEmitter(emitterIntensity);
    scene.add(emitter);

    // ── Camera ── inside the box, looking at a wall.
    PerspectiveCamera camera(60.f, canvas.aspect(), 0.01f, 100.f);
    camera.position.set(0.f, ROOM * 0.5f, ROOM * 0.25f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, ROOM * 0.5f, -1.f);
    controls.update();

    // ── State ──
    bool restirOn = true;
    int maxBounces = pathTracer.maxBounces();
    int measureEveryN = 128;// ρ=1 box converges slowly; coarser measurement interval
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

    // ── ImGui ──
    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10});
        ImGui::SetNextWindowSize({380, 0});
        ImGui::Begin("White Furnace Test");

        ImGui::Text("Frames accumulated: %d", pathTracer.frameCount());
        if (!fresh) ImGui::Text("(measured @ frame %d, fg pixels=%d)", statsFrame, stats.samples);
        ImGui::Separator();

        ImGui::Text("Target: every pixel = %.3f", emitterIntensity);
        if (emitterIntensity > 1.0f) {
            ImGui::TextColored(ImVec4(1.f, 0.5f, 0.3f, 1.f),
                               "WARN: LDR readback clips at 1.0 — set Le ≤ 1");
        }
        if (fresh) {
            ImGui::TextDisabled("(accumulating... stats update every %d frames)", measureEveryN);
        } else {
            const float tgt = std::min(emitterIntensity, 1.0f);
            const float devPct = 100.f * (stats.mean - tgt) / std::max(tgt, 1e-6f);
            ImGui::Text("Mean RGB: (%.4f, %.4f, %.4f)", stats.meanR, stats.meanG, stats.meanB);
            ImGui::Text("Luminance mean: %.4f", stats.mean);
            ImVec4 col = (std::abs(devPct) < 1.f)   ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
                         : (std::abs(devPct) < 3.f) ? ImVec4(1.f, 1.f, 0.4f, 1.f)
                                                    : ImVec4(1.f, 0.4f, 0.4f, 1.f);
            ImGui::TextColored(col, "Deviation: %+.2f%%", devPct);
            ImGui::Text("Stddev: %.4f", stats.stddev);
        }
        ImGui::Separator();

        bool dirty = false;
        if (ImGui::Checkbox("ReSTIR DI", &restirOn)) {
            pathTracer.setReSTIREnabled(restirOn);
            dirty = true;
        }
        if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 128)) {
            pathTracer.setMaxBounces(maxBounces);
            dirty = true;
        }
        if (ImGui::SliderFloat("Emitter Le", &emitterIntensity, 0.1f, 2.f)) {
            auto* m = dynamic_cast<MeshStandardMaterial*>(emitter->material().get());
            if (m) {
                m->emissiveIntensity = emitterIntensity;
                m->needsUpdate();
            }
            dirty = true;
        }
        ImGui::SliderInt("Measure every N frames", &measureEveryN, 1, 1024);
        if (ImGui::Button("Reset accumulation")) dirty = true;
        if (dirty) resetAll();

        ImGui::Separator();
        ImGui::TextDisabled("Method");
        ImGui::TextDisabled("  Tonemap=None, Encoding=Linear");
        ImGui::TextDisabled("  Walls: albedo=1 Lambertian");
        ImGui::TextDisabled("  Emitter: albedo=0, Le=%.2f", emitterIntensity);
        ImGui::TextDisabled("Interpretation");
        ImGui::TextDisabled("  mean < Le → per-bounce energy leak");
        ImGui::TextDisabled("  ReSTIR vs OFF disagree → MIS/W bias");
        ImGui::TextDisabled("  Slow to converge; run ≥1000 frames.");
        ImGui::TextDisabled("  See wgpu_furnace_env_test for one-bounce sanity.");

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
        // Recreate rt + preview texture at the new size.
        rt = RenderTarget::create(static_cast<unsigned>(ns.width()),
                                  static_cast<unsigned>(ns.height()),
                                  RenderTarget::Options{});
        previewTex = DataTexture::create(3, ns.width(), ns.height());
        previewTex->format = Format::RGB;
        previewTex->magFilter = Filter::Linear;
        previewTex->minFilter = Filter::Linear;
        previewMat->map = previewTex;
        previewMat->needsUpdate();
        resetStats();// setSize already resets pathTracer accumulation
    });

    // Camera-move detection to invalidate stats
    Vector3 lastCamPos = camera.position;
    Vector3 lastCamTgt = controls.target;

    canvas.animate([&] {
        controls.update();

        if (!camera.position.equals(lastCamPos) || !controls.target.equals(lastCamTgt)) {
            lastCamPos = camera.position;
            lastCamTgt = controls.target;
            resetStats();
        }

        // 1) Render pathtracer into the offscreen target.
        renderer.setRenderTarget(rt.get());
        pathTracer.render(scene, camera);

        // 2) Read back the target. readRGBPixels returns {} if no target is set.
        pixels = renderer.readRGBPixels();
        const auto sz = canvas.size();
        const int w = sz.width();
        const int h = sz.height();

        if (!pixels.empty()) {
            // Flip vertically — Wgpu readback is top-down, GL/Texture conventions expect bottom-up.
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
                stats = computeStats(pixels);
                statsFrame = fc;
                fresh = false;
            }
        }

        // 3) Blit target → canvas via a fullscreen preview quad.
        renderer.setRenderTarget(nullptr);
        renderer.render(*previewScene, *previewCam);

        ui.render();
    });
}
