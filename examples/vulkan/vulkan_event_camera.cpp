// Vulkan PT — event camera (DVS) sensor demo.
//
// Three pendulums swinging at different frequencies in front of a striped
// wall. The event camera helper (EventCameraSensor) reads the renderer's
// current frame each tick and emits per-pixel events whenever
// |Δlog luminance| crosses the configured contrast threshold — exactly
// the model real DVS sensors implement at the analog photoreceptor.
//
// The right-hand panel shows the live event accumulator: bright pixels
// where a recent positive (brighter) event fired, dark where negative,
// mid-grey where nothing changed. The decay rate controls how long
// trails persist; high values give the smooth swooping motion blur a
// real DVS captures; low values strip everything to instantaneous
// silhouettes.
//
// The pendulums' phases are deliberately offset so the demo never has
// a frame of zero events — handy for spotting whether the detection
// pipeline is alive.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

using namespace threepp;

namespace {

    struct Pendulum {
        std::shared_ptr<Object3D> pivot;    // attached to the scene; rotates
        std::shared_ptr<Mesh>     rod;      // child of pivot, points -Y
        std::shared_ptr<Mesh>     bob;      // sphere at the end of the rod
        float                     amplitude;// radians
        float                     omega;    // rad/s
        float                     phase;    // radians
        float                     length;
    };

    Pendulum makePendulum(Scene& scene, const Vector3& pivotPos,
                          float length, float amplitude, float omega, float phase,
                          const Color& bobColor) {
        Pendulum p;
        p.amplitude = amplitude;
        p.omega     = omega;
        p.phase     = phase;
        p.length    = length;

        p.pivot = std::make_shared<Object3D>();
        p.pivot->position.copy(pivotPos);
        scene.add(p.pivot);

        auto rodMat = MeshStandardMaterial::create({
                {"color", Color(0x444444)},
                {"roughness", 0.6f},
                {"metalness", 0.3f},
        });
        p.rod = Mesh::create(CylinderGeometry::create(0.025f, 0.025f, length, 16, 1), rodMat);
        // Cylinder is local +Y axis-aligned; centre it so its top sits at the
        // pivot and the bob attaches at the bottom.
        p.rod->position.set(0.f, -length * 0.5f, 0.f);
        p.pivot->add(p.rod);

        auto bobMat = MeshStandardMaterial::create({
                {"color", bobColor},
                {"roughness", 0.35f},
                {"metalness", 0.1f},
        });
        p.bob = Mesh::create(SphereGeometry::create(0.15f, 32, 24), bobMat);
        p.bob->position.set(0.f, -length, 0.f);
        p.pivot->add(p.bob);
        return p;
    }

    // Striped wall behind the pendulums — gives the event camera per-pixel
    // contrast to trigger off when the pendulum silhouette sweeps across.
    void addBackdrop(Scene& scene) {
        constexpr int stripeCount = 9;
        constexpr float wallW = 6.f;
        constexpr float wallH = 4.f;
        constexpr float stripeW = wallW / stripeCount;

        auto darkMat   = MeshStandardMaterial::create({{"color", Color(0x222222)}, {"roughness", 0.95f}});
        auto lightMat  = MeshStandardMaterial::create({{"color", Color(0xdddddd)}, {"roughness", 0.95f}});

        for (int i = 0; i < stripeCount; ++i) {
            const float x = -wallW * 0.5f + (static_cast<float>(i) + 0.5f) * stripeW;
            auto stripe = Mesh::create(BoxGeometry::create(stripeW, wallH, 0.05f),
                                       (i % 2 == 0) ? lightMat : darkMat);
            stripe->position.set(x, 1.5f, -2.5f);
            scene.add(stripe);
        }

        // Matte floor for context. Floor never moves, so it won't generate
        // events on its own — only when pendulum shadows sweep across it.
        auto floor = Mesh::create(BoxGeometry::create(8.f, 0.1f, 6.f),
                                  MeshStandardMaterial::create({{"color", Color(0x888888)}, {"roughness", 0.95f}}));
        floor->position.set(0.f, -0.05f, 0.f);
        scene.add(floor);
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT - event camera (DVS)",
                  {{"vsync", false}, {"size", WindowSize{1280, 720}}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.background = Color(0.05f, 0.05f, 0.06f);
    addBackdrop(scene);

    // Three pendulums with offset phases / frequencies so the event
    // accumulator is always busy somewhere.
    std::array<Pendulum, 3> pendulums{
            makePendulum(scene, Vector3(-1.6f, 3.0f, -1.2f), 2.2f, 0.7f, 2.2f, 0.0f, Color(0xff5060)),
            makePendulum(scene, Vector3( 0.0f, 3.0f, -1.2f), 2.0f, 0.5f, 2.8f, 1.2f, Color(0x60ff70)),
            makePendulum(scene, Vector3( 1.6f, 3.0f, -1.2f), 2.4f, 0.6f, 2.5f, 2.4f, Color(0x6080ff)),
    };

    scene.add(AmbientLight::create(Color(0xffffff), 0.3f));
    auto sun = DirectionalLight::create(Color(0xfff0d6), 2.0f);
    sun->position.set(2.5f, 6.f, 3.f);
    scene.add(sun);

    auto camera = PerspectiveCamera::create(50.f, canvas.aspect(), 0.1f, 50.f);
    camera->position.set(0.f, 2.0f, 4.5f);
    camera->lookAt(Vector3(0.f, 1.5f, -1.5f));

    // ── Event camera sensor ────────────────────────────────────────────
    // Resolution matches the swapchain — readRGBPixels returns the full
    // swapchain image. The accumulator visualisation is shown at quarter
    // size in the corner via a screen-space Sprite.
    const auto canvasSize = canvas.size();
    // GPU-side event camera. Implicit: enables sceneCapture (its input)
    // and lazy-creates the compute pipeline + accumulator/history images
    // at the current swapchain size. No CPU per-pixel loop, no
    // vkDeviceWaitIdle — reads from a 3-slot ring with 2 frames of
    // display latency.
    {
        VulkanRenderer::EventCameraParams p{};
        p.threshold = 0.20f;
        p.decay     = 0.88f;
        renderer.setEventCameraParams(p);
        // Pin a DVS-realistic sensor resolution. 640×480 is roughly
        // Prophesee Gen3/4 territory; physically meaningful AND about
        // 3× fewer pixels than the swapchain → smaller detector + readback
        // work. Pass (0,0) instead to match the swapchain.
        renderer.setEventCameraResolution(640, 480);
        renderer.setEventCameraEnabled(true);
    }

    // The accumulator is a sensor-resolution RGBA8 image readable via
    // renderer.readEventCameraVisualisation(); we drop those bytes into
    // a DataTexture each frame so the existing sprite path can display
    // it (the sprite then upscales to the configured panel size).
    // Same pattern the LIDAR readout panel uses.
    const auto sensorRes = renderer.eventCameraResolution();
    auto eventVizTex = DataTexture::create(
            ImageData{std::vector<unsigned char>(
                    static_cast<size_t>(sensorRes.first) *
                            static_cast<size_t>(sensorRes.second) * 4,
                    128u)},
            sensorRes.first, sensorRes.second);
    eventVizTex->colorSpace = ColorSpace::sRGB;

    auto eventVizMat = SpriteMaterial::create();
    eventVizMat->map = eventVizTex;
    auto eventViz = Sprite::create(eventVizMat);
    constexpr float kPanelDispW = 480.f;
    constexpr float kPanelDispH = 270.f;
    eventViz->scale.set(kPanelDispW, kPanelDispH, 1.f);
    eventViz->screenSpace = true;
    eventViz->screenAnchor.set(1.f, 1.f);
    eventViz->center.set(1.f, 1.f);
    eventViz->position.set(-10.f, -10.f, 0.f);
    scene.add(eventViz);

    // Helper used by the UI toggle + resize hook to flip the sprite
    // between corner-panel and full-screen layouts. In events-only mode
    // the path tracer doesn't produce a visual at all (swapchain is
    // cleared to black), so the accumulator IS the only content the
    // user sees — it gets the whole window.
    auto applySpriteLayout = [&](bool fullScreen, WindowSize sz) {
        if (fullScreen) {
            eventViz->scale.set(static_cast<float>(sz.width()),
                                static_cast<float>(sz.height()),
                                1.f);
            eventViz->screenAnchor.set(0.f, 0.f);
            eventViz->center.set(0.f, 0.f);
            eventViz->position.set(0.f, 0.f, 0.f);
        } else {
            eventViz->scale.set(kPanelDispW, kPanelDispH, 1.f);
            eventViz->screenAnchor.set(1.f, 1.f);
            eventViz->center.set(1.f, 1.f);
            eventViz->position.set(-10.f, -10.f, 0.f);
        }
    };

    // ── UI ─────────────────────────────────────────────────────────────
    float lastCaptureMs     = 0.f;
    bool  animatePendulums  = true;
    bool  eventsOnly        = false;
    auto  evParams          = renderer.eventCameraParams();

    // Sparse event-stream buffer — reused frame-to-frame; the demo
    // pulls events into it after every render() and the UI lambda
    // reads `lastEventCount` for the count display.
    std::vector<VulkanRenderer::Event> events(256u * 1024u);
    size_t lastEventCount = 0;
    bool   lastOverflowed = false;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        // Re-enable: the renderer will idle and resize the detector's
        // images. With a user-pinned sensor resolution the detector
        // stays at sensor res (not the new swapchain); the host-side
        // DataTexture follows whatever the detector reports.
        renderer.setEventCameraEnabled(false);
        renderer.setEventCameraEnabled(true);

        const auto sr = renderer.eventCameraResolution();
        const size_t pixels = static_cast<size_t>(sr.first) *
                              static_cast<size_t>(sr.second);
        eventVizTex = DataTexture::create(
                ImageData{std::vector<unsigned char>(pixels * 4, 128u)},
                sr.first, sr.second);
        eventVizTex->colorSpace = ColorSpace::sRGB;
        eventVizMat->map = eventVizTex;

        applySpriteLayout(eventsOnly, size);
    });

    float smoothedFps = 0.f;
    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({380, 0});
        ImGui::Begin("Vulkan PT - event camera (DVS)");
        ImGui::TextWrapped(
                "Three pendulums swing in front of a striped wall. "
                "A GPU compute pass detects per-pixel log-intensity "
                "changes against a persistent reference; events fire "
                "where |Δlog I| exceeds the threshold.");
        ImGui::Separator();
        ImGui::Checkbox("Animate pendulums", &animatePendulums);

        // Events-only mode: when on, the renderer skips PT/denoise/TAA
        // entirely — only the raster gbuf prepass + event_shade +
        // event_detect run. The pendulums no longer appear in the
        // swapchain (just a black clear), but the event camera readout
        // covers the full window so the sensor is still observable.
        if (ImGui::Checkbox("Events-only render (~500 Hz path)", &eventsOnly)) {
            renderer.setEventsOnlyMode(eventsOnly);
            applySpriteLayout(eventsOnly, canvas.size());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextWrapped("Skips path tracing entirely. The accumulator "
                               "fills the window; the rendered scene is not "
                               "drawn. Use to maximise the event rate.");
            ImGui::EndTooltip();
        }
        bool changed = false;
        changed |= ImGui::SliderFloat("Contrast threshold (log)", &evParams.threshold,
                                      0.05f, 0.6f, "%.3f");
        changed |= ImGui::SliderFloat("Viz decay / frame", &evParams.decay,
                                      0.f, 0.995f, "%.3f");
        int maxEv = static_cast<int>(evParams.maxEventsPerPixel);
        if (ImGui::SliderInt("Max events / pixel / frame", &maxEv, 1, 20)) {
            evParams.maxEventsPerPixel = static_cast<uint32_t>(std::max(1, maxEv));
            changed = true;
        }
        if (changed) renderer.setEventCameraParams(evParams);
        ImGui::Separator();
        ImGui::Text("Frame rate:     %.0f Hz", static_cast<double>(smoothedFps));
        ImGui::Text("Readback cost:  %.2f ms", static_cast<double>(lastCaptureMs));
        ImGui::Text("Sensor res:     %u × %u",
                    static_cast<unsigned>(eventVizTex->image().width),
                    static_cast<unsigned>(eventVizTex->image().height));
        // Sparse event stream stats. Counts of bright vs dark events
        // computed by walking the host buffer — costs ~10 µs for typical
        // counts so safe to do inline; switch to a GPU-side polarity
        // histogram if events climb to many-millions.
        size_t nPos = 0, nNeg = 0;
        for (size_t i = 0; i < lastEventCount; ++i) {
            if (events[i].polarity > 0) ++nPos;
            else                        ++nNeg;
        }
        ImGui::Text("Events / frame: %zu  (+%zu / -%zu)",
                    lastEventCount, nPos, nNeg);
        ImGui::Text("Event rate:     %.1f kHz",
                    static_cast<double>(lastEventCount) *
                            static_cast<double>(smoothedFps) / 1000.0);
        if (lastOverflowed) {
            ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
                               "OVERFLOW — events dropped this frame");
        }
        // Per-stage timings — GPU values come from VkQueryPool, CPU
        // values from std::chrono around the host hot path.
        // cpuRecord covers record-start → vkEndCommandBuffer (so it
        // INCLUDES the post-render() endFrame work: ImGui overlay record +
        // present transition); cpuFrame is just render() wall time. They
        // overlap by recordCommandBuffer; cpuRecord - (cpuFrame - cpuEnsureScene)
        // ≈ endFrame's host-side cost.
        const auto ft = renderer.lastFrameTimings();
        ImGui::Separator();
        ImGui::Text("GPU raster gbuf:   %.2f ms", static_cast<double>(ft.rasterGbufMs));
        ImGui::Text("GPU path trace:    %.2f ms", static_cast<double>(ft.pathTraceMs));
        ImGui::Text("GPU denoise:       %.2f ms", static_cast<double>(ft.denoiseMs));
        ImGui::Text("GPU TAA:           %.2f ms", static_cast<double>(ft.taaMs));
        ImGui::Text("CPU ensureScene:   %.2f ms", static_cast<double>(ft.cpuEnsureSceneMs));
        ImGui::Text("CPU render():      %.2f ms", static_cast<double>(ft.cpuFrameMs));
        ImGui::Text("CPU record→submit: %.2f ms", static_cast<double>(ft.cpuRecordMs));
        const float endFrameMs = ft.cpuRecordMs - (ft.cpuFrameMs - ft.cpuEnsureSceneMs);
        ImGui::Text("  ↳ endFrame:      %.2f ms", static_cast<double>(endFrameMs));
        ImGui::TextWrapped("Detection runs on GPU; only the visualisation "
                           "memcpy is host-side. 2-frame display latency "
                           "due to the readback ring.");
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    Clock clock;
    auto frameTp = std::chrono::steady_clock::now();

    canvas.animate([&] {
        const float t = clock.getElapsedTime();
        if (animatePendulums) {
            for (auto& p : pendulums) {
                const float theta = p.amplitude * std::sin(p.omega * t + p.phase);
                p.pivot->rotation.set(0.f, 0.f, theta);
            }
        }

        // Stamp this render() with a microsecond clock. Carried via
        // EventCameraParams.frameTimeUs and tagged onto every event the
        // shader emits this frame.
        {
            auto p = renderer.eventCameraParams();
            p.frameTimeUs = static_cast<uint32_t>(t * 1.0e6f);
            renderer.setEventCameraParams(p);
        }

        renderer.render(scene, *camera);

        // Pull the accumulator the renderer's GPU compute pass painted
        // this frame. No vkDeviceWaitIdle — the ring's oldest slot is
        // guaranteed complete by the kFramesInFlight in-flight fences.
        // Zero-copy variant: writes straight into the DataTexture's
        // backing buffer, skipping the per-frame std::vector alloc and
        // the redundant mapped→vec→DataTexture double copy.
        const auto t0 = std::chrono::steady_clock::now();
        auto& dstBytes = eventVizTex->image().data<unsigned char>();
        const size_t got = renderer.readEventCameraVisualisationInto(
                dstBytes.data(), dstBytes.size());
        const auto t1 = std::chrono::steady_clock::now();
        lastCaptureMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (got > 0) {
            eventVizTex->needsUpdate();
        }

        // Pull the sparse event stream (oldest ring slot — already
        // complete; no GPU wait). This is what a real DVS sensor outputs
        // to its consumer; the visualisation above is for the demo's
        // sake. Counts and overflow shown in the HUD.
        lastEventCount = renderer.readEventStreamInto(
                events.data(), events.size(), &lastOverflowed);

        // Smoothed FPS for the HUD. Single-pole IIR (~1 s half-life at
        // ~60 Hz, faster at higher rates where the demo gets useful) so
        // the events-only number is readable without bouncing.
        const auto nowTp = std::chrono::steady_clock::now();
        const float frameMs = std::chrono::duration<float, std::milli>(
                                      nowTp - frameTp).count();
        frameTp = nowTp;
        if (frameMs > 0.f) {
            const float inst = 1000.f / frameMs;
            smoothedFps = (smoothedFps == 0.f) ? inst
                                                : 0.9f * smoothedFps + 0.1f * inst;
        }

        ui.render();
    });
}
