
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/FBXLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <filesystem>

using namespace threepp;
namespace fs = std::filesystem;


int main(int argc, char** argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_bistro_folder>" << std::endl;
        return 1;
    }

    fs::path modelFolder = argv[1];
    if (!fs::exists(modelFolder) || !fs::is_directory(modelFolder)) {
        std::cerr << "Invalid folder path: " << fs::absolute(modelFolder) << std::endl;
        return 1;
    }

    // ---- Window & Renderer ----
    Canvas canvas("Bistro scene",
                  {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::Neutral;
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(8.f);
    renderer.setRenderScale(0.8f);

    RGBELoader imgLoader;
    auto env = imgLoader.load(modelFolder / "san_giuseppe_bridge_4k.hdr");

    // ---- Scene ----
    Scene scene;
    scene.background = env;
    scene.environment = env;

    // ---- Camera ----
    PerspectiveCamera camera(60.f, canvas.aspect(), 0.1f, 1000.f);// near 0.01→0.1: far/near ratio was 100k → z-fighting
    camera.position.set(-10.f, 3.f, -5.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
    FBXLoader loader;
    // Bistro emissives (lamps, signs, string lights) are authored at low factors;
    // the official Falcor scene multiplies every emissive factor by 1000 to get a
    // well-exposed image, so do the same here for interior lighting to read.
    loader.emissiveScale = 1000.0f;
    auto interior = loadAsync(loader, modelFolder / "BistroInterior_Wine.fbx");
    auto exterior = loadAsync(loader, modelFolder / "BistroExterior.fbx");

    scene.add(interior);
    scene.add(exterior);

    auto toggleBistroLights = [&] {
        scene.traverseType<Light>([&](Light& l) {
            l.visible = !l.visible;
        });
    };

    // ---- UI ----
    bool denoiserOn = renderer.denoise();
    bool restdirOn = renderer.restirDIEnabled();
    bool restdirVisReuse = renderer.restirDIVisibilityReuse();
    bool restdirGiOn = renderer.restirGIEnabled();
    // Tone-map dropdown state, initialized from the renderer's current setting so
    // the label matches what's actually applied at startup. The selection maps
    // straight to ToneMapping (index 2 -> Reinhard, etc.).
    const char* tmNames[] = {"ACES Filmic", "Neutral (PBR)", "Reinhard", "Cineon", "Linear"};
    const ToneMapping tmVals[] = {ToneMapping::ACESFilmic, ToneMapping::Neutral,
                                  ToneMapping::Reinhard, ToneMapping::Cineon, ToneMapping::Linear};
    int tmIdx = 0;
    for (int i = 0; i < IM_ARRAYSIZE(tmVals); ++i)
        if (tmVals[i] == renderer.toneMapping) { tmIdx = i; break; }
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;


    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("Bistro scene");
        ImGui::Text("FPS: %.1f", fps);

        ImGui::Separator();
        ImGui::SliderFloat("Exposure", &renderer.toneMappingExposure, 0.1f, 2.0f);

        // Tone-map selector. Reinhard is per-channel so it keeps saturated
        // emitters (blue/green bulbs) coloured; ACES and Neutral desaturate very
        // bright values toward white. Switchable live.
        if (ImGui::Combo("Tone map", &tmIdx, tmNames, IM_ARRAYSIZE(tmNames))) {
            renderer.toneMapping = tmVals[tmIdx];
        }

        if (ImGui::Checkbox("Denoiser", &denoiserOn)) {
            renderer.setDenoise(denoiserOn);
        }
        if (ImGui::Checkbox("REsTDIR DI", &restdirOn)) {
            renderer.setRestirDIEnabled(restdirOn);
        }
        if (ImGui::Checkbox("  DI visibility reuse", &restdirVisReuse)) {
            renderer.setRestirDIVisibilityReuse(restdirVisReuse);
        }
        if (ImGui::Checkbox("REsTDIR GI", &restdirGiOn)) {
            renderer.setRestirGIEnabled(restdirGiOn);
        }

        if (ImGui::Button("Toggle bistro lights")) {
            toggleBistroLights();
        }

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
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();

        renderer.render(scene, camera);
        // auto t = renderer.lastFrameTimings();
        // static int frames = 0;
        // static double accum = 0.0;
        // accum += t.cpuFrameMs;
        // if (++frames >= 60) {
        //     std::cout << std::fixed << std::setprecision(2)
        //               << "frame " << t.cpuFrameMs << " ms"
        //               << " | PT " << t.pathTraceMs
        //               << " | denoise " << t.denoiseMs
        //               << " | TAA " << t.taaMs
        //               << " | gbuf " << t.rasterGbufMs
        //               << " | overlay " << t.overlayMs
        //               << " | photon " << t.photonEmitMs
        //               << " | cpu(scene " << t.cpuEnsureSceneMs
        //               << ", record " << t.cpuRecordMs << ")"
        //               << '\n';
        //     frames = 0;
        //     accum = 0.0;
        // }

        ui.render();
    });
}
