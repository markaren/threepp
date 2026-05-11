
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/FBXLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
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
    renderer.toneMapping = ToneMapping::ACESFilmic;

    RGBELoader imgLoader;
    auto env = imgLoader.load(modelFolder / "san_giuseppe_bridge_4k.hdr");

    // ---- Scene ----
    Scene scene;
    scene.background = env;
    scene.environment = env;

    // ---- Camera ----
    PerspectiveCamera camera(60.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(-10.f, 3.f, -5.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
    FBXLoader loader;
    auto interior = loader.load(modelFolder / "BistroInterior_Wine.fbx");
    auto exterior = loader.load(modelFolder / "BistroExterior.fbx");

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
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {

    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("Bistro scene");
        ImGui::Text("FPS: %.1f", fps);
        // if (!raster) ImGui::Text("Frames: %d", renderer.frameCount());

        ImGui::Separator();
        ImGui::SliderFloat("Exposure", &renderer.toneMappingExposure, 0.1f, 2.0f);


        if (ImGui::Checkbox("Denoiser", &denoiserOn)) {
            renderer.setDenoise(denoiserOn);
        }
        if (ImGui::Checkbox("REsTDIR DI", &restdirOn)) {
            renderer.setRestirDIEnabled(restdirOn);
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
