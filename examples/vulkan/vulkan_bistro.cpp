
#include "threepp/extras/imgui/ImguiContext.hpp"
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
                  { {"vsync", false}});

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
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(0.f, 1.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
    ModelLoader loader;
    auto interior = loader.load(modelFolder / "BistroInterior_Wine.fbx");
    auto exterior = loader.load(modelFolder / "BistroExterior.fbx");

    scene.add(interior);
    scene.add(exterior);

    scene.traverseType<Light>([&](Light& l) {
        l.visible = false;
    });


    // Auto-fit camera to model bounding box
    auto fitCamera = [&](Object3D& obj) {
        Box3 bbox;
        bbox.setFromObject(obj);
        auto center = bbox.getCenter();
        auto size = bbox.getSize();
        float maxDim = std::max({size.x, size.y, size.z});
        float dist = maxDim * 1.5f / std::tan(camera.fov * 0.5f * math::PI / 180.f);

        controls.target.copy(center);
        camera.position.set(center.x, center.y, center.z + dist);
        camera.nearPlane = dist * 0.01f;
        camera.farPlane = dist * 100.f;
        camera.updateProjectionMatrix();
        controls.update();
    };

    fitCamera(scene);

    // ---- UI ----
    bool raster = false;
    bool denoiserOn = renderer.denoise();
    bool restdirOn = renderer.restirDIEnabled();
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            raster = !raster;
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("GLTF Samples");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Mode: %s (T to cycle)", raster ? "Raster" : "Path tracer");
        // if (!raster) ImGui::Text("Frames: %d", renderer.frameCount());

        ImGui::Separator();
        ImGui::SliderFloat("Exposure", &renderer.toneMappingExposure, 0.1f, 2.0f);

        if (!raster && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                renderer.setDenoise(denoiserOn);
            if (ImGui::Checkbox("REsTDIR DI", &restdirOn))
                renderer.setRestirDIEnabled(restdirOn);

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

        ui.render();
    });
}
