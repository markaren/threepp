
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <filesystem>
#include <future>

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
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(1.0f);
    pathTracer.setExposure(1.0f);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setReSTIREnabled(false);
    pathTracer.setMaxBounces(4);

    ImageLoader imgLoader;
    auto img = imgLoader.loadHDR(modelFolder / "san_giuseppe_bridge_4k.hdr", 4, false);
    auto env = Texture::create(*img);

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

    exterior->traverseType<Light>([&](Light& l) {
        l.visible = false;
    });

    interior->traverseType<Light>([&](Light& l) {
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

    fitCamera(*exterior);

    // ---- UI ----
    bool raster = false;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool restdirOn = pathTracer.restirEnabled();
    bool restdirGIOn = pathTracer.restirGiEnabled();
    float exposure = pathTracer.exposure();
    float envIntensity = pathTracer.envIntensity();
    float fps = 0.f, fpsAccum = 0.f;
    float pixelScale = pathTracer.pixelScale();
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
        if (!raster) ImGui::Text("Frames: %d", pathTracer.frameCount());

        ImGui::Separator();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
            pathTracer.setExposure(exposure);
        if (ImGui::SliderFloat("EnvIntensity", &envIntensity, 0.0f, 1.0f))
            pathTracer.setEnvIntensity(envIntensity);
        if (ImGui::SliderFloat("Pixel Scale", &pixelScale, 0.25f, 1.2f, "%.2f"))
            pathTracer.setPixelScale(pixelScale);

        if (!raster && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::Checkbox("REsTDIR DI", &restdirOn))
                pathTracer.setReSTIREnabled(restdirOn);
            if (ImGui::Checkbox("REsTDIR GI", &restdirGIOn))
                pathTracer.setReSTIRGIEnabled(restdirGIOn);

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
        pathTracer.setSize({ns.width(), ns.height()});
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

        if (raster) {
            renderer.render(scene, camera);
        } else {
            pathTracer.render(scene, camera);
        }

        ui.render();
    });
}
