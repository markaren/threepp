
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

// Scan for glTF models: supports both flat (folder/Model.gltf) and
// nested (folder/Model/glTF/Model.gltf) layouts.
struct ModelEntry {
    std::string name;
    fs::path path;
};

static std::vector<ModelEntry> scanModels(const fs::path& root) {
    std::vector<ModelEntry> entries;
    for (auto& dir : fs::directory_iterator(root)) {
        try {
            if (!dir.is_directory()) continue;
            auto name = dir.path().filename().string();

            // Try nested: <name>/glTF/<name>.gltf
            auto nested = dir.path() / "glTF" / (name + ".gltf");
            if (fs::exists(nested)) {
                entries.push_back({name, nested});
                continue;
            }
            // Try flat: <name>/<name>.gltf
            auto flat = dir.path() / (name + ".gltf");
            if (fs::exists(flat)) {
                entries.push_back({name, flat});
                continue;
            }
            // Try any .gltf in the folder
            for (auto& f : fs::directory_iterator(dir.path())) {
                if (f.path().extension() == ".gltf") {
                    entries.push_back({name, f.path()});
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Skipping folder: " << e.what() << std::endl;
        }
    }
    std::ranges::sort(entries,
                      [](const auto& a, const auto& b) { return a.name < b.name; });
    return entries;
}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_gltf_sample_folder>" << std::endl;
        return 1;
    }

    fs::path modelFolder = argv[1];
    if (!fs::exists(modelFolder) || !fs::is_directory(modelFolder)) {
        std::cerr << "Invalid folder path: " << fs::absolute(modelFolder) << std::endl;
        return 1;
    }

    auto models = scanModels(modelFolder);
    if (models.empty()) {
        std::cerr << "No glTF models found in: " << fs::absolute(modelFolder) << std::endl;
        return 1;
    }
    std::cout << "Found " << models.size() << " models" << std::endl;

    // ---- Window & Renderer ----
    Canvas canvas("GLTF Samples",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(0.5f);
    pathTracer.setExposure(1.0f);
    pathTracer.setSamplesPerPixel(2);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setReSTIREnabled(false);
    pathTracer.setMaxBounces(4);
    pathTracer.setMode(WgpuPathTracer::Mode::Raytracer);
    pathTracer.setHybridMode(true);

    ImageLoader imgLoader;
    auto img = imgLoader.loadHDR(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr", 4, false);
    auto env = Texture::create(*img);

    // ---- Scene ----
    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto light = DirectionalLight::create(0xffffff, 1.0f);
    light->position.set(1, 1, 1);
    light->visible = false;
    scene.add(light);
    scene.add(AmbientLight::create(0xffffff, 0.2f));

    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(0.f, 1.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
    ModelLoader loader;
    int currentModel = -1;
    std::shared_ptr<Group> loadedModel;
    std::future<std::shared_ptr<Group>> modelFuture;
    bool loadPending = false;

    auto loadModel = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(models.size())) return;
        if (loadPending) return;

        currentModel = idx;
        loadPending = true;
        auto path = models[idx].path;
        std::cout << "Loading: " << models[idx].name << " (" << path << ")" << std::endl;

        modelFuture = std::async(std::launch::async, [&loader, path, name = models[idx].name]() -> std::shared_ptr<Group> {
            try {
                auto result = loader.load(path.string());
                // Verify it has renderable geometry
                bool hasMesh = false;
                if (result) {
                    result->traverseType<Mesh>([&](Mesh&) {
                        hasMesh = true;
                    });
                    result->traverseType<Light>([&](Light& l) {
                        l.visible = true;
                    });
                }
                if (!hasMesh) {
                    std::cerr << "Skipping '" << name << "': no mesh geometry" << std::endl;
                    return nullptr;
                }
                return result;
            } catch (const std::exception& e) {
                std::cerr << "Load failed '" << name << "': " << e.what() << std::endl;
                return nullptr;
            }
        });
    };

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

    // Load first model
    loadModel(0);

    // ---- UI ----
    int renderMode = 2;
    const char* modeNames[] = {"Raytracer", "PathTracer", "Raster"};
    bool dirLight = false;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool restdirOn = pathTracer.restirEnabled();
    bool hybridMode = pathTracer.hybridMode();
    float exposure = pathTracer.exposure();
    float envIntensity = pathTracer.envIntensity();
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            renderMode = (renderMode + 1) % 3;
            if (renderMode < 2) {
                pathTracer.setMode(renderMode == 1 ? WgpuPathTracer::Mode::PathTracer
                                                   : WgpuPathTracer::Mode::Raytracer);
            }
        }
        if (ev.key == Key::RIGHT || ev.key == Key::N) {
            loadModel((currentModel + 1) % static_cast<int>(models.size()));
        }
        if (ev.key == Key::LEFT || ev.key == Key::P) {
            loadModel((currentModel - 1 + static_cast<int>(models.size())) % static_cast<int>(models.size()));
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("GLTF Samples");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Mode: %s (T to cycle)", modeNames[renderMode]);
        if (renderMode < 2) ImGui::Text("Frames: %d", pathTracer.frameCount());

        ImGui::Separator();
        ImGui::Text("Model: %s", currentModel >= 0 ? models[currentModel].name.c_str() : "none");
        if (loadPending) ImGui::Text("Loading...");
        ImGui::Text("Left/Right arrows to browse");

        // Model list
        if (ImGui::CollapsingHeader("Models")) {
            for (int i = 0; i < static_cast<int>(models.size()); i++) {
                const bool selected = (i == currentModel);
                if (ImGui::Selectable(models[i].name.c_str(), selected)) {
                    loadModel(i);
                }
            }
        }

        ImGui::Separator();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
            pathTracer.setExposure(exposure);
        if (ImGui::SliderFloat("EnvIntensity", &envIntensity, 0.0f, 1.0f))
            pathTracer.setEnvIntensity(envIntensity);

        if (renderMode == 1) {
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::Checkbox("REsTDIR", &restdirOn))
                pathTracer.setReSTIREnabled(restdirOn);
            if (ImGui::Checkbox("Hybrid mode", &hybridMode))
                pathTracer.setHybridMode(hybridMode);
        }
        if (renderMode != 2) {
            if (ImGui::Checkbox("Show DirLight", &dirLight)) {
                light->visible = dirLight;
                pathTracer.markDirty();
            }
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

        light->visible = dirLight || renderMode == 2;

        // Check async model load
        if (loadPending && modelFuture.valid() &&
            modelFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto newModel = modelFuture.get();
            loadPending = false;
            if (loadedModel) {
                scene.remove(*loadedModel);
                loadedModel.reset();
            }
            loadedModel = newModel;
            if (loadedModel) {
                scene.add(loadedModel);
                fitCamera(*loadedModel);
                pathTracer.markDirty();
                std::cout << "Loaded: " << models[currentModel].name << std::endl;
            }
        }

        controls.update();

        if (renderMode == 2) {
            renderer.render(scene, camera);
        } else {
            pathTracer.render(scene, camera);
        }

        ui.render();
    });
}
