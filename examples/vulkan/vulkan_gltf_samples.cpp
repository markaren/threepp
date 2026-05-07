
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>

using namespace threepp;
namespace fs = std::filesystem;

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

            auto nested = dir.path() / "glTF" / (name + ".gltf");
            if (fs::exists(nested)) {
                entries.push_back({name, nested});
                continue;
            }
            auto flat = dir.path() / (name + ".gltf");
            if (fs::exists(flat)) {
                entries.push_back({name, flat});
                continue;
            }
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
        // https://github.com/KhronosGroup/glTF-Sample-Assets
        std::cout << "Usage: " << argv[0] << " <path_to_gltf_Models_folder>" << std::endl;
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
    std::cout << "Found " << models.size() << " models. Use Left/Right (or P/N) to browse." << std::endl;

    Canvas canvas("Vulkan PT - GLTF Samples", {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    renderer.setHybridEnabled(true);
    renderer.setHybridDebugView(0);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) +
                         "/textures/env/citrus_orchard_road_puresky_2k.hdr");

    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto sun = DirectionalLight::create(Color(0xffffff), 3.0f);
    sun->position.set(0.4f, 1.0f, 0.3f);
    scene.add(sun);

    PerspectiveCamera camera(50.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(0.f, 1.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    struct LoadedGltf {
        std::shared_ptr<Group> scene;
        std::vector<std::shared_ptr<AnimationClip>> animations;
    };

    GLTFLoader loader;
    int currentModel = -1;
    std::shared_ptr<Group> loadedModel;
    std::unique_ptr<AnimationMixer> mixer;
    std::vector<std::shared_ptr<AnimationClip>> clips;
    std::future<LoadedGltf> modelFuture;
    bool loadPending = false;

    auto loadModel = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(models.size())) return;
        if (loadPending) return;

        currentModel = idx;
        loadPending = true;
        auto path = models[idx].path;
        std::cout << "Loading: " << models[idx].name << " (" << path << ")" << std::endl;

        modelFuture = std::async(std::launch::async, [&loader, path, name = models[idx].name]() -> LoadedGltf {
            try {
                auto result = loader.load(path);
                if (!result || !result->scene) {
                    std::cerr << "Load failed '" << name << "'" << std::endl;
                    return {};
                }
                auto& root = result->scene;
                bool hasMesh = false;
                root->traverseType<Mesh>([&](Mesh&) { hasMesh = true; });
                root->traverseType<Light>([&](Light& l) {
                    l.visible = false;
                    l.intensity = std::max(l.intensity, 1.0f);
                });
                if (!hasMesh) {
                    std::cerr << "Skipping '" << name << "': no mesh geometry" << std::endl;
                    return {};
                }
                return {root, std::move(result->animations)};
            } catch (const std::exception& e) {
                std::cerr << "Load failed '" << name << "': " << e.what() << std::endl;
                return {};
            }
        });
    };

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

    loadModel(0);

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::RIGHT || ev.key == Key::N) {
            loadModel((currentModel + 1) % static_cast<int>(models.size()));
        }
        if (ev.key == Key::LEFT || ev.key == Key::P) {
            loadModel((currentModel - 1 + static_cast<int>(models.size())) % static_cast<int>(models.size()));
        }
    });
    canvas.addKeyListener(keyAdapter);

    float exposure = renderer.toneMappingExposure;
    int toneMode = static_cast<int>(renderer.toneMapping);
    bool dirLight = sun->visible;
    int spp = renderer.samplesPerPixel();
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("Vulkan PT - GLTF Samples");
        ImGui::Text("FPS: %.1f", fps);

        ImGui::Separator();
        ImGui::Text("Model: %s", currentModel >= 0 ? models[currentModel].name.c_str() : "none");
        if (loadPending) ImGui::Text("Loading...");
        ImGui::Text("Left/Right arrows to browse");

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
            renderer.toneMappingExposure = exposure;

        const char* toneItems[] = {"None", "Linear", "Reinhard", "Cineon", "ACESFilmic"};
        if (ImGui::Combo("Tone mapping", &toneMode, toneItems, IM_ARRAYSIZE(toneItems)))
            renderer.toneMapping = static_cast<ToneMapping>(toneMode);

        if (ImGui::Checkbox("DirLight", &dirLight))
            sun->visible = dirLight;

        bool denoise = renderer.denoise();
        if (ImGui::Checkbox("Denoise", &denoise))
            renderer.setDenoise(denoise);

        bool hybrid = renderer.hybridEnabled();
        if (ImGui::Checkbox("Hybrid (raster + PT)", &hybrid))
            renderer.setHybridEnabled(hybrid);

        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 16))
            renderer.setSamplesPerPixel(spp);

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

        if (loadPending && modelFuture.valid() &&
            modelFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto loaded = modelFuture.get();
            loadPending = false;
            if (loadedModel) {
                scene.remove(*loadedModel);
                loadedModel.reset();
            }
            mixer.reset();
            clips.clear();
            loadedModel = loaded.scene;
            if (loadedModel) {
                scene.add(loadedModel);
                fitCamera(*loadedModel);
                clips = std::move(loaded.animations);
                if (!clips.empty()) {
                    mixer = std::make_unique<AnimationMixer>(*loadedModel);
                    mixer->clipAction(clips.front())->play();
                    std::cout << "Playing animation: " << clips.front()->name()
                              << " (" << clips.size() << " clip(s))" << std::endl;
                }
                std::cout << "Loaded: " << models[currentModel].name << std::endl;
            }
        }

        if (mixer) mixer->update(dt);

        controls.update();
        renderer.render(scene, camera);

        ui.render();
    });

    return 0;
}
