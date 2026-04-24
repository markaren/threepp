
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
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
    Canvas canvas("GLTF Samples", {{"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputEncoding = Encoding::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(1.0f);
    pathTracer.setExposure(1.0f);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setReSTIREnabled(false);
    pathTracer.setMaxBounces(4);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setTlasEnabled(false);

    RGBELoader imgLoader;
    auto env = imgLoader.load(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr", false);

    // ---- Scene ----
    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto light = DirectionalLight::create(0xffffff, 1.0f);
    light->position.set(1, 1, 1);
    light->visible = false;
    light->castShadow = true;
    scene.add(light);
    scene.add(AmbientLight::create(0xffffff, 0.2f));

    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(0.f, 1.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
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
                root->receiveShadow = true;
                root->castShadow = true;
                bool hasMesh = false;
                root->traverseType<Mesh>([&](Mesh&) {
                    hasMesh = true;
                });
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
    bool raster = false;
    bool dirLight = light->visible;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool restdirOn = pathTracer.restirEnabled();
    bool restdirGIOn = pathTracer.restirGiEnabled();
    float exposure = pathTracer.exposure();
    float envIntensity = pathTracer.envIntensity();
    float fps = 0.f, fpsAccum = 0.f;
    float pixelScale = pathTracer.pixelScale();
    int fpsFrames = 0;
    int aovMode = pathTracer.aovMode();
    bool foveatOn = pathTracer.foveatedRendering();
    bool tlasOn = pathTracer.tlasEnabled();

    bool   dofEnabled    = false;
    float  lensFStop     = 2.8f;
    float  lensFocusDist = 5.0f;
    int    lensBlades    = 0;
    float  lensRotation  = 0.0f;
    bool   autofocus     = false;

    // Volumetric fog (single-scattering path tracer).
    bool  fogOn       = false;
    float fogDensity  = 0.05f;                   // moderate; let beams show through
    float fogColor[3] = {0.90f, 0.92f, 1.00f};
    float fogG        = 0.75f;                   // strong forward -> god rays
    // Remember user's denoiser preference so we can flip it on with fog
    // (single-sample volume NEE is noisy) and restore afterwards.
    bool  fogSavedDenoiser = denoiserOn;
    bool  fogSavedDirLight = dirLight;
    bool  fogPrevOn        = false;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            raster = !raster;
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
        ImGui::Text("Mode: %s (T to cycle)", raster ? "Raster" : "Path tracer");
        if (!raster) ImGui::Text("Frames: %d", pathTracer.frameCount());

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

        if (!raster && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::SliderFloat("EnvIntensity", &envIntensity, 0.0f, 1.0f))
                pathTracer.setEnvIntensity(envIntensity);
            if (ImGui::SliderFloat("Pixel Scale", &pixelScale, 0.25f, 1.2f, "%.2f"))
                pathTracer.setPixelScale(pixelScale);

            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::Checkbox("REsTDIR DI", &restdirOn))
                pathTracer.setReSTIREnabled(restdirOn);
            if (ImGui::Checkbox("REsTDIR GI", &restdirGIOn))
                pathTracer.setReSTIRGIEnabled(restdirGIOn);
            if (ImGui::Checkbox("Foveated Rendering", &foveatOn))
                pathTracer.setFoveatedRendering(foveatOn);
            if (ImGui::Checkbox("TLAS/BLAS", &tlasOn))
                pathTracer.setTlasEnabled(tlasOn);

            if (ImGui::Checkbox("Show DirLight", &dirLight)) {
                light->visible = dirLight;
            }

            const char* aovItems[] = { "Off", "Depth", "Normals", "Albedo", "Instance ID", "Roughness", "Adaptive Bounce" };
            if (ImGui::Combo("AOV", &aovMode, aovItems, IM_ARRAYSIZE(aovItems)))
                pathTracer.setAOVMode(aovMode);

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Volumetric Fog", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Fog", &fogOn);
                if (fogOn) {
                    ImGui::SliderFloat("Density", &fogDensity, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    ImGui::ColorEdit3("Color", fogColor);
                    ImGui::SliderFloat("Anisotropy g", &fogG, -0.9f, 0.9f, "%.2f");
                    ImGui::TextWrapped("DirLight auto-enabled for god rays. g>0 points beams toward the sun.");
                }
            }

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Lens / DOF")) {
                bool lensChanged = false;
                lensChanged |= ImGui::Checkbox("Enable DOF", &dofEnabled);
                if (dofEnabled) {
                    lensChanged |= ImGui::SliderFloat("f-Stop", &lensFStop, 0.7f, 22.0f, "f/%.1f");
                    lensChanged |= ImGui::SliderFloat("Focus Distance", &lensFocusDist, 0.1f, 50.0f, "%.2f m");
                    const char* bladeItems[] = { "Circular", "3", "4", "5", "6", "7", "8" };
                    int bladeSel = (lensBlades < 3) ? 0 : (lensBlades - 2);
                    if (ImGui::Combo("Aperture Shape", &bladeSel, bladeItems, IM_ARRAYSIZE(bladeItems))) {
                        lensBlades = (bladeSel == 0) ? 0 : (bladeSel + 2);
                        lensChanged = true;
                    }
                    if (lensBlades >= 3)
                        lensChanged |= ImGui::SliderAngle("Aperture Rotation", &lensRotation, -180.f, 180.f);
                    lensChanged |= ImGui::Checkbox("Autofocus", &autofocus);
                    if (!autofocus && loadedModel && ImGui::Button("Focus on model")) {
                        pathTracer.focusOn(camera, *loadedModel);
                        lensFocusDist = pathTracer.lens().focusDistance;
                    }
                    if (autofocus)
                        ImGui::TextDisabled("Tracking orbit target");
                }
                if (lensChanged) {
                    pathTracer.setLens({
                        dofEnabled ? lensFStop : 0.0f,
                        lensFocusDist,
                        lensBlades,
                        lensRotation
                    });
                }
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

        light->intensity = raster ? 1.0f : 10.0f;
        light->visible = dirLight || raster;

        // Check async model load
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
                pathTracer.markDirty();
                std::cout << "Loaded: " << models[currentModel].name << std::endl;
            }
        }

        if (mixer) {
            mixer->update(dt);
        }

        controls.update();

        if (dofEnabled && autofocus) {
            const float targetDist = camera.position.distanceTo(controls.target);
            if (std::abs(targetDist - lensFocusDist) > 0.01f * lensFocusDist) {
                lensFocusDist = targetDist;
                pathTracer.setLens({lensFStop, lensFocusDist, lensBlades, lensRotation});
            }
        }

        if (fogOn != fogPrevOn) {
            if (fogOn) {
                // Enabling fog: stash user's current denoiser/dirLight, force
                // both on for a clear showcase (single-sample NEE is noisy).
                fogSavedDenoiser = denoiserOn;
                fogSavedDirLight = dirLight;
                if (!denoiserOn) { denoiserOn = true; pathTracer.setDenoiserEnabled(true); }
                if (!dirLight)   { dirLight = true;  light->visible = true; }
            } else {
                // Disabling fog: restore the saved state.
                if (denoiserOn != fogSavedDenoiser) { denoiserOn = fogSavedDenoiser; pathTracer.setDenoiserEnabled(denoiserOn); }
                dirLight = fogSavedDirLight;
                light->visible = dirLight || raster;
            }
            fogPrevOn = fogOn;
        }

        if (fogOn) {
            scene.fog = FogExp2(Color(fogColor[0], fogColor[1], fogColor[2]), fogDensity);
            pathTracer.setFogAnisotropy(fogG);
        } else {
            scene.fog.reset();
        }

        if (raster) {
            renderer.render(scene, camera);
        } else {
            pathTracer.render(scene, camera);
        }

        ui.render();
    });
}
