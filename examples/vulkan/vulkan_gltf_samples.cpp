
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/VulkanPathTracer.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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

    if (argc < 2) {
        // https://github.com/KhronosGroup/glTF-Sample-Assets
        std::cout << "Usage: " << argv[0]
                  << " <path_to_gltf_Models_folder> [--debug N] [--shot <name.png>] [--frames N]"
                  << std::endl;
        return 1;
    }

    fs::path modelFolder = argv[1];

    // Optional headless capture (dev): pass a debug view + a shot name to grab
    // a G-buffer debug PNG and exit, e.g. `... <folder> --debug 2 --shot mot.png`.
    // 1 Normal, 2 Motion, 3 InstanceID, 4 Albedo.
    std::string shotPath;
    int shotFrames = 120, shotFrame = 0, cliDebugView = 0;
    // Dev harness: --seq N writes N CONSECUTIVE frames (shot stem + _000.png…)
    // after the settle period — frame-to-frame diffs measure temporal shake.
    // --probe/--sunext/--sunrad toggle the deferred features under test.
    int seqN = 0, seqI = 0;
    int optProbe = -1, optSunExt = -1;
    float optSunRad = -1.f;
    float orbitDeg = 0.f;// --orbit d: rotate the camera d° per frame during capture (motion-shake harness)
    bool shotAnim = false;// --anim: keep animations playing during capture
    int winW = 0, winH = 0;// --size W H: window size (resolution-dependent behaviour)
    bool camSet = false;   // --cam px py pz tx ty tz: fixed camera (interior shots)
    float camV[6] = {};
    int optSun = -1;       // --sun 0|1: hide/show the example's stand-in DirectionalLight
    float optLightRad = -1.f;// --lightrad r: physical source radius on loaded point/spot lights (soft-shadow triage)
    int churnN = 0;        // --churn N: add/remove a tiny cube every N frames (structural-rebuild stressor)
    std::string envPath;   // --env <hdr>: environment override (default citrus orchard)
    std::string sunPolicy; // --sunpolicy auto|always|off
    bool usePT = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--debug" && i + 1 < argc) cliDebugView = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--seq" && i + 1 < argc) seqN = std::atoi(argv[++i]);
        else if (a == "--probe" && i + 1 < argc) optProbe = std::atoi(argv[++i]);
        else if (a == "--sunext" && i + 1 < argc) optSunExt = std::atoi(argv[++i]);
        else if (a == "--sunrad" && i + 1 < argc) optSunRad = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--orbit" && i + 1 < argc) orbitDeg = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--anim") shotAnim = true;
        else if (a == "--size" && i + 2 < argc) { winW = std::atoi(argv[++i]); winH = std::atoi(argv[++i]); }
        else if (a == "--cam" && i + 6 < argc) {
            for (int k = 0; k < 6; ++k) camV[k] = static_cast<float>(std::atof(argv[++i]));
            camSet = true;
        }
        else if (a == "--sun" && i + 1 < argc) optSun = std::atoi(argv[++i]);
        else if (a == "--lightrad" && i + 1 < argc) optLightRad = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--churn" && i + 1 < argc) churnN = std::atoi(argv[++i]);
        else if (a == "--env" && i + 1 < argc) envPath = argv[++i];
        else if (a == "--sunpolicy" && i + 1 < argc) sunPolicy = argv[++i];
        else if (a == "--pt") usePT = true;
    }
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

    Canvas canvas(Canvas::Parameters()
                          .title("Vulkan PT - GLTF Samples")
                          .vsync(false)
                          .size(winW > 0 ? winW : 960, winH > 0 ? winH : 600));

    std::unique_ptr<VulkanRendererCore> rendererPtr =
            usePT ? std::unique_ptr<VulkanRendererCore>(std::make_unique<VulkanPathTracer>(canvas))
                  : std::unique_ptr<VulkanRendererCore>(std::make_unique<VulkanRenderer>(canvas));
    VulkanRendererCore& renderer = *rendererPtr;
    auto* pt = dynamic_cast<VulkanPathTracer*>(&renderer);
    if (auto* vr = dynamic_cast<VulkanRenderer*>(&renderer)) {
        if (optProbe >= 0) vr->setProbeGI(optProbe != 0);
        if (optSunExt >= 0) vr->setEnvSunExtraction(optSunExt != 0);
        if (sunPolicy == "always") vr->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Always);
        else if (sunPolicy == "off") vr->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Off);
        else if (sunPolicy == "auto") vr->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Auto);
    }
    if (optSunRad >= 0.f) renderer.setSunAngularRadius(optSunRad);
    renderer.setHybridDebugView(cliDebugView);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    RGBELoader rgbe;
    auto env = rgbe.load(envPath.empty() ? std::string(DATA_FOLDER) +
                                                   "/textures/env/citrus_orchard_road_puresky_2k.hdr"
                                         : envPath);

    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto sun = DirectionalLight::create(Color(0xffffff), 3.0f);
    sun->position.set(0.4f, 1.0f, 0.3f);
    if (optSun >= 0) sun->visible = optSun != 0;
    scene.add(sun);

    PerspectiveCamera camera(50.f, canvas.aspect(), 0.01f, 1000.f);
    camera.position.set(0.f, 1.f, 3.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    GLTFLoader loader;
    bool modelReady = false;// async load complete (gates the capture settle counter)
    int currentModel = -1;
    std::shared_ptr<AsyncGroup> loadedModel;
    std::unique_ptr<AnimationMixer> mixer;

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

    auto loadModel = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(models.size())) return;

        if (loadedModel) {
            scene.remove(*loadedModel);
            loadedModel.reset();
        }
        mixer.reset();

        currentModel = idx;
        auto path = models[idx].path;
        auto name = models[idx].name;
        std::cout << "Loading: " << name << " (" << path << ")" << std::endl;

        loadedModel = loadAsync([&loader, path, name, optLightRad]() -> std::shared_ptr<Group> {
            try {
                auto result = loader.load(path);
                if (!result || !result->scene) {
                    std::cerr << "Load failed '" << name << "'" << std::endl;
                    return nullptr;
                }
                auto& root = result->scene;
                bool hasMesh = false;
                root->traverseType<Mesh>([&](Mesh&) { hasMesh = true; });
                root->traverseType<Light>([&](Light& l) {
                    l.visible = true;
                    l.intensity = std::max(l.intensity, 1.0f);
                    // --lightrad: physical source radius on the model's punctual
                    // lights (RT soft-shadow triage; 0/absent = exact hard).
                    if (optLightRad >= 0.f) {
                        if (auto* pl = dynamic_cast<PointLight*>(&l)) pl->radius = optLightRad;
                        else if (auto* sl = dynamic_cast<SpotLight*>(&l)) sl->radius = optLightRad;
                    }
                });
                if (!hasMesh) {
                    std::cerr << "Skipping '" << name << "': no mesh geometry" << std::endl;
                    return nullptr;
                }
                root->animations = result->animations;
                return root;
            } catch (const std::exception& e) {
                std::cerr << "Load failed '" << name << "': " << e.what() << std::endl;
                return nullptr;
            }
        });

        loadedModel->onLoaded([&](AsyncGroup& g) {
            fitCamera(g);
            if (!g.animations.empty() && (shotPath.empty() || shotAnim)) {// capture harness: static scene unless --anim
                mixer = std::make_unique<AnimationMixer>(g);
                mixer->clipAction(g.animations.front())->play();
                std::cout << "Playing animation: " << g.animations.front()->name()
                          << " (" << g.animations.size() << " clip(s))" << std::endl;
            }
            std::cout << "Loaded: " << models[currentModel].name << std::endl;
            modelReady = true;// capture harness: --frames settle counts from HERE
        });

        scene.add(loadedModel);
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
    int spp = pt ? pt->samplesPerPixel() : 1;
    bool fogOn = false;
    float fogDensity = 0.05f;
    float fogColor[3] = {0.55f, 0.6f, 0.7f};
    float fogG = 0.5f;// HG anisotropy: moderately forward-scattering by default
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({300, 0});
        ImGui::Begin("Vulkan PT - GLTF Samples");
        ImGui::Text("FPS: %.1f", fps);

        ImGui::Separator();
        ImGui::Text("Model: %s", currentModel >= 0 ? models[currentModel].name.c_str() : "none");
        if (loadedModel && loadedModel->isLoading()) ImGui::Text("Loading...");
        ImGui::Text("Left/Right arrows to browse");
        ImGui::TextDisabled(pt ? "Mode: Path tracer (--pt)" : "Mode: Deferred (default)");

        // Raster G-buffer debug views. "Albedo" exercises the new raster-first
        // material attachment (linear base colour in rgb, metalness in alpha).
        static int debugView = 0;
        const char* dbgItems[] = {"Off (PT)", "Normal", "Motion", "InstanceID", "Albedo"};
        if (ImGui::Combo("Debug view", &debugView, dbgItems, IM_ARRAYSIZE(dbgItems)))
            renderer.setHybridDebugView(debugView);

        if (ImGui::CollapsingHeader("Models")) {
            for (int i = 0; i < static_cast<int>(models.size()); i++) {
                const bool selected = (i == currentModel);
                if (ImGui::Selectable(models[i].name.c_str(), selected)) {
                    loadModel(i);
                }
            }
        }

        ImGui::Separator();

        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 2.0f))
            renderer.toneMappingExposure = exposure;

        const char* toneItems[] = {"None", "Linear", "Reinhard", "Cineon", "ACESFilmic"};
        if (ImGui::Combo("Tone mapping", &toneMode, toneItems, IM_ARRAYSIZE(toneItems)))
            renderer.toneMapping = static_cast<ToneMapping>(toneMode);

        if (ImGui::Checkbox("DirLight", &dirLight))
            sun->visible = dirLight;

        bool denoise = renderer.denoise();
        if (ImGui::Checkbox("Denoise", &denoise)) {
            renderer.setDenoise(denoise);
        }

        if (pt) {
            bool perSpp = pt->perSppJitterHybrid();
            if (ImGui::Checkbox("Per-spp AA jitter", &perSpp))
                pt->setPerSppJitterHybrid(perSpp);
        }

        bool restirDI = renderer.restirDIEnabled();
        if (ImGui::Checkbox("ReSTIR DI", &restirDI)) {
            renderer.setRestirDIEnabled(restirDI);
        }

        if (pt) {
            bool restirGI = pt->restirGIEnabled();
            if (ImGui::Checkbox("ReSTIR GI", &restirGI)) {
                pt->setRestirGIEnabled(restirGI);
            }
        }

        if (ImGui::CollapsingHeader("Fog (FogExp2 + HG)")) {
            ImGui::Checkbox("Fog", &fogOn);
            if (fogOn) {
                ImGui::SliderFloat("Density", &fogDensity, 0.001f, 0.5f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::ColorEdit3("Color", fogColor);
                ImGui::SliderFloat("Anisotropy g", &fogG, -0.9f, 0.9f, "%.2f");
            }
        }

        if (pt && ImGui::SliderInt("Samples / pixel", &spp, 1, 16))
            pt->setSamplesPerPixel(spp);

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

    // --churn: a tiny far-corner cube toggled in/out of the scene every N
    // frames — emulates gameplay spawn/despawn (tracers, casings, decals) to
    // stress STRUCTURAL scene rebuilds without changing the visible image.
    std::shared_ptr<Mesh> churnCube;
    if (churnN > 0) {
        churnCube = Mesh::create(BoxGeometry::create(0.01f, 0.01f, 0.01f),
                                 MeshStandardMaterial::create());
        // INSIDE the scene bounds (like gameplay spawns — tracers, casings):
        // an out-of-bounds position would legitimately change the scene AABB
        // and defeat the probe-grid hysteresis this flag exists to test.
        churnCube->position.set(0.f, 0.5f, 0.f);
    }
    int churnFrame = 0;

    Clock clock;
    canvas.animate([&] {
        if (churnCube && ++churnFrame % churnN == 0) {
            if (churnCube->parent) scene.remove(*churnCube);
            else scene.add(churnCube);
        }
        const float dt = clock.getDelta();
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (mixer) mixer->update(dt);

        if (fogOn) {
            scene.fog = FogExp2(Color(fogColor[0], fogColor[1], fogColor[2]), fogDensity);
            renderer.setFogAnisotropy(fogG);
        } else {
            scene.fog.reset();
        }

        // Orbit only through the last stretch of the settle + the capture — a
        // full-settle orbit sweeps the camera tens of degrees off the --cam pose.
        const bool orbitNow = !shotPath.empty() && orbitDeg != 0.f && shotFrame >= shotFrames - 60;
        if (orbitNow) {
            // Constant slow orbit about the target's Y axis — exercises the
            // temporal/reprojection paths a static capture never touches.
            // With --cam it rotates the STORED base so the motion accumulates
            // (re-applying the fixed pose each frame would cancel it).
            const float a = orbitDeg * math::PI / 180.f;
            if (camSet) {
                const float rx = camV[0] - camV[3], rz = camV[2] - camV[5];
                camV[0] = camV[3] + rx * std::cos(a) - rz * std::sin(a);
                camV[2] = camV[5] + rx * std::sin(a) + rz * std::cos(a);
                camera.position.set(camV[0], camV[1], camV[2]);
                controls.target.set(camV[3], camV[4], camV[5]);
            } else {
                const Vector3 rel = camera.position.clone().sub(controls.target);
                camera.position.set(controls.target.x + rel.x * std::cos(a) - rel.z * std::sin(a),
                                    camera.position.y,
                                    controls.target.z + rel.x * std::sin(a) + rel.z * std::cos(a));
            }
            camera.lookAt(controls.target);
        } else if (camSet) {// fixed interior camera overrides controls/fitCamera
            camera.position.set(camV[0], camV[1], camV[2]);
            controls.target.set(camV[3], camV[4], camV[5]);
            camera.lookAt(controls.target);
        } else if (shotPath.empty() || orbitDeg == 0.f) {
            controls.update();
        }
        renderer.render(scene, camera);

        if (shotPath.empty()) {
            ui.render();
        } else if (modelReady && ++shotFrame >= shotFrames) {
            if (auto* vr = dynamic_cast<VulkanRenderer*>(&renderer); vr && shotFrame == shotFrames) {
                const auto d = vr->envSunDirection();
                const auto c = vr->envSunColor();
                std::cout << "envSunFound=" << vr->envSunFound()
                          << " dir=(" << d.x << "," << d.y << "," << d.z << ")"
                          << " colorE=(" << c.x << "," << c.y << "," << c.z << ")" << std::endl;
            }
            if (seqN > 0) {// consecutive-frame sequence (temporal-shake harness)
                const auto stem = fs::path(shotPath).stem().string();
                const auto ext  = fs::path(shotPath).extension().string();
                char buf[16];
                std::snprintf(buf, sizeof(buf), "_%03d", seqI);
                const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / (stem + buf + ext);
                renderer.writeFramebuffer(path);
                if (++seqI >= seqN) {
                    std::cout << "wrote " << seqN << " frames to aaa_caps/" << stem << "_*.png" << std::endl;
                    std::exit(0);
                }
            } else {
                const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
                renderer.writeFramebuffer(path);
                std::cout << "wrote " << path.string() << std::endl;
                std::exit(0);
            }
        }
    });

    return 0;
}
