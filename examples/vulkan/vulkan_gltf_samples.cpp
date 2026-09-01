
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>

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
                  << " <path_to_gltf_Models_folder> [--gl] [--solo N] [--debug N]"
                     " [--shot <name.png>] [--frames N]"
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
    // --browse K M: auto-advance to the next model K times, M frames apart
    // (the interactive Left/Right browsing flow, scripted) — the capture then
    // settles + runs on the FINAL model. Repro harness for state that only
    // goes bad after several load/remove cycles.
    int browseK = 0, browseEvery = 40, browseFrame = 0;
    int optProbe = -1, optSunExt = -1;
    float optSunRad = -1.f;
    float orbitDeg = 0.f;// --orbit d: rotate the camera d° per frame during capture (motion-shake harness)
    bool shotAnim = false;// --anim: keep animations playing during capture
    int winW = 0, winH = 0;// --size W H: window size (resolution-dependent behaviour)
    bool camSet = false;   // --cam px py pz tx ty tz: fixed camera (interior shots)
    float camV[6] = {};
    int optSun = -1;       // --sun 0|1: hide/show the example's stand-in DirectionalLight
    int optAo = -1;        // --ao 0|1: half-res RT ambient occlusion + bent normals
    float optLightRad = -1.f;// --lightrad r: physical source radius on loaded point/spot lights (soft-shadow triage)
    bool  fogOnCli = false; float fogDensityCli = 0.05f;// --fog d: FogExp2 at density d (volumetric-fog triage)
    bool  volFogCli = false;// --volfog: enable the volumetric dir-light fog (sun shafts)
    float mblurCli = 0.f;  // --mblur s: post-TAA motion blur, shutter fraction (pair with --orbit)
    int churnN = 0;        // --churn N: add/remove a tiny cube every N frames (structural-rebuild stressor)
    std::string envPath;   // --env <hdr>: environment override (default citrus orchard)
    std::string sunPolicy; // --sunpolicy auto|always|off
    float dofFocus = 0.f;  // --dof S: thin-lens DoF focused at S meters (f/2 aperture)
    int optOccl = -1;      // --occl 0|1: two-phase GPU occlusion culling
    int optLod = -1;       // --lod 0|1: automatic mesh LOD (shading-error triage)
    int optDlss = -1;      // --dlss 0|1: DLSS upscaler (0 falls back to FSR/TAA)
    int optFsr = -1;       // --fsr 0|1: FSR upscaler (--dlss 0 --fsr 0 = built-in TAA)
    int optDenoise = -1;   // --denoise 0|1: deferred SVGF denoiser (raw 1-spp when 0)
    int optMsaa = -1;      // --msaa 1|2|4: raster G-buffer MSAA (edge shading dispatch B)
    int optAutoExp = -1;   // --autoexp 0|1: histogram auto-exposure (interior triage)
    bool useGl = false;    // --gl: run the browser on the OpenGL backend instead
    int soloClip = -1;     // --solo N: play only clip N (-1 = every clip at once)
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--debug" && i + 1 < argc) cliDebugView = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--seq" && i + 1 < argc) seqN = std::atoi(argv[++i]);
        else if (a == "--browse" && i + 2 < argc) { browseK = std::atoi(argv[++i]); browseEvery = std::atoi(argv[++i]); }
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
        else if (a == "--ao" && i + 1 < argc) optAo = std::atoi(argv[++i]);
        else if (a == "--lightrad" && i + 1 < argc) optLightRad = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--fog" && i + 1 < argc) { fogOnCli = true; fogDensityCli = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--volfog") volFogCli = true;
        else if (a == "--mblur" && i + 1 < argc) mblurCli = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--churn" && i + 1 < argc) churnN = std::atoi(argv[++i]);
        else if (a == "--env" && i + 1 < argc) envPath = argv[++i];
        else if (a == "--sunpolicy" && i + 1 < argc) sunPolicy = argv[++i];
        else if (a == "--dof" && i + 1 < argc) dofFocus = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--occl" && i + 1 < argc) optOccl = std::atoi(argv[++i]);
        else if (a == "--lod" && i + 1 < argc) optLod = std::atoi(argv[++i]);
        else if (a == "--dlss" && i + 1 < argc) optDlss = std::atoi(argv[++i]);
        else if (a == "--fsr" && i + 1 < argc) optFsr = std::atoi(argv[++i]);
        else if (a == "--denoise" && i + 1 < argc) optDenoise = std::atoi(argv[++i]);
        else if (a == "--msaa" && i + 1 < argc) optMsaa = std::atoi(argv[++i]);
        else if (a == "--autoexp" && i + 1 < argc) optAutoExp = std::atoi(argv[++i]);
        // Run the same browser on the OpenGL backend. Everything backend-neutral
        // (scene, env/IBL, tone mapping, animation, browsing, capture, the
        // settings panel) works unchanged; the Vulkan-only knobs below are simply
        // skipped, and GL gets shadow maps since it has no ray-traced shadows.
        else if (a == "--gl") useGl = true;
        // Play every clip in the file at once instead of just the first. Default
        // for a sample browser: assets like InterpolationTest ship one clip per
        // node and are meaningless with only clip 0 running. Press A to cycle
        // through the clips solo.
        else if (a == "--solo" && i + 1 < argc) soloClip = std::atoi(argv[++i]);
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

    const std::string title = useGl ? "OpenGL - GLTF Samples" : "Vulkan Deferred - GLTF Samples";

    Canvas canvas(Canvas::Parameters()
                          .title(title)
                          .vsync(false)
                          .antialiasing(4)
                          .size(winW > 0 ? winW : 960, winH > 0 ? winH : 600));

    // One of the two backends owns the window; `renderer` is the backend-neutral
    // handle everything below goes through, and `vk` is non-null only on Vulkan
    // so the deferred-only knobs can be skipped rather than duplicated.
    std::unique_ptr<GLRenderer> glRenderer;
    std::unique_ptr<VulkanRenderer> vkRenderer;
    Renderer* renderer = nullptr;
    VulkanRenderer* vk = nullptr;

    if (useGl) {
        glRenderer = std::make_unique<GLRenderer>(canvas);
        renderer = glRenderer.get();
        // GL has no ray-traced shadows, so give it shadow maps or every model
        // reads as flat-lit. PCF soft is the closest match to the deferred look.
        glRenderer->shadowMap().enabled = true;
        glRenderer->shadowMap().type = ShadowMap::PFCSoft;
    } else {
        vkRenderer = std::make_unique<VulkanRenderer>(canvas);
        renderer = vkRenderer.get();
        vk = vkRenderer.get();
    }

    if (vk) {
        if (optProbe >= 0) vk->setProbeGI(optProbe != 0);
        if (optSunExt >= 0) vk->setEnvSunExtraction(optSunExt != 0);
        if (volFogCli) vk->setVolumetricFog(true);
        if (mblurCli > 0.f) vk->setMotionBlur(mblurCli);
        if (sunPolicy == "always") vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Always);
        else if (sunPolicy == "off") vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Off);
        else if (sunPolicy == "auto") vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Auto);
        if (optSunRad >= 0.f) vk->setSunAngularRadius(optSunRad);
        if (optAo >= 0) vk->setDeferredAO(optAo != 0);
        if (optOccl >= 0) vk->setOcclusionCulling(optOccl != 0);
        if (optLod >= 0) vk->setAutoLod(optLod != 0);
        if (optDlss >= 0) vk->setDlss(optDlss != 0);
        if (optFsr >= 0) vk->setFsr(optFsr != 0);
        if (optDenoise >= 0) vk->setDenoise(optDenoise != 0);
        if (optMsaa > 0) vk->setGbufferMsaa(static_cast<uint32_t>(optMsaa));
        if (optAutoExp >= 0) vk->setAutoExposure(optAutoExp != 0);
        if (dofFocus > 0.f) {// thin-lens DoF: wide-open aperture, focus at S meters
            vk->setCameraExposure(2.0f, 1.f / 125.f, 100.f);
            vk->setFocusDistance(dofFocus);
            vk->setDepthOfField(true);
        }
        vk->setHybridDebugView(cliDebugView);
    } else if (cliDebugView != 0 || volFogCli || mblurCli > 0.f || dofFocus > 0.f) {
        std::cerr << "note: --debug/--volfog/--mblur/--dof are Vulkan-only; ignored on --gl"
                  << std::endl;
    }

    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;

    RGBELoader rgbe;
    auto env = rgbe.load(envPath.empty() ? std::string(DATA_FOLDER) +
                                                   "/textures/env/citrus_orchard_road_puresky_2k.hdr"
                                         : envPath);

    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto sun = DirectionalLight::create(Color(0xffffff), 3.0f);
    sun->position.set(0.4f, 1.0f, 0.3f);
    // Needed by GL's shadow-map pass; the Vulkan backend traces shadows and
    // ignores the flag, so setting it unconditionally costs nothing there.
    sun->castShadow = true;
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

    // One action per clip in the current model. soloClip < 0 runs them all;
    // otherwise only that index contributes (weight 0 elsewhere rather than
    // stop(), so the reset pose is not re-applied every frame).
    std::vector<AnimationAction*> clipActions;
    std::vector<std::string> clipNames;
    auto applyClipSelection = [&] {
        for (size_t i = 0; i < clipActions.size(); ++i) {
            const bool on = soloClip < 0 || static_cast<int>(i) == soloClip;
            clipActions[i]->setEffectiveWeight(on ? 1.f : 0.f).play();
        }
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
                root->traverseType<Mesh>([&](Mesh& m) {
                    hasMesh = true;
                    // Same rationale as the sun's castShadow: required by GL's
                    // shadow-map pass, ignored by the Vulkan tracer.
                    m.castShadow = true;
                    m.receiveShadow = true;
                });
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
            // Narrow the static vertex attributes before first upload (onLoaded
            // runs on the main thread pre-upload). Skinned / morph / deforming
            // sample assets are skipped automatically, so animated glTF samples
            // keep their float attributes and animate exactly as before.
            if (const size_t saved = compressSceneAttributes(g)) {
                std::cout << "[gltf_samples] compressed vertex attributes: "
                          << saved / (1024.0 * 1024.0) << " MiB reclaimed" << std::endl;
            }
            fitCamera(g);
            clipActions.clear();
            clipNames.clear();
            if (!g.animations.empty() && (shotPath.empty() || shotAnim)) {// capture harness: static scene unless --anim
                mixer = std::make_unique<AnimationMixer>(g);

                // EVERY clip, not just the first. Multi-clip sample assets
                // generally drive DISJOINT nodes — InterpolationTest ships nine
                // clips, one per box (Step/Linear/CubicSpline x scale/rotation/
                // translation) — so playing only clip 0 left eight boxes frozen
                // and made the file useless as an interpolation reference. Use
                // --solo N, or press A to cycle, for assets whose clips are
                // alternatives for the same node.
                for (const auto& c : g.animations) {
                    clipActions.push_back(mixer->clipAction(c));
                    clipNames.push_back(c->name());
                }
                applyClipSelection();

                std::cout << "Animations (" << clipNames.size() << "): ";
                for (size_t i = 0; i < clipNames.size(); ++i) {
                    std::cout << (i ? ", " : "") << "[" << i << "] " << clipNames[i];
                }
                std::cout << (soloClip < 0 ? "  -> all playing" : "  -> solo") << std::endl;
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
        if (ev.key == Key::A && !clipActions.empty()) {// cycle: all -> 0 -> 1 -> ... -> all
            soloClip = (soloClip + 1 >= static_cast<int>(clipActions.size())) ? -1 : soloClip + 1;
            applyClipSelection();
            std::cout << "clips: " << (soloClip < 0 ? "all" : "[" + std::to_string(soloClip) + "] " + clipNames[soloClip])
                      << std::endl;
        }
        if (ev.key == Key::C) {// dump the current pose as a --cam repro line
            std::cout << "--cam " << camera.position.x << " " << camera.position.y << " "
                      << camera.position.z << " " << controls.target.x << " "
                      << controls.target.y << " " << controls.target.z << std::endl;
        }
    });
    canvas.addKeyListener(keyAdapter);

    bool dirLight = sun->visible;
    bool fogOn = fogOnCli;
    float fogDensity = fogOnCli ? fogDensityCli : 0.05f;
    float fogColor[3] = {0.55f, 0.6f, 0.7f};
    float fogG = 0.5f;// HG anisotropy: moderately forward-scattering by default

    // Generic renderer settings (exposure, tone map, denoiser, debug views,
    // ...) come from the shared panel; only the scene-specific widgets live
    // here. Interactive runs only — the headless capture paths must not draw
    // UI into the measured frames.
    std::unique_ptr<RendererSettingsUi> ui;
    if (shotPath.empty()) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            ImGui::Text("Model: %s", currentModel >= 0 ? models[currentModel].name.c_str() : "none");
            if (loadedModel && loadedModel->isLoading()) ImGui::Text("Loading...");
            ImGui::Text("Left/Right arrows to browse");

            if (!clipNames.empty()) {
                ImGui::Separator();
                ImGui::Text("Animations (A cycles)");
                bool all = soloClip < 0;
                if (ImGui::RadioButton("All at once", all)) {
                    soloClip = -1;
                    applyClipSelection();
                }
                for (int i = 0; i < static_cast<int>(clipNames.size()); ++i) {
                    if (ImGui::RadioButton(clipNames[i].c_str(), soloClip == i)) {
                        soloClip = i;
                        applyClipSelection();
                    }
                }
            }

            if (ImGui::CollapsingHeader("Models")) {
                for (int i = 0; i < static_cast<int>(models.size()); i++) {
                    const bool selected = (i == currentModel);
                    if (ImGui::Selectable(models[i].name.c_str(), selected)) {
                        loadModel(i);
                    }
                }
            }

            ImGui::Separator();

            if (ImGui::Checkbox("DirLight", &dirLight))
                sun->visible = dirLight;

            if (ImGui::CollapsingHeader("Fog (FogExp2 + HG)")) {
                ImGui::Checkbox("Fog", &fogOn);
                if (fogOn) {
                    ImGui::SliderFloat("Density", &fogDensity, 0.001f, 0.5f, "%.3f",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::ColorEdit3("Color", fogColor);
                    ImGui::SliderFloat("Anisotropy g", &fogG, -0.9f, 0.9f, "%.2f");
                }
            }
        }, "Vulkan Deferred - GLTF Samples");
    }

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer->setSize(ns);
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

        if (mixer) mixer->update(dt);

        // --browse: scripted Left/Right browsing. Advances on a pure frame
        // timer — a fast browser clicks Right BEFORE the previous async load
        // lands, so overlapping loads are part of what this reproduces. Each
        // advance re-arms the capture settle (modelReady/shotFrame) so
        // --frames applies to the final model only.
        if (browseK > 0 && ++browseFrame >= browseEvery) {
            browseFrame = 0;
            --browseK;
            modelReady = false;
            shotFrame  = 0;
            loadModel((currentModel + 1) % static_cast<int>(models.size()));
        }

        if (fogOn) {
            scene.fog = FogExp2(Color(fogColor[0], fogColor[1], fogColor[2]), fogDensity);
            if (vk) vk->setFogAnisotropy(fogG);
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
        renderer->render(scene, camera);

        // Perf harness: average the frame time over the last 60 settle frames
        // before the capture (interleave probe/AO on-off runs in one session).
        static auto perfLast = std::chrono::steady_clock::now();
        static double perfSum = 0.0;
        static int    perfN = 0;
        const auto perfNow = std::chrono::steady_clock::now();
        const double perfDt = std::chrono::duration<double>(perfNow - perfLast).count();
        perfLast = perfNow;
        if (!shotPath.empty() && modelReady && shotFrame >= shotFrames - 60) {
            perfSum += perfDt;
            ++perfN;
        }

        if (shotPath.empty()) {
            ui->render();
        } else if (modelReady && ++shotFrame >= shotFrames) {
            if (perfN > 0)
                std::cout << "[perf] avg " << (perfSum / perfN * 1000.0) << " ms/frame ("
                          << (perfN / perfSum) << " fps) over " << perfN << " frames" << std::endl;
            if (shotFrame == shotFrames && vk) {// env-sun extraction is Vulkan-only
                const auto d = vk->envSunDirection();
                const auto c = vk->envSunColor();
                std::cout << "envSunFound=" << vk->envSunFound()
                          << " dir=(" << d.x << "," << d.y << "," << d.z << ")"
                          << " colorE=(" << c.x << "," << c.y << "," << c.z << ")" << std::endl;
            }
            if (seqN > 0) {// consecutive-frame sequence (temporal-shake harness)
                const auto stem = fs::path(shotPath).stem().string();
                const auto ext  = fs::path(shotPath).extension().string();
                char buf[16];
                std::snprintf(buf, sizeof(buf), "_%03d", seqI);
                const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / (stem + buf + ext);
                renderer->writeFramebuffer(path);
                if (++seqI >= seqN) {
                    std::cout << "wrote " << seqN << " frames to aaa_caps/" << stem << "_*.png" << std::endl;
                    std::exit(0);
                }
            } else {
                const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
                renderer->writeFramebuffer(path);
                std::cout << "wrote " << path.string() << std::endl;
                std::exit(0);
            }
        }
    });

    return 0;
}
