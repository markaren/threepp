// Vulkan PT — procedural mountain configurator (M1 + M2).
//
// A single hero mountain massif rendered through the Vulkan path tracer, with a
// live ImGui panel that re-rolls and re-shapes the terrain. The heightfield is
// generated on the CPU by threepp::terrain::TerrainGenerator (fBm / ridged /
// hybrid multifractal + domain warp), optionally carved by droplet-hydraulic +
// thermal/talus EROSION, and baked into a horizontal PlaneGeometry; the renderer
// rebuilds the mesh BLAS automatically when the displaced vertex positions
// change (the plain dynamic-geometry path — no special mesh type, no renderer
// surgery). Lighting is HDRI image-based + a directional sun.
//
// Noise/shape edits re-roll a fast RAW preview on release; the (slower) erosion
// pass runs on demand via the Generate button. Presets bake fully eroded.
//
// M1 = noise heightfield + configurator; M2 = erosion (this file). Slope/
// altitude texturing (M3) and atmosphere/haze polish (M4) build on top.
//
// Headless capture (dev): vulkan_mountains --shot <name.png> [--frames N]
//   [--preset 0..3] [--no-erode] [--pt]

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;

    // A named preset is just a TerrainParams value set (the configurator's
    // "Custom" entry leaves the current values untouched).
    void applyPreset(int preset, TerrainParams& p) {
        switch (preset) {
            case 0:// Alpine — ridged peaks. (Crisper drainage / eroded ridges land in M2.)
                p.noiseType = NoiseType::Ridged;
                p.worldSize = 1200.f;
                p.featureScale = 520.f;
                p.octaves = 6;// keep finest octave above the mesh sampling limit (avoids aliased spikes)
                p.lacunarity = 2.0f;
                p.gain = 0.5f;
                p.amplitude = 480.f;
                p.warp = 0.45f;
                p.ridgeSharpness = 0.5f;
                p.heightExponent = 1.25f;
                p.terraces = 0;
                p.falloff = Falloff::Radial;
                p.falloffStart = 0.62f;
                p.erosion = ErosionType::Both;// hydraulic drainage + light talus on the ridges
                p.droplets = 110000;
                p.erodeSpeed = 0.4f;
                p.erosionRadius = 3;
                p.talusAngle = 42.f;
                p.thermalIterations = 22;// keep ridgelines crisp between carved valleys
                break;
            case 1:// Rolling hills — soft fBm; hills across most of the patch, tapering at the rim.
                p.noiseType = NoiseType::fBm;
                p.worldSize = 1600.f;
                p.featureScale = 430.f;
                p.octaves = 6;
                p.lacunarity = 2.0f;
                p.gain = 0.5f;
                p.amplitude = 300.f;
                p.warp = 0.3f;
                p.ridgeSharpness = 0.0f;
                p.heightExponent = 1.0f;
                p.terraces = 0;
                p.falloff = Falloff::Radial;
                p.falloffStart = 0.8f;
                p.erosion = ErosionType::Thermal;// gentle talus smoothing
                p.talusAngle = 32.f;
                p.thermalIterations = 40;
                break;
            case 2:// Desert mesa — terraced strata.
                p.noiseType = NoiseType::Hybrid;
                p.worldSize = 1400.f;
                p.featureScale = 520.f;
                p.octaves = 6;
                p.lacunarity = 2.0f;
                p.gain = 0.45f;
                p.amplitude = 320.f;
                p.warp = 0.25f;
                p.ridgeSharpness = 0.2f;
                p.heightExponent = 1.0f;
                p.terraces = 10;
                p.falloff = Falloff::Radial;
                p.falloffStart = 0.72f;
                p.erosion = ErosionType::Hydraulic;// gullies cut into the strata
                p.droplets = 60000;
                p.erodeSpeed = 0.3f;
                p.depositSpeed = 0.4f;
                break;
            case 3:// Volcanic — single radial cone.
                p.noiseType = NoiseType::Ridged;
                p.worldSize = 1200.f;
                p.featureScale = 360.f;
                p.octaves = 6;// fewer octaves: ridges, not aliased spikes
                p.lacunarity = 2.2f;
                p.gain = 0.55f;
                p.amplitude = 600.f;
                p.warp = 0.35f;
                p.ridgeSharpness = 0.45f;
                p.heightExponent = 1.4f;
                p.terraces = 0;
                p.falloff = Falloff::Radial;
                p.falloffStart = 0.5f;
                p.erosion = ErosionType::Both;// deep radial channels + talus to tame the spires
                p.droplets = 100000;
                p.erodeSpeed = 0.45f;
                p.erosionRadius = 2;
                p.talusAngle = 40.f;
                p.thermalIterations = 22;
                break;
            default: break;
        }
    }

}// namespace

int main(int argc, char** argv) {

    // Headless capture (dev): vulkan_mountains --shot <name.png> [--frames N] [--pt]
    std::string shotPath;
    int shotFrames = 240, shotFrame = 0;
    bool shotPT = false;
    bool stress = false;// re-roll repeatedly to exercise the runtime regen / BLAS-rebuild path
    bool rerollOnce = false;// single in-place re-roll at frame 20, then settle (motion-vector test)
    bool noErode = false;// force erosion off (capture the pre-erosion baseline)
    bool topDown = false;// overhead camera (inspect drainage channels)
    int startPreset = 0;// 0 Alpine, 1 Rolling, 2 Mesa, 3 Volcanic
    float sceneScale = 1.f;     // debug: scale world/amplitude/feature (precision A/B)
    float ovSunAz = -1, ovSunEl = -1;// debug: override sun azimuth/elevation
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (std::string(argv[i]) == "--frames" && i + 1 < argc)
            shotFrames = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--preset" && i + 1 < argc)
            startPreset = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--scale" && i + 1 < argc)
            sceneScale = static_cast<float>(std::atof(argv[++i]));
        else if (std::string(argv[i]) == "--sun" && i + 2 < argc) {
            ovSunAz = static_cast<float>(std::atof(argv[++i]));
            ovSunEl = static_cast<float>(std::atof(argv[++i]));
        } else if (std::string(argv[i]) == "--stress")
            stress = true;
        else if (std::string(argv[i]) == "--reroll")
            rerollOnce = true;
        else if (std::string(argv[i]) == "--no-erode")
            noErode = true;
        else if (std::string(argv[i]) == "--topdown")
            topDown = true;
        else if (std::string(argv[i]) == "--pt")
            shotPT = true;
    }

    Canvas canvas("Vulkan PT - Mountains", {{"vsync", false}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    if (shotPT) renderer.setRenderMode(VulkanRenderer::RenderMode::ReferencePT);

    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;// image-based lighting (env CDF + MIS in the PT)
    } else {
        scene.background = Color(0.55f, 0.70f, 0.92f);
        std::cerr << "[mountains] HDRI not found — falling back to flat sky background\n";
    }

    // Directional sun. Driven each frame from azimuth/elevation sliders; the
    // HDRI already carries a sun, so this mostly rakes the ridges for clear
    // relief and drives the shadow direction.
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.88f), 2.6f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // Terrain.
    TerrainParams params;
    startPreset = std::clamp(startPreset, 0, 3);
    applyPreset(startPreset, params);
    params.resolution = 512;// good detail/cost balance; 256 for snappier re-rolls, 1024 for a final bake
    if (sceneScale != 1.f) {// keep terrain shape identical, just change world-coordinate magnitude
        params.worldSize *= sceneScale;
        params.amplitude *= sceneScale;
        params.featureScale *= sceneScale;
    }
    if (noErode) params.erosion = ErosionType::None;

    TerrainGenerator gen(params.seed);
    auto terrainMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                           .color(Color(0.46f, 0.43f, 0.40f))
                                                           .roughness(0.93f)
                                                           .metalness(0.0f));
    // Initial bake runs erosion if the preset calls for it (a ~1s one-off).
    auto terrain = Mesh::create(gen.createGeometry(params, params.erosion != ErosionType::None), terrainMat);
    scene.add(terrain);

    // Surrounding plain — a large flat ground a touch below the base so the
    // EdgeFade'd massif rises out of it instead of floating. Sits just under
    // y=0 (terrain valley floors) to avoid a coplanar z-fight at the seam.
    auto groundMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                          .color(Color(0.34f, 0.35f, 0.30f))
                                                          .roughness(1.0f)
                                                          .metalness(0.0f));
    auto ground = Mesh::create(PlaneGeometry::create(params.worldSize * 12.f, params.worldSize * 12.f), groundMat);
    ground->rotation.x = -math::PI / 2.f;
    ground->position.y = -2.0f;// terrain rim sinks below this and is occluded by the plain
    scene.add(ground);

    int builtResolution = params.resolution;
    float builtWorldSize = params.worldSize;

    // Side-on 3/4 view: ridges read against the sky rather than as a top-down
    // spike field, and the finite-patch edges sit closer to the horizon.
    PerspectiveCamera camera(50.f, canvas.aspect(), 1.f, 10000.f);
    camera.position.set(params.worldSize * 0.95f, params.amplitude * 0.85f, params.worldSize * 0.95f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, params.amplitude * 0.45f, 0.f);
    if (topDown) {// overhead — reveals the drainage network on the surface
        camera.position.set(0.f, params.worldSize * 1.3f, 0.1f);
        controls.target.set(0.f, 0.f, 0.f);
    }
    controls.update();

    // ---- Configurator state ----
    int preset = startPreset;// 0 Alpine, 1 Rolling, 2 Mesa, 3 Volcanic, 4 Custom
    int noiseTypeIdx = static_cast<int>(params.noiseType);
    int falloffIdx = static_cast<int>(params.falloff);
    int erosionIdx = static_cast<int>(params.erosion);
    float sunAzimuth = ovSunAz >= 0 ? ovSunAz : 135.f;
    float sunElevation = ovSunEl >= 0 ? ovSunEl : 32.f;
    int spp = renderer.samplesPerPixel();

    bool regenRequested = false;
    bool regenErode = false;// when servicing a regen, also run the (slow) erosion pass
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    // Noise/shape edits re-roll a fast RAW preview (no erosion); the eroded
    // result is produced on demand by the Generate button (erosion is the
    // expensive pass, ~1s, so it must not run on every slider release).
    auto markCustom = [&](bool changed) { if (changed) { preset = 4; regenRequested = true; } };

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Mountain Configurator");

        ImGui::Text("FPS: %.1f   verts: %d", fps,
                    terrain->geometry() && terrain->geometry()->getAttribute<float>("position")
                            ? terrain->geometry()->getAttribute<float>("position")->count()
                            : 0);
        ImGui::Separator();

        // Preset + seed.
        if (ImGui::Combo("Preset", &preset, "Alpine\0Rolling Hills\0Desert Mesa\0Volcanic\0Custom\0")) {
            if (preset < 4) {
                applyPreset(preset, params);
                noiseTypeIdx = static_cast<int>(params.noiseType);
                falloffIdx = static_cast<int>(params.falloff);
                erosionIdx = static_cast<int>(params.erosion);
                regenRequested = true;
                regenErode = (params.erosion != ErosionType::None);// presets define a complete eroded look
            }
        }
        {
            int seedI = static_cast<int>(params.seed);
            if (ImGui::InputInt("Seed", &seedI)) {
                params.seed = static_cast<unsigned int>(std::max(seedI, 0));
                regenRequested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize")) {
                params.seed = std::random_device{}();
                regenRequested = true;
            }
        }

        ImGui::SeparatorText("Grid");
        {
            static const int resVals[4] = {128, 256, 512, 1024};
            int resIdx = 1;
            for (int i = 0; i < 4; ++i)
                if (resVals[i] == params.resolution) resIdx = i;
            if (ImGui::Combo("Resolution", &resIdx, "128\0" "256\0" "512\0" "1024\0")) {
                params.resolution = resVals[resIdx];
                regenRequested = true;
            }
            markCustom(ImGui::SliderFloat("World size (m)", &params.worldSize, 200.f, 4000.f, "%.0f"));
        }

        ImGui::SeparatorText("Base noise");
        if (ImGui::Combo("Type", &noiseTypeIdx, "fBm\0Ridged\0Hybrid\0")) {
            params.noiseType = static_cast<NoiseType>(noiseTypeIdx);
            preset = 4;
            regenRequested = true;
        }
        markCustom(ImGui::SliderFloat("Feature scale (m)", &params.featureScale, 40.f, 2000.f, "%.0f"));
        markCustom(ImGui::SliderInt("Octaves", &params.octaves, 1, 11));
        markCustom(ImGui::SliderFloat("Lacunarity", &params.lacunarity, 1.5f, 3.0f, "%.2f"));
        markCustom(ImGui::SliderFloat("Gain", &params.gain, 0.2f, 0.8f, "%.2f"));
        markCustom(ImGui::SliderFloat("Amplitude (m)", &params.amplitude, 0.f, 1500.f, "%.0f"));
        markCustom(ImGui::SliderFloat("Domain warp", &params.warp, 0.f, 1.f, "%.2f"));

        ImGui::SeparatorText("Shape");
        if (params.noiseType != NoiseType::fBm)
            markCustom(ImGui::SliderFloat("Ridge sharpness", &params.ridgeSharpness, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Height exponent", &params.heightExponent, 0.6f, 2.0f, "%.2f"));
        markCustom(ImGui::SliderInt("Terraces", &params.terraces, 0, 16));
        if (ImGui::Combo("Falloff", &falloffIdx, "None\0Radial\0")) {
            params.falloff = static_cast<Falloff>(falloffIdx);
            preset = 4;
            regenRequested = true;
        }

        ImGui::SeparatorText("Erosion");
        // Erosion knobs only take effect on the next Generate (it's the slow
        // pass). Noise/shape edits above show the raw, un-eroded shape live.
        if (ImGui::Combo("Erosion type", &erosionIdx, "None\0Hydraulic\0Thermal\0Both\0")) {
            params.erosion = static_cast<ErosionType>(erosionIdx);
            preset = 4;
        }
        ImGui::SliderInt("Droplets", &params.droplets, 0, 400000);
        ImGui::SliderFloat("Erode rate", &params.erodeSpeed, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Deposit rate", &params.depositSpeed, 0.f, 1.f, "%.2f");
        ImGui::SliderInt("Erosion radius", &params.erosionRadius, 1, 6);
        ImGui::SliderFloat("Talus angle", &params.talusAngle, 20.f, 60.f, "%.0f");
        ImGui::SliderInt("Thermal iters", &params.thermalIterations, 0, 200);
        if (ImGui::Button("Generate (erode)")) {
            preset = 4;
            regenRequested = true;
            regenErode = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("~1s pass");

        ImGui::SeparatorText("Sun & render");
        ImGui::SliderFloat("Sun azimuth", &sunAzimuth, 0.f, 360.f, "%.0f");
        ImGui::SliderFloat("Sun elevation", &sunElevation, 1.f, 89.f, "%.0f");
        ImGui::SliderFloat("Sun intensity", &sun->intensity, 0.f, 8.f, "%.2f");
        ImGui::SliderFloat("Exposure", &renderer.toneMappingExposure, 0.1f, 3.0f, "%.2f");
        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 16))
            renderer.setSamplesPerPixel(spp);

        ImGui::Separator();
        ImGui::TextDisabled("Drag = orbit, scroll = zoom");
        ImGui::TextDisabled("Noise edits preview raw on release;");
        ImGui::TextDisabled("press Generate to erode.");
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

        // Single in-place re-roll at frame 20 (then settle): exercises the
        // applyTo path and lets the motion vector settle for the prevVertex
        // re-sync test. With the bug, the re-rolled mesh shows constant nonzero
        // motion forever; fixed, it returns to zero a frame after the re-roll.
        {
            static int loopFrame = 0;
            ++loopFrame;
            if (rerollOnce && loopFrame == 20) {
                params.seed += 24601u;
                regenRequested = true;
                rerollOnce = false;
                std::cout << "[reroll] in-place re-roll, seed=" << params.seed << std::endl;
            }
        }

        // Stress mode: drive the same regen path the UI uses, alternating
        // resolution so both the same-topology refit and the topology-change
        // rebuild get exercised at runtime. Headless validation of "re-roll".
        if (stress) {
            static int sTick = 0;
            if (++sTick % 25 == 0) {
                params.seed += 7919u;
                params.resolution = (params.resolution == 512 ? 256 : 512);
                regenRequested = true;
                std::cout << "[stress] re-roll seed=" << params.seed
                          << " res=" << params.resolution << std::endl;
            }
        }

        // Sun direction from azimuth/elevation. position = direction * range
        // (DirectionalLight points from position toward its target at origin).
        const float az = sunAzimuth * kDeg2Rad;
        const float el = sunElevation * kDeg2Rad;
        sun->position.set(std::cos(el) * std::sin(az),
                          std::sin(el),
                          std::cos(el) * std::cos(az));
        sun->position.multiplyScalar(std::max(params.worldSize, 1000.f));

        // Regenerate the terrain once the user releases the control (avoids a
        // per-frame BLAS rebuild while a slider is being dragged). In headless
        // capture/stress mode there is no live UI, so regenerate immediately.
        const bool uiBusy = shotPath.empty() && ImGui::IsAnyItemActive();
        if (regenRequested && !uiBusy) {
            if (params.seed != gen.seed()) gen.reseed(params.seed);
            gen.buildField(params);              // noise heightfield (fast)
            if (regenErode) gen.erode(params);   // droplet + thermal (the ~1s pass)
            if (params.resolution != builtResolution || params.worldSize != builtWorldSize) {
                terrain->setGeometry(gen.makeGeometry(params));
                builtResolution = params.resolution;
                builtWorldSize = params.worldSize;
            } else {
                gen.displaceTo(*terrain->geometry(), params);
            }
            regenRequested = false;
            regenErode = false;
        }

        renderer.render(scene, camera);

        if (shotPath.empty()) {
            ui.render();
        } else if (++shotFrame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer.writeFramebuffer(path);
            std::cout << "wrote " << path.string() << std::endl;
            std::exit(0);
        }
    });

    return 0;
}
