// Vulkan deferred — volumetric clouds showcase (drone flight).
//
// A procedural rolling-hills landscape under a wind-driven, procedurally-shaped
// volumetric cloud deck (VulkanRenderer::setClouds). A scripted drone camera
// takes off, climbs toward the deck, breaks out on top and descends, so the
// clouds can be seen from below (ceiling), from within the layer's altitude,
// and from above (tops). WASD/orbit override with the mouse; press C to toggle
// the clouds, F to toggle the scripted flight.
//
// Headless capture (dev):  vulkan_drone_clouds --shot <name.png> [--frames N]
//                          [--cam 0..3]   (0 = below deck, 1 = mid, 2 = top-down,
//                                          3 = horizon pan)
//
// The cloud layer is a far-field raymarched Perlin-Worley density field lit by
// the scene's DirectionalLight sun (Beer light-march + powder + dual-lobe HG
// phase) composited over the HDR sky.  Phase A: sky-only compositing.

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // ── Value-noise fBm terrain height (CPU) ────────────────────────────────
    float hash2(int x, int y) {
        int n = x * 374761393 + y * 668265263;
        n = (n ^ (n >> 13)) * 1274126177;
        return static_cast<float>((n ^ (n >> 16)) & 0x7fffffff) / static_cast<float>(0x7fffffff);
    }
    float valueNoise(float x, float y) {
        const int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
        float fx = x - xi, fy = y - yi;
        fx = fx * fx * (3.f - 2.f * fx);
        fy = fy * fy * (3.f - 2.f * fy);
        const float a = hash2(xi, yi), b = hash2(xi + 1, yi);
        const float c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }
    float terrainHeight(float x, float z) {
        // fBm rolling hills; peaks ~180 m — well below the cloud base so the
        // deck reads as a distinct ceiling.
        float h = 0.f, amp = 1.f, freq = 1.f / 1400.f, norm = 0.f;
        for (int o = 0; o < 5; ++o) {
            h += amp * valueNoise(x * freq + 13.1f, z * freq + 7.7f);
            norm += amp;
            amp *= 0.5f;
            freq *= 2.03f;
        }
        h /= norm;                          // → [0,1]
        const float ridge = 0.30f + 0.70f * h;
        return 200.f * (ridge * ridge) - 20.f;
    }

    std::shared_ptr<Mesh> makeTerrain() {
        constexpr int   N = 220;     // grid resolution
        constexpr float W = 7000.f;  // world extent (m)
        std::vector<float> pos;
        std::vector<float> col;
        std::vector<unsigned int> idx;
        pos.reserve((N + 1) * (N + 1) * 3);
        for (int j = 0; j <= N; ++j) {
            for (int i = 0; i <= N; ++i) {
                const float x = (static_cast<float>(i) / N - 0.5f) * W;
                const float z = (static_cast<float>(j) / N - 0.5f) * W;
                const float y = terrainHeight(x, z);
                pos.insert(pos.end(), {x, y, z});
                // Height-graded colour: grassy low, rocky high, a touch of snow
                // only on the highest ridges.
                const float t = std::clamp((y + 20.f) / 200.f, 0.f, 1.f);
                Color grass(0.20f, 0.32f, 0.13f), rock(0.36f, 0.34f, 0.30f), snow(0.86f, 0.89f, 0.94f);
                Color c = grass;
                c.lerp(rock, std::clamp((t - 0.25f) / 0.55f, 0.f, 1.f));
                if (t > 0.88f) c.lerp(snow, std::clamp((t - 0.88f) / 0.12f, 0.f, 1.f));
                col.insert(col.end(), {c.r, c.g, c.b});
            }
        }
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const auto a = static_cast<unsigned int>(j * (N + 1) + i);
                const auto b = a + 1;
                const auto c = a + (N + 1);
                const auto d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        geo->computeVertexNormals();
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                        .roughness(0.95f)
                                                        .metalness(0.f)
                                                        .vertexColors(true));
        return Mesh::create(geo, mat);
    }

}// namespace

int main(int argc, char** argv) {
    std::string shotPath;
    int shotFrames = 160, shotFrame = 0, shotCam = 0;
    bool startFree = false;
    bool noClouds = false;// perf A/B: render with the cloud layer off
    bool heightFog = false;// near-field heterogeneous height fog (Phase C)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--cam" && i + 1 < argc) shotCam = std::atoi(argv[++i]);
        else if (a == "--free") startFree = true;
        else if (a == "--noclouds") noClouds = true;
        else if (a == "--heightfog") heightFog = true;
    }

    Canvas canvas("Vulkan Deferred - Volumetric Clouds", {{"vsync", false}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.setAutoExposure(true);
    renderer.setAutoExposureRange(-2.0f, 1.5f);
    renderer.setAutoExposureSpeed(shotPath.empty() ? 2.0f : 12.0f);
    renderer.setRenderScale(0.85f);
    renderer.setSunAngularRadius(0.5f);

    Scene scene;
    RGBELoader rgbe;
    auto envMap = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr");
    scene.background  = envMap;
    scene.environment = envMap;

    scene.add(makeTerrain());

    // The sun — the cloud march lights the deck from this DirectionalLight
    // (one-sun policy; an explicit sun suppresses env-sun injection).
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.9f), 3.2f);
    sun->position.set(-2200.f, 2600.f, 1400.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);
    scene.add(AmbientLight::create(Color(0.55f, 0.62f, 0.72f), 0.4f));

    // The cloud deck.
    VulkanRenderer::CloudSettings cloudCfg;
    cloudCfg.coverage    = 0.5f;
    cloudCfg.density     = 1.0f;
    cloudCfg.bottomY     = 520.f;
    cloudCfg.topY        = 1150.f;
    cloudCfg.wind        = Vector3(14.f, 0.f, 4.f);
    cloudCfg.evolveSpeed = 1.0f;
    bool cloudsOn = !noClouds;
    if (cloudsOn) renderer.setClouds(cloudCfg);

    // Near-field heterogeneous height fog filling the valley (Phase C). Toggled
    // with H interactively or forced on with --heightfog for capture.
    VulkanRenderer::HeightFogSettings hfCfg;
    hfCfg.density     = 0.06f;
    hfCfg.baseY       = 0.f;
    hfCfg.falloff     = 200.f;
    hfCfg.noiseAmount = 0.7f;
    bool heightFogOn = heightFog;
    if (heightFogOn) renderer.setHeightFog(hfCfg);

    PerspectiveCamera camera(55.f, canvas.aspect(), 1.f, 60000.f);
    OrbitControls controls{camera, canvas};
    controls.enabled = shotPath.empty() && startFree;

    // Fixed cinematic vantages for headless capture.
    auto applyShotCam = [&](int cam) {
        switch (cam) {
            case 0:// below the deck, looking up at the cloud ceiling
                camera.position.set(-900.f, 200.f, 900.f);
                camera.lookAt(Vector3(200.f, 780.f, -300.f));
                break;
            case 1:// mid-altitude, deck at eye level across the valley
                camera.position.set(-1400.f, 760.f, 1200.f);
                camera.lookAt(Vector3(300.f, 820.f, -400.f));
                break;
            case 2:// above the deck, looking down at the cloud tops
                camera.position.set(-600.f, 1650.f, 700.f);
                camera.lookAt(Vector3(300.f, 950.f, -500.f));
                break;
            default:// horizon pan just under the deck
                camera.position.set(0.f, 360.f, 1800.f);
                camera.lookAt(Vector3(0.f, 620.f, -2000.f));
                break;
        }
    };

    // Scripted drone flight (interactive): takeoff → climb → breakout → descend.
    auto flightCam = [&](float t) {// t in [0,1), loops
        Vector3 eye, look;
        const float spinX = std::cos(t * 6.2831853f) * 1500.f;
        const float spinZ = std::sin(t * 6.2831853f) * 1500.f;
        float alt;
        if (t < 0.18f) alt = 20.f + (t / 0.18f) * 120.f;               // takeoff
        else if (t < 0.5f) alt = 140.f + ((t - 0.18f) / 0.32f) * 1000.f;// climb through deck
        else if (t < 0.7f) alt = 1140.f + ((t - 0.5f) / 0.2f) * 360.f;  // breakout
        else alt = 1500.f - ((t - 0.7f) / 0.3f) * 1300.f;              // descent
        eye.set(spinX, alt, spinZ);
        // Look toward the centre + a little ahead along the orbit.
        const float la = (t + 0.05f) * 6.2831853f;
        look.set(std::cos(la) * 400.f, alt * 0.75f + 200.f, std::sin(la) * 400.f);
        camera.position.copy(eye);
        camera.lookAt(look);
    };

    if (!shotPath.empty()) applyShotCam(shotCam);
    else if (!startFree) flightCam(0.f);

    bool flightOn = !startFree;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;
    float flightT = 0.f;

    // Generic renderer settings (clouds coverage/density/height live in the
    // panel's Effects section) come from the shared panel; only the demo's
    // hotkey-coupled toggles + the evolve-speed slider (not in the panel) stay
    // here. Interactive runs only — capture frames stay UI-free.
    std::unique_ptr<RendererSettingsUi> ui;
    if (shotPath.empty()) {
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            if (ImGui::Checkbox("Clouds (C)", &cloudsOn))
                renderer.setClouds(cloudsOn ? std::optional{cloudCfg} : std::nullopt);
            ImGui::Checkbox("Scripted flight (F)", &flightOn);
            if (ImGui::SliderFloat("Evolve", &cloudCfg.evolveSpeed, 0.f, 4.f)) {
                if (cloudsOn) renderer.setClouds(cloudCfg);
            }
            ImGui::TextDisabled("F: flight  C: clouds  H: height fog  drag: orbit");
        }, "Volumetric Clouds");
    }

    KeyAdapter keyAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::C) { cloudsOn = !cloudsOn; renderer.setClouds(cloudsOn ? std::optional{cloudCfg} : std::nullopt); }
        else if (evt.key == Key::H) { heightFogOn = !heightFogOn; renderer.setHeightFog(heightFogOn ? std::optional{hfCfg} : std::nullopt); }
        else if (evt.key == Key::F) { flightOn = !flightOn; controls.enabled = !flightOn; }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        if (++fpsFrames, fpsAccum >= 0.5f) { fps = fpsFrames / fpsAccum; fpsAccum = 0.f; fpsFrames = 0; }

        if (shotPath.empty()) {
            if (flightOn) {
                flightT += dt / 60.f;// 60 s loop
                if (flightT >= 1.f) flightT -= 1.f;
                flightCam(flightT);
            } else {
                controls.update();
            }
        }

        renderer.render(scene, camera);
        if (shotPath.empty()) {
            ui->render();
        } else if (++shotFrame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer.writeFramebuffer(path);
            const auto tm = renderer.lastFrameTimings();
            std::cout << "wrote " << path.string() << " (" << fps << " fps)"
                      << "  shade " << tm.pathTraceMs << " ms\n";
            std::exit(0);
        }
    });

    return 0;
}
