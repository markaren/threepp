// Vulkan PT — minimal ocean.
//
// The "fancy water" (3-cascade Phillips/FFT-displaced surface with foam and
// transmission) with NO hero object: just an env HDR, a sand floor, and an
// Ocean. Where examples/vulkan/vulkan_ocean.cpp wraps the same surface in a
// boat, lighthouse, archipelago, LIDAR and radar, this shows that the water
// itself is fully standalone — the whole scene is a handful of lines built on
// the first-party threepp::Ocean type (objects/Ocean.hpp).
//
// The adaptive vertex warp follows the orbit target (the point you're looking
// at), demonstrating that the density focus is just a world coordinate — it was
// never tied to the boat; here it's the camera.
//
// Controls: drag to orbit, scroll to zoom.
//   Up/Down    — wind speed ±1 m/s (live; the sea morphs into the new state)
//   Left/Right — wind direction ∓15°
// Headless capture: vulkan_ocean_minimal --shot <name.png> [--frames N] [--pt]
//   (plus --cam x,y,z / --look x,y,z to reframe with no rebuild, and
//   --wind <m/s> to capture a specific sea state).
// Validation hooks:
//   --pond        — 16 m garden-pond preset: shallow sandy bottom + rocks,
//                   calm wind. Exercises the scale-aware Ocean defaults and
//                   the shallow-water (visible-bottom) transmission path.
//   --wind2 <m/s> — setWind() to this halfway through a --shot run; exercises
//                   the live-wind respectrum path headlessly.
//   --probes      — 3×3 emissive spheres pinned to sampleHeight() around the
//                   orbit target: the CPU/GPU wave-field parity check (spheres
//                   must sit ON the surface), and a lazy-height-readback smoke
//                   test in one flag.

#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include "capture_util.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

using namespace threepp;

int main(int argc, char** argv) {

    // ── Headless capture (dev iteration): N warm-up frames then one PNG ──────
    std::string shotPath;
    int shotFrames = 240, shotFrame = 0;
    float windOverride = -1.f;// <0 = keep the Ocean default
    float wind2 = -1.f;       // ≥0 = setWind() to this halfway through a --shot run (live-wind test)
    bool probes = false;      // sampleHeight parity probes (debug)
    bool pond = false;        // 16 m pond preset: shallow bright bottom, calm wind
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--wind") == 0 && i + 1 < argc) windOverride = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--wind2") == 0 && i + 1 < argc) wind2 = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--probes") == 0) probes = true;
        else if (std::strcmp(argv[i], "--pond") == 0) pond = true;
    }
    const capture::Args capArgs = capture::parseArgs(argc, argv);
    if (capArgs.frames) shotFrames = *capArgs.frames;

    Canvas canvas("Vulkan PT - Ocean (minimal)", {{"vsync", false}, {"size", WindowSize{1600, 900}}});

    auto renderer = VulkanRenderer(canvas);
    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);

    // Trace at slightly reduced resolution; TAA upsamples by accumulating
    // jittered low-res samples into the full-res history.
    renderer.setRenderScale(0.9f);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.7f;

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) +
                         "/textures/env/autumn_field_puresky_2k.hdr");

    Scene scene;
    if (env) {
        scene.background = env;  // sky shows wherever rays miss the water
        scene.environment = env; // image-based lighting for the surface
    }

    // Gentle sun — the HDR already carries one (env CDF + MIS importance-sample
    // it); this mostly drives the photon-mapped caustics on the sand floor.
    auto sun = DirectionalLight::create(Color(1.0f, 0.95f, 0.85f), 2.0f);
    sun->position.set(2.f, 1.f, 2.f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // --pond swaps the 1 km deep ocean for a 16 m garden pond: shallow bright
    // sandy bottom with a few rocks, calm wind. Exercises the scale-aware
    // Ocean defaults (auto tiles → dm-scale ripples) and the shallow-water
    // transmission (the bottom is genuinely SEEN through the surface via the
    // refracted-ray path in deferred_shade — not a repainted water body).
    const float kSize = pond ? 16.0f : 1000.0f;

    // Floor under the water. Ocean: near-black 5 m down — reads as deep water
    // (the analytic body dominates). Pond: bright sand 1.2 m down so the
    // shallow transmission has something worth refracting to.
    auto floorMat = MeshStandardMaterial::create(
            pond ? MeshStandardMaterial::Params{}.color(Color(0.55f, 0.42f, 0.24f)).roughness(1.0f)
                 : MeshStandardMaterial::Params{}.color(Color(0.02f, 0.02f, 0.02f)).roughness(1.0f));
    auto floor = Mesh::create(PlaneGeometry::create(pond ? kSize * 3.f : kSize,
                                                    pond ? kSize * 3.f : kSize),
                              floorMat);
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = pond ? -1.2f : -5.f;
    scene.add(floor);

    if (pond) {
        // A few rocks on the bottom — proof the refracted view resolves real
        // geometry (position, tint shift with depth), plus one breaking the
        // surface as an above-water reference.
        auto rockMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.42f, 0.38f, 0.33f)).roughness(0.9f));
        const struct { float x, z, r, y; } rocks[] = {
                {-2.5f, -1.5f, 0.55f, -1.05f}, {1.8f, 0.8f, 0.75f, -0.95f},
                {0.2f, -3.0f, 0.4f, -1.1f},    {3.4f, -2.2f, 0.5f, -1.0f},
                {-1.2f, 2.6f, 1.05f, -0.35f}// last one clearly pierces the surface
        };
        for (const auto& r : rocks) {
            auto rock = Mesh::create(SphereGeometry::create(r.r, 24, 18), rockMat);
            rock->position.set(r.x, r.y, r.z);
            rock->scale.y = 0.75f;// flattened boulders
            scene.add(rock);
        }
    }

    // The whole "fancy water" in one line. For the pond, size alone retunes
    // the cascades (auto tiles); calm default wind unless overridden.
    Ocean::Options opts;
    opts.size = kSize;
    if (pond) opts.windSpeed = 3.0f;
    if (windOverride >= 0.f) opts.windSpeed = windOverride;
    auto ocean = Ocean::create(opts);
    scene.add(ocean);

    // Live wind control — setWind() takes effect next frame; the sea morphs
    // smoothly because the spectrum noise is persistent.
    KeyAdapter windKeys(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        auto& p = ocean->params;
        float speed = p.windSpeed, theta = p.windTheta;
        if (ev.key == Key::UP)         speed = std::min(speed + 1.f, 30.f);
        else if (ev.key == Key::DOWN)  speed = std::max(speed - 1.f, 0.5f);
        else if (ev.key == Key::LEFT)  theta -= math::PI / 12.f;
        else if (ev.key == Key::RIGHT) theta += math::PI / 12.f;
        else return;
        ocean->setWind(speed, theta);
        std::cout << "wind: " << speed << " m/s, "
                  << theta * math::RAD2DEG << " deg" << std::endl;
    });
    canvas.addKeyListener(windKeys);

    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 2000.f);
    if (pond) camera.position.set(0.f, 2.4f, 8.5f);
    else      camera.position.set(0.f, 10.f, 35.f);
    if (capArgs.camPos) camera.position.copy(*capArgs.camPos);

    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    if (capArgs.camTarget) controls.target.copy(*capArgs.camTarget);
    controls.update();

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    // sampleHeight parity probes (--probes): emissive spheres pinned to the
    // CPU-mirrored wave height each frame. If the CPU sampler matches the GPU
    // displacement they sit ON the surface; any axis flip / offset shows as
    // spheres hovering or submerged. Also exercises the lazy height readback.
    std::vector<std::shared_ptr<Mesh>> probeSpheres;
    if (probes) {
        auto probeMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(1.f, 0.15f, 0.05f)).roughness(0.4f));
        probeMat->emissive = Color(1.f, 0.2f, 0.05f);
        probeMat->emissiveIntensity = 2.f;
        for (int i = 0; i < 9; ++i) {
            auto s = Mesh::create(SphereGeometry::create(pond ? 0.12f : 0.5f, 16, 12), probeMat);
            probeSpheres.push_back(s);
            scene.add(s);
        }
    }

    canvas.animate([&] {
        controls.update();

        // Pack vertex density toward where the camera is looking. The focus is
        // just a world coordinate — the same warp the showcase points at a boat.
        ocean->warpToward(controls.target.x, controls.target.z, 0.3f);

        if (wind2 >= 0.f && shotFrame == shotFrames / 2)
            ocean->setWind(wind2, ocean->params.windTheta);

        if (!probeSpheres.empty()) {
            const float spacing = pond ? 3.f : 12.f;
            int k = 0;
            for (int gz = -1; gz <= 1; ++gz)
                for (int gx = -1; gx <= 1; ++gx) {
                    const float px = controls.target.x + float(gx) * spacing;
                    const float pz = controls.target.z + float(gz) * spacing;
                    probeSpheres[k++]->position.set(
                            px, ocean->sampleHeight(px, pz), pz);
                }
        }

        renderer.render(scene, camera);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            std::filesystem::create_directories(path.parent_path());
            renderer.writeFramebuffer(path);
            std::cout << "wrote " << path.string() << std::endl;
            std::exit(0);
        }
    });

    return 0;
}
