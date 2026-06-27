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
// Headless capture: vulkan_ocean_minimal --shot <name.png> [--frames N] [--pt]
//   (plus --cam x,y,z / --look x,y,z to reframe with no rebuild).

#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include "capture_util.hpp"

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
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
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

    constexpr float kSize = 1000.0f;

    // Dark sand floor under the water so refraction and caustics read. Matches
    // the ocean extent so there's no visible sand frame around the tile.
    auto floorMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color(0.02f, 0.02f, 0.02f)).roughness(1.0f));
    auto floor = Mesh::create(PlaneGeometry::create(kSize, kSize), floorMat);
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -5.f;
    scene.add(floor);

    // The whole "fancy water" in one line.
    Ocean::Options opts;
    opts.size = kSize;
    auto ocean = Ocean::create(opts);
    scene.add(ocean);

    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 2000.f);
    camera.position.set(0.f, 10.f, 35.f);
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

    canvas.animate([&] {
        controls.update();

        // Pack vertex density toward where the camera is looking. The focus is
        // just a world coordinate — the same warp the showcase points at a boat.
        ocean->warpToward(controls.target.x, controls.target.z, 0.3f);

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
