
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/loaders/FBXLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <filesystem>

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
                  {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::Neutral;
    renderer.setFireflyClamp(8.f);
    renderer.setRenderScale(0.75);
    // renderer.setGbufferMsaa(2);
    renderer.setOcclusionCulling(true);


    // ---- Scene ----
    Scene scene;

    RGBELoader imgLoader;
    if (auto env = imgLoader.load(modelFolder / "san_giuseppe_bridge_4k.hdr")) {
        scene.background = env;
        scene.environment = env;
    }

    // ---- Camera ----
    PerspectiveCamera camera(60.f, canvas.aspect(), 0.01f, 1000.f);// 100k far/near is fine now: reversed-Z raster (was z-fighting before; near was bumped to 0.1 as a workaround)
    camera.position.set(-10.f, 3.f, -5.f);
    OrbitControls controls{camera, canvas};
    controls.enableKeys = false;
    controls.update();

    // ---- Async model loading ----
    FBXLoader loader;
    // Bistro emissives (lamps, signs, string lights) are authored at low factors;
    // the official Falcor scene multiplies every emissive factor by 1000 to get a
    // well-exposed image, so do the same here for interior lighting to read.
    loader.emissiveScale = 100.0f;
    auto interior = loadAsync(loader, modelFolder / "BistroInterior_Wine.fbx");
    auto exterior = loadAsync(loader, modelFolder / "BistroExterior.fbx");

    scene.add(interior);
    scene.add(exterior);

    auto toggleBistroLights = [&] {
        scene.traverseType<Light>([&](Light& l) {
            l.visible = !l.visible;
        });
    };

    // ---- UI ----
    // Generic renderer settings (exposure, tone map, upscaler, GI, ...) come
    // from the shared panel; only the scene-specific button is added here.
    RendererSettingsUi ui(canvas, renderer, [&] {
        if (ImGui::Button("Toggle bistro lights")) {
            toggleBistroLights();
        }
    }, "Bistro scene");

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    canvas.animate([&] {
        controls.update();

        renderer.render(scene, camera);

        ui.render();
    });
}
