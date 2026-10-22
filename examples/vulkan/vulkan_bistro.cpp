
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <unordered_set>

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
    // The .gltf files come from the Bistro FBXs through scripts/fbx_to_gltf.py
    // (written next to them, so their DDS textures resolve):
    //   blender --background --python scripts/fbx_to_gltf.py -- BistroExterior.fbx BistroExterior.gltf
    //   blender --background --python scripts/fbx_to_gltf.py -- BistroInterior_Wine.fbx BistroInterior_Wine.gltf
    ModelLoader loader;
    auto interior = loader.loadAsync(modelFolder / "BistroInterior_Wine.gltf");
    auto exterior = loader.loadAsync(modelFolder / "BistroExterior.gltf");

    scene.add(interior);
    scene.add(exterior);

    // Two things glTF cannot say about these materials, applied once per
    // material as each half arrives:
    //  - Bistro's normal maps are DirectX convention (green down).
    //  - Its emissives (lamps, signs, string lights) are authored at low
    //    factors; the official Falcor scene multiplies every one by 1000 to
    //    get a well-exposed image, so scale them here for the interior to read.
    auto fixMaterials = [](Object3D& part) {
        std::unordered_set<Material*> seen;
        part.traverseType<Mesh>([&](Mesh& mesh) {
            for (const auto& m : mesh.materials()) {
                auto* pbr = dynamic_cast<MeshStandardMaterial*>(m.get());
                if (!pbr || !seen.insert(pbr).second) continue;
                if (pbr->normalMap) pbr->normalScale.y = -pbr->normalScale.y;
                pbr->emissiveIntensity *= 100.f;
            }
        });
    };

    // Narrow the static vertex attributes (normal snorm16 / uv unorm16 /
    // color unorm8) as soon as each half arrives — onLoaded fires on the main
    // thread before the new children are uploaded, so the renderer only ever
    // sees the compressed arrays. Bistro is all static geometry; skinned or
    // deforming meshes would be skipped automatically.
    for (auto& part : {interior, exterior}) {
        part->onLoaded([fixMaterials](AsyncGroup& grp) {
            fixMaterials(grp);
            const size_t saved = compressSceneAttributes(grp);
            std::cout << "[bistro] compressed vertex attributes: "
                      << saved / (1024.0 * 1024.0) << " MiB reclaimed" << std::endl;
        });
    }

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
