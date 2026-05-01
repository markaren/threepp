// Vulkan renderer smoke test (Phase 4b).
//
// Walks a real threepp Scene, builds one BLAS per BufferGeometry, and a TLAS
// with one instance per Mesh. The closest-hit shader still returns barycentric
// RGB so each Mesh shows the standard tri-color gradient — Phase 5 will swap
// in real shading. Rays come from a PerspectiveCamera UBO (Phase 4a), so
// OrbitControls confirm world-space anchoring and per-instance transforms.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iostream>

using namespace threepp;

int main() {

    Canvas::Parameters params;
    params.title("Vulkan smoke (Phase 4b: scene ingestion)")
            .size(800, 600);

    Canvas canvas(params);

    VulkanRenderer renderer(canvas);

    auto scene = Scene::create();

    // One geometry, two instances at different world positions: confirms
    // both BLAS reuse (cache hit) and the per-instance transform path.
    // Distinct materials exercise the binding-4 MaterialDesc path: a rough
    // dielectric red vs a smooth blue metal makes the GGX/Schlick contrast
    // obvious vs the Phase 5a Lambert-only output.
    auto boxGeom = BoxGeometry::create(1.0f, 1.0f, 1.0f);

    auto matA = MeshStandardMaterial::create();
    matA->color.setHex(0xcc3333);
    matA->roughness = 0.85f;
    matA->metalness = 0.0f;
    auto boxA = Mesh::create(boxGeom, matA);
    boxA->position.set(-1.0f, 0.0f, 0.0f);
    scene->add(boxA);

    auto matB = MeshStandardMaterial::create();
    matB->color.setHex(0x3366cc);
    matB->roughness = 0.2f;
    matB->metalness = 1.0f;
    auto boxB = Mesh::create(boxGeom, matB);
    boxB->position.set(1.0f, 0.0f, 0.0f);
    boxB->rotation.set(0.4f, 0.8f, 0.0f);
    scene->add(boxB);

    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 1.5f, 4.f);
    camera->lookAt(0, 0, 0);

    OrbitControls controls{*camera, canvas};

    canvas.onWindowResize([&](const WindowSize&) {
        camera->aspect = canvas.aspect();
        camera->updateProjectionMatrix();
    });

    std::cout << "[vulkan_smoke] expecting rough red box (left) + smooth blue metal (right) under a fixed sun" << std::endl;

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
