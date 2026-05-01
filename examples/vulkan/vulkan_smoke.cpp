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
    auto boxGeom = BoxGeometry::create(1.0f, 1.0f, 1.0f);
    auto boxA = Mesh::create(boxGeom);
    boxA->position.set(-1.0f, 0.0f, 0.0f);
    scene->add(boxA);
    auto boxB = Mesh::create(boxGeom);
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

    std::cout << "[vulkan_smoke] expecting two RGB-bary boxes, draggable via mouse" << std::endl;

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
