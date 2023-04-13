
#include "threepp/helpers/CameraHelper.hpp"
#include <threepp/threepp.hpp>

#include <cmath>

using namespace threepp;

namespace {

    auto createSphere() {
        auto sphereGeometry = SphereGeometry::create(1, 10, 10);
        auto sphereMaterial = MeshBasicMaterial::create();
        auto sphereMesh = Mesh::create(sphereGeometry, sphereMaterial);
        sphereMesh->position.z = -8;

        auto sphereMaterialWireframe = MeshBasicMaterial::create({{"color", Color::black}, {"wireframe", true}});
        sphereMaterialWireframe->wireframe = true;
        sphereMaterialWireframe->color = Color::black;
        auto sphereMeshWireframe = Mesh::create(sphereGeometry, sphereMaterialWireframe);
        sphereMesh->add(sphereMeshWireframe);

        return sphereMesh;
    }

}// namespace

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer(canvas);
    renderer.autoClear = false;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, 0.5f * canvas.getAspect(), 1, 10);

    auto sphere = createSphere();
    scene->add(sphere);

    auto camera2 = PerspectiveCamera::create(50, 0.5f * canvas.getAspect(), 1, 1000);
    camera2->position.x = 30;

    OrbitControls controls{camera2, canvas};
    controls.target = sphere->position;
    controls.update();

    auto helper = CameraHelper::create(*camera);
    scene->add(helper);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = 0.5f * size.getAspect();
        camera->updateProjectionMatrix();
        camera2->aspect = 0.5f * size.getAspect();
        camera2->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        auto size = canvas.getSize();

        renderer.clear();

        helper->visible = false;

        renderer.setViewport({size.width / 2, 0, size.width / 2, size.height});
        renderer.render(scene, camera);

        helper->visible = true;

        renderer.setViewport({0, 0, size.width / 2, size.height});
        renderer.render(scene, camera2);

        camera->position.z = 4 * std::sin(math::TWO_PI * 0.1f * t);
    });
}
