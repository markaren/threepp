
#include <threepp/threepp.hpp>

using namespace threepp;

const int frustumSize = 600;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.autoClear = false;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(50, 0.5f * canvas.getAspect(), 1, 10);

    auto sphereGeometry = SphereGeometry::create();
    auto sphereMaterial = MeshBasicMaterial::create();
    auto sphereMesh = Mesh::create(sphereGeometry, sphereMaterial);
    sphereMesh->position.z = -10;
    scene->add(sphereMesh);

    auto camera2 = PerspectiveCamera::create(50, 0.5f * canvas.getAspect(), 1, 1000);
    camera2->position.z = 5;
    camera2->position.x = 10;

    OrbitControls controls{camera2, canvas};
    controls.target = sphereMesh->position;
    controls.update();

    auto helper = CameraHelper::create(camera);
    scene->add(helper);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = 0.5f* size.getAspect();
        camera->updateProjectionMatrix();
        camera2->aspect = 0.5f* size.getAspect();
        camera2->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        auto size = canvas.getSize();

        renderer.clear();

        helper->visible = false;

        renderer.setViewport({size.width / 2, 0, size.width / 2, size.height});
        renderer.render(scene, camera);

        helper->visible = true;

        renderer.setViewport({0, 0, size.width / 2, size.height});
        renderer.render(scene, camera2);

        camera->position.z -= dt;

    });
}