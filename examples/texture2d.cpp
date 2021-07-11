
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getSize());

    TextureLoader loader{};

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->map = loader.loadTexture("textures/checker.png");
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(1);
    scene->add(sphere);

    const auto sphereGeometry1 = SphereGeometry::create(0.5f);
    const auto sphereMaterial1 = MeshBasicMaterial::create();
    sphereMaterial1->map = loader.loadTexture("textures/brick_bump.jpg");
    auto sphere1 = Mesh::create(sphereGeometry1, sphereMaterial1);
    sphere1->position.setX(2);
    scene->add(sphere1);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->map = loader.loadTexture("textures/crate.gif");
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setX(-1);
    scene->add(box);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    box->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        box->rotation.y(box->rotation.y() + 0.5f * dt);
        scene->rotation.x(scene->rotation.x() + 1.f * dt);

        renderer.render(scene, camera);
    });
}
