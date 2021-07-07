
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getWidth(), canvas.getHeight());

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setHex(0xff0000);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setX(1);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create();
    const auto sphereMaterial = MeshBasicMaterial::create();
//    sphereMaterial->color.setHex(0x00ff00);
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(-1);
    scene->add(sphere);

    camera->position.z = 5;

    canvas.animate([&](float dt) {
        box->rotation.x(box->rotation.x() + 0.01f);
        box->rotation.y(box->rotation.y() + 0.01f);

        renderer.render(scene, camera);
    });
}
