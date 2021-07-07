
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

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(0,1,1);
    boxMaterial->transparent = true;
    boxMaterial->opacity = 0.5f;
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setX(1);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create();
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x00ff00);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(-1);
    scene->add(sphere);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setHex(Color::lightgray);
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setZ(-2);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size){
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        box->rotation.x(box->rotation.x() + 1.f * dt);
        box->rotation.y(box->rotation.y() + 0.5f * dt);

        sphere->rotation.x(sphere->rotation.x() + 1.f * dt);

        renderer.render(scene, camera);
    });
}
