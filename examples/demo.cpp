
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1,0,0);
    boxMaterial->transparent = true;
    boxMaterial->opacity = 0.1f;
    auto box = Mesh::create(boxGeometry, boxMaterial);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x00ff00);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(-1);
    box->add(sphere);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setHex(Color::yellow);
    planeMaterial->transparent = true;
    planeMaterial->opacity = 0.5f;
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setZ(-2);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size){
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    box->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        box->rotation.y +=  0.5f * dt;
        scene->rotation.x += 1.f * dt;

        renderer.render(scene, camera);
    });
}
