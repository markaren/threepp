
#include "threepp/threepp.hpp"

#include "BulletEngine.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5, 5, 5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color = Color::blue;
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setY(6);
    scene->add(box);

    const auto sphereGeometry = SphereGeometry::create(0.5);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color = Color::gray;
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.set(0, 5, 0.5);
    scene->add(sphere);

    const auto planeGeometry = BoxGeometry::create(10, 0.1, 10);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color = Color::red;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->rotateZ(math::DEG2RAD*25);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


    BulletEngine engine;

    engine.register_mesh(box, 1);
    engine.register_mesh(sphere, 0.1);
    engine.register_mesh(plane, 0);

    box->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        engine.step(dt);

        renderer.render(scene, camera);
    });
}
