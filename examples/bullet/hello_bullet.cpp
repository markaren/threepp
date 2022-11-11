
#include "threepp/threepp.hpp"

#include "BulletEngine.hpp"

using namespace threepp;

int main() {


    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5,5,5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getSize());

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1,0,0);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setY(5);
    scene->add(box);

    const auto planeGeometry = BoxGeometry::create(1, 1, 1);
//    planeGeometry->rotateX(math::degToRad(-90));
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setRGB(0,0,1);
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    scene->add(plane);


    canvas.onWindowResize([&](WindowSize size){
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    BulletEngine engine;

    engine.register_mesh(box, 1);
    engine.register_mesh(plane, 0);

    box->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {

        engine.step(dt);

        renderer.render(scene, camera);
    });
}