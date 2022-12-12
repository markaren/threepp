
#include "threepp/threepp.hpp"
#include <future>

using namespace threepp;

std::shared_ptr<Group> loadCrane() {

    OBJLoader loader;
    auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
    auto part2 = loader.load("data/models/obj/Crane3R/7000/7000.obj");
    part2->position.set(0, 0, 4.2);
    auto part3 = loader.load("data/models/obj/Crane3R/5200/5200.obj");
    part3->position.set(7, 0, 4.2);

    auto crane = Group::create();
    crane->add(part1);
    crane->add(part2);
    crane->add(part3);

    crane->rotateX(-math::PI/2);

    return crane;
}


int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create();
    camera->position.set(10, 0, 10);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    std::thread t([&]{
        auto crane = loadCrane();
        canvas.invokeLater([&, crane]{
            scene->add(crane);
        });
    });

    canvas.onWindowResize([&](WindowSize size){
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });

    t.join();


}