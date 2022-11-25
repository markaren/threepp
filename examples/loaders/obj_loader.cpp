
#include <threepp/loaders/OBJLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 1;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);
    renderer.setSize(canvas.getSize());

    OrbitControls controls{camera, canvas};

    OBJLoader loader;
    auto obj = loader.load("data/models/obj/female02/female02.obj");
    scene->add(obj);

    canvas.animate([&](float dt) {

        renderer.render(scene, camera);
    });
}
