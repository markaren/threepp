
#include <threepp/loaders/OBJLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 100, 175);

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);
    renderer.setSize(canvas.getSize());

    OBJLoader loader;
    auto obj = loader.load("data/models/obj/female02/female02.obj");

    scene->add(obj);

    auto light = AmbientLight::create(0xffffff);
   // scene->add(light);

    canvas.animate([&](float dt) {

        obj->rotation.y += 1 * dt;

        renderer.render(scene, camera);
    });
}
