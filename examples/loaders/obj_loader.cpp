
#include <threepp/loaders/OBJLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 100, 175);

    GLRenderer renderer(canvas);

    OBJLoader loader;
    auto obj = loader.load("data/models/obj/female02/female02.obj");

    scene->add(obj);

    auto light1 = PointLight::create(Color(0xffffff).multiplyScalar(0.5f), 1.f);
    light1->position.set(25, 115, 25);
    scene->add(light1);

    auto light2 = PointLight::create(Color(0xffffff).multiplyScalar(0.5f), 1.f);
    light1->position.set(-25, 115, 125);
    scene->add(light2);

    canvas.animate([&](float dt) {

        obj->rotation.y += 1 * dt;

        renderer.render(scene, camera);
    });
}
