
#include <threepp/threepp.hpp>
#include <threepp/loaders/AssimpLoader.hpp>

using namespace threepp;

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 100, 175);

    AssimpLoader loader;
    auto objModel = loader.load("data/models/obj/female02/female02.obj");
    scene->add(objModel);

    auto light1 = PointLight::create(Color(0xffffff), 1.f);
    light1->position.set(25, 115, 25);
    scene->add(light1);

    auto light2 = PointLight::create(Color(0xffffff), 1.f);
    light2->position.set(-25, 115, 125);
    scene->add(light2);

    auto light3 = PointLight::create(Color(0xffffff), 1.f);
    light3->position.set(0, 25, -30);
    scene->add(light3);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {

        objModel->rotation.y += 1 * dt;

        renderer.render(scene, camera);
    });
}
