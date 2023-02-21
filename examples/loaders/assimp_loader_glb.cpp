
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(0, 0.1, 1);

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    AssimpLoader loader;
    auto objModel = loader.load("data/models/gltf/zedm.glb", true);
    scene->add(objModel);

    auto light = AmbientLight::create(Color(0xfffff));
    scene->add(light);

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
