
#include "threepp/threepp.hpp"

#include "Youbot.hpp"

#include <thread>

using namespace threepp;


int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto light = DirectionalLight::create(0xffffff, 0.1f);
    scene->position.set(1, 1, 1);
    scene->add(light);

    auto light2 = AmbientLight::create(0xffffff, 0.1f);
    scene->add(light2);

    std::unique_ptr<Youbot> youbot;
    std::thread t([&] {
        AssimpLoader loader;
        youbot = Youbot::create("data/models/collada/youbot.dae");
        youbot->setup(canvas);

        canvas.invokeLater([&] {
            scene->add(youbot->base);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        renderer.render(scene, camera);

        if (youbot) {
            youbot->update(dt);
        }
    });

    t.join();
}
