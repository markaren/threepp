
#include "threepp/threepp.hpp"
#include "threepp/loaders/AssimpLoader.hpp"

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

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    std::shared_ptr<Group> youbot;
    std::thread t([&] {
        AssimpLoader loader;
        youbot = loader.load(R"(C:\Users\Lars Ivar Hatledal\Downloads\youbot.dae)");
        youbot->scale.multiplyScalar(10);

        canvas.invokeLater([&, youbot] {
            scene->add(youbot);
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
            auto j1 = youbot->getObjectByName("arm_joint_1");
            if (j1) {
                j1->rotateZ(1*dt);
            }
        }

    });

    t.join();
}
