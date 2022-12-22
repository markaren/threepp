
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/threepp.hpp"

#include <thread>

using namespace threepp;

struct Youbot {

    std::shared_ptr<Group> base;
    Object3D *front_left_wheel;
    Object3D *front_right_wheel;
    Object3D *back_left_wheel;
    Object3D *back_right_wheel;

    explicit Youbot(std::shared_ptr<Group> model) : base(std::move(model)) {

        back_left_wheel = base->getObjectByName("back-left_wheel");
        back_right_wheel = base->getObjectByName("back-right_wheel");
        front_left_wheel = base->getObjectByName("front-left_wheel_join");
        front_right_wheel = base->getObjectByName("front-right_wheel");
    }
};

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
        youbot = std::make_unique<Youbot>(loader.load("data/models/collada/youbot.dae"));
        youbot->base->scale.multiplyScalar(10);

        canvas.invokeLater([&] {
            scene->add(youbot->base);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    struct {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } wasd;

    canvas.addKeyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == 87) {
            wasd.up = true;
        } else if (evt.key == 83) {
            wasd.down = true;
        } else if (evt.key == 68) {
            wasd.right = true;
        } else if (evt.key == 65) {
            wasd.left = true;
        }
    });

    canvas.addKeyAdapter(KeyAdapter::Mode::KEY_RELEASED, [&](KeyEvent evt) {
        if (evt.key == 87) {
            wasd.up = false;
        } else if (evt.key == 83) {
            wasd.down = false;
        } else if (evt.key == 68) {
            wasd.right = false;
        } else if (evt.key == 65) {
            wasd.left = false;
        }
    });

    float translationSpeed = 5;
    float rotationSpeed = 2;
    canvas.animate([&](float dt) {
        renderer.render(scene, camera);

        if (youbot) {

            if (wasd.up) {
                youbot->base->translateX(translationSpeed * dt);
                youbot->back_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->back_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->front_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->front_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
            }
            if (wasd.down) {
                youbot->base->translateX(-translationSpeed * dt);
                youbot->back_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->back_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->front_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
                youbot->front_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
            }
            if (wasd.right) {
                youbot->base->rotateY(-rotationSpeed * dt);
                youbot->back_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->back_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->front_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->front_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 100 * dt);
            }
            if (wasd.left) {
                youbot->base->rotateY(rotationSpeed * dt);
                youbot->back_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->back_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->front_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 100 * dt);
                youbot->front_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 100 * dt);
            }

        }

    });

    t.join();
}
