
#include "threepp/threepp.hpp"

#include "Youbot.hpp"

#include <future>

using namespace threepp;


int main() {

    Canvas canvas{Canvas::Parameters().title("Youbot").size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas.size()};
    renderer.autoClear = false;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(*camera, canvas);

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto light1 = DirectionalLight::create(0xffffff, 1.f);
    light1->position.set(1, 1, 1);
    scene->add(light1);

    auto light2 = AmbientLight::create(0xffffff, 1.f);
    scene->add(light2);

    HUD hud(canvas.size());
    FontLoader fontLoader;
    const auto font = *fontLoader.load("data/fonts/helvetiker_regular.typeface.json");

    TextGeometry::Options opts(font, 20, 5);
    auto handle = Text2D(opts, "Loading model..");
    handle.setColor(Color::black);
    hud.add(handle, HUD::Options()
                            .setNormalizedPosition({0, 1})
                            .setVerticalAlignment(HUD::VerticalAlignment::TOP));


    std::shared_ptr<Youbot> youbot;
    auto future = std::async([&] {
        youbot = Youbot::create("data/models/collada/youbot.dae");
        renderer.invokeLater([&] {
            canvas.addKeyListener(*youbot);
            scene->add(youbot);
            handle.setText("Use WASD keys to steer robot", opts);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        renderer.clear();
        renderer.render(*scene, *camera);
        hud.apply(renderer);

        if (youbot) youbot->update(dt);
    });
}
