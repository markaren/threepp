
#include "threepp/threepp.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include "Youbot.hpp"


using namespace threepp;


int main() {

    Canvas canvas{Canvas::Parameters().title("Youbot").size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas.size()};
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();

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

    TextRenderer textRenderer;
    auto& handle = textRenderer.createHandle("Loading model..");
    handle.scale = 2;

    utils::ThreadPool pool;
    std::shared_ptr<Youbot> youbot;
    pool.submit([&] {
        youbot = Youbot::create("data/models/collada/youbot.dae");
        canvas.invokeLater([&] {
            canvas.addKeyListener(youbot.get());
            scene->add(youbot);
            handle.setText("Use WASD keys to steer robot");
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        renderer.render(*scene, *camera);
        renderer.resetState();

        textRenderer.render();

        if (youbot) youbot->update(dt);
    });
}
