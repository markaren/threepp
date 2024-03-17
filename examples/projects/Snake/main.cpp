
#include "SnakeScene.hpp"


int main() {

    SnakeGame game(10);

    Canvas canvas("Snake");
    int height = canvas.monitorSize().height / 2;
    canvas.setSize({height, height});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;

    auto scene = SnakeScene(game);
    canvas.addKeyListener(scene);

    auto camera = OrthographicCamera::create(0, game.gridSize(), 0, game.gridSize());
    camera->position.z = 1;

    HUD hud(canvas.size());
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();

    TextGeometry::Options opts(font, 15, 5);
    auto handle = Text2D(opts, "Press 'r' to reset");
    handle.setColor(Color::red);
    hud.add(handle, HUD::Options()
                            .setNormalizedPosition({0, 1})
                            .setVerticalAlignment(HUD::VerticalAlignment::TOP));

    canvas.onWindowResize([&](WindowSize size) {
        camera->right = game.gridSize() * size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);

        hud.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        if (game.isRunning()) {

            game.update(dt);
            scene.update();
        }

        renderer.clear();
        renderer.render(scene, *camera);
        hud.apply(renderer);
    });
}
