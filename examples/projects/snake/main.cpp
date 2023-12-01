
#include "SnakeScene.hpp"

int main() {

    SnakeGame game(10);

    Canvas canvas("Snake");
    int height = canvas.monitorSize().height / 2;
    canvas.setSize({height, height});
    GLRenderer renderer(canvas.size());

    auto scene = SnakeScene(game);
    canvas.addKeyListener(&scene);

    canvas.onWindowResize([&](WindowSize size) {
        scene.camera().updateProjectionMatrix();
        renderer.setSize(size);
    });

    TextRenderer textRenderer;
    auto& handle = textRenderer.createHandle("Press \"r\" to reset");

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        if (game.isRunning()) {

            game.update(dt);
            scene.update();
        }
        renderer.render(scene, scene.camera());
        renderer.resetState();

        textRenderer.render();
    });
}
