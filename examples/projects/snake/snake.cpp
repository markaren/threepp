
#include "SnakeScene.hpp"

int main() {

    SnakeGame game(10);

    Canvas canvas(Canvas::Parameters().title("Snake"));
    GLRenderer renderer(canvas);

    auto scene = std::make_shared<SnakeScene>(game);
    canvas.addKeyListener(scene.get());

    canvas.onWindowResize([&](WindowSize size) {
        scene->camera().updateProjectionMatrix();
        renderer.setSize(size);
    });

    renderer.enableTextRendering();
    auto& handle = renderer.textHandle("Press \"r\" to reset");

    canvas.animate([&](float dt) {
        if (game.isRunning()) {

            game.update(dt);
            scene->update();
        }
        renderer.render(scene.get(), &scene->camera());
    });
}
