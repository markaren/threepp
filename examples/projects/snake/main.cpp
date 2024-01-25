
#include "SnakeScene.hpp"

int main() {

    SnakeGame game(10);

    Canvas canvas("Snake");
    int height = canvas.monitorSize().height / 2;
    canvas.setSize({height, height});
    GLRenderer renderer(canvas.size());
//    renderer.setSize(canvas.size());
    renderer.autoClear = false;

    auto scene = SnakeScene(game);
    canvas.addKeyListener(scene);

    canvas.onWindowResize([&](WindowSize size) {
        scene.camera().updateProjectionMatrix();
        renderer.setSize(size);
    });

    HUD hud;
    HudText text("data/fonts/helvetiker_regular.typeface.json");
    text.setText("Press 'r' to reset");
    text.setColor(Color::red);
    text.setPosition(0, 1);
    text.setVerticalAlignment(HudText::VerticalAlignment::TOP);
    hud.addText(text);

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        if (game.isRunning()) {

            game.update(dt);
            scene.update();
        }

        renderer.clear();
        renderer.render(scene, scene.camera());
        hud.apply(renderer);
    });
}
