
#include "SnakeScene.hpp"

#include "threepp/canvas/Monitor.hpp"


int main() {

    SnakeGame game(10);

    Canvas canvas("Snake");
    int height = monitor::monitorSize().height() / 2;
    canvas.setSize({height, height});
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;

    auto scene = SnakeScene(game);
    canvas.addKeyListener(scene);

    OrthographicCamera camera(
        0, static_cast<float>(game.gridSize()),
        0, static_cast<float>(game.gridSize()));
    camera.position.z = 1;

    HUD hud(renderer);
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();

    TextGeometry::Options opts(font, 15, 5);
    auto handle = Text2D(opts, "Press 'r' to reset");
    handle.setColor(Color::red);
    hud.add(handle).setNormalizedPosition({0, 1})
                            .setVerticalAlignment(HUD::VerticalAlignment::BELOW);


    canvas.onWindowResize([&](WindowSize size) {
        camera.right = static_cast<float>(game.gridSize()) * size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        if (game.isRunning()) {

            game.update(dt);
            scene.update();
        }

        renderer.clear();
        renderer.render(scene, camera);
        hud.render();
    });
}
