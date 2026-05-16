
#include "SnakeScene.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/objects/TextSprite.hpp"


int main() {

    SnakeGame game(10);

    Canvas canvas("Snake");
    int height = monitor::monitorSize().height() / 2;
    canvas.setSize({height, height});
    auto renderer = createRenderer(canvas);

    auto scene = SnakeScene(game);
    canvas.addKeyListener(scene);

    OrthographicCamera camera(
        0, static_cast<float>(game.gridSize()),
        0, static_cast<float>(game.gridSize()));
    camera.position.z = 1;

    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();

    auto handle = TextSprite::create(font, 20.f * monitor::contentScale().first);
    handle->setText("Press 'r' to reset");
    handle->setColor(Color::red);
    handle->screenSpace = true;
    handle->screenAnchor.set(0.f, 1.f);    // top-left of viewport
    handle->position.set(5.f, -5.f, 0.f);  // 5 px margin in from top-left
    scene.add(handle);


    canvas.onWindowResize([&](WindowSize size) {
        camera.right = static_cast<float>(game.gridSize()) * size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
        camera.right = game.gridSize() * size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        if (game.isRunning()) {

            game.update(dt);
            scene.update();
        }

        renderer->render(scene, camera);
    });
}
