
#include "SnakeScene.hpp"


class MyKeyListener : public KeyListener {

public:
    explicit MyKeyListener(SnakeScene &scene, SnakeGame &state)
        : scene_(scene),
          game_(state) {}

    void onKeyPressed(KeyEvent evt) override {

        if (game_.isRunning()) {

            if (evt.key == 265 && game_.direction != Direction::DOWN) {
                game_.nextDirection = Direction::UP;
            }
            if (evt.key == 264 && game_.direction != Direction::UP) {
                game_.nextDirection = Direction::DOWN;
            }
            if (evt.key == 263 && game_.direction != Direction::RIGHT) {
                game_.nextDirection = Direction::LEFT;
            }
            if (evt.key == 262 && game_.direction != Direction::LEFT) {
                game_.nextDirection = Direction::RIGHT;
            }

        } else if (evt.key == 82 /*r*/) {

            game_.reset();
            scene_.reset();
        }
    }

private:
    SnakeScene &scene_;
    SnakeGame &game_;
};

int main() {

    SnakeGame game(10);

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = std::make_shared<SnakeScene>(game);
    canvas.addKeyListener(std::make_shared<MyKeyListener>(*scene, game));

    canvas.onWindowResize([&](WindowSize size) {
        scene->camera().updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        if (game.isRunning()) {

            game.update(dt);
            scene->update();

        }
        renderer.render(scene.get(), &scene->camera());
    });
}
