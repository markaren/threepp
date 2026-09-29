// main.cpp: Snake with threepp as the view. This is the "after" picture in
// "A threepp lesson, Part 0c: threepp is the view".
//
// The rules live in model/ (which can't see threepp), the looks in view/ and the
// players in control/; main() only wires them together. Like Part 0b's app.cpp it
// grows in steps marked `#if STEP >= n` (in CMake: -DLESSON_SNAKE_STEP=n). The
// default is the whole app:
//
//   1  a model and a view            the keys steer the game; the view snaps to each tick
//   2  two clocks                    the view slides between ticks; the rules don't change
//   3  players are an interface      KeyboardController or GreedyAI, TAB swaps them
//   4  events                        the view pops a ring when the snake eats
//   5  more views of one game        a minimap, and the board as text with --ascii
//   6  a second game                 a rival AI on its own board: just more objects

#ifndef STEP
#define STEP 99
#endif

#include "model/Game.hpp"
#include "view/SnakeView.hpp"
#if STEP >= 3
#include "control/GreedyAI.hpp"
#include "control/KeyboardController.hpp"
#endif
#if STEP >= 5
#include "view/AsciiView.hpp"
#include "view/MinimapView.hpp"
#endif

#include "threepp/threepp.hpp"

#include <iostream>
#include <string>

using namespace snake;
using namespace threepp;

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    Canvas canvas("snake", {{"size", WindowSize{1280, 720}}, {"antialiasing", 4}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1b1f24));
    renderer.shadowMap().enabled = true;

    Scene scene;

    PerspectiveCamera camera(50, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 20, 10);
    camera.lookAt(0, 0, 1.5f);
    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    scene.add(HemisphereLight::create(0xdfe8ff, 0x2d3a2e, 1.5f));
    auto light = DirectionalLight::create(0xffffff, 2.f);
    light->position.set(6, 14, 8);
    light->castShadow = true;
    auto* shadowCamera = light->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -24;
    shadowCamera->right = shadowCamera->top = 24;
    shadowCamera->updateProjectionMatrix();
    light->shadow->mapSize.set(2048, 2048);
    scene.add(light);

    Game game(20, 20, 7);
    SnakeView view(game);
    scene.add(view.object());

#if STEP >= 3
    KeyboardController keyboard;
    GreedyAI ai;
    canvas.addKeyListener(keyboard);

    Controller* player = &keyboard;// who plays; borrowed, not owned
    KeyAdapter swap(KeyAdapter::KEY_PRESSED, [&](KeyEvent e) {
        if (e.key != Key::TAB) return;
        if (player == &keyboard) player = &ai;
        else player = &keyboard;
    });
    canvas.addKeyListener(swap);
#else
    KeyAdapter keys(KeyAdapter::KEY_PRESSED, [&](KeyEvent e) {
        if (e.key == Key::UP) game.steer(Direction::Up);
        if (e.key == Key::DOWN) game.steer(Direction::Down);
        if (e.key == Key::LEFT) game.steer(Direction::Left);
        if (e.key == Key::RIGHT) game.steer(Direction::Right);
    });
    canvas.addKeyListener(keys);
#endif

#if STEP >= 4
    game.onEat([&](const Game& g) { view.pop(g.snake().head()); });
    game.onDie([](const Game& g) {
        std::cout << "game over, score " << g.score() << std::endl;
    });
#endif

#if STEP >= 5
    MinimapView minimap(game);
    AsciiView ascii(std::cout);
    const bool showAscii = argc > 1 && std::string(argv[1]) == "--ascii";
#endif

#if STEP >= 6
    Game rival(20, 20, 8);
    SnakeView rivalView(rival);
    GreedyAI rivalAI;
    scene.add(rivalView.object());
    rival.onEat([&](const Game& g) { rivalView.pop(g.snake().head()); });

    view.object()->position.x = -11;
    rivalView.object()->position.x = 11;
    camera.position.set(0, 28, 15);
    camera.lookAt(0, 0, -3.5f);
#endif

    const float tickTime = 0.2f;// five moves a second
    float sinceTick = 0;

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        sinceTick += dt;
        while (sinceTick >= tickTime) {
            sinceTick -= tickTime;
#if STEP >= 3
            game.steer(player->next(game));
#endif
            game.tick();
#if STEP >= 6
            rival.steer(rivalAI.next(rival));
            rival.tick();
#endif
        }

#if STEP >= 2
        const float alpha = sinceTick / tickTime;
#else
        const float alpha = 1;
#endif
        view.update(game, alpha, dt);
#if STEP >= 6
        rivalView.update(rival, alpha, dt);
#endif
#if STEP >= 5
        minimap.update(game);
        if (showAscii) ascii.update(game);
#endif

        renderer.render(scene, camera);
#if STEP >= 5
        minimap.render(renderer, canvas.size());
#endif
    });
}
