// tangle/main.cpp: Snake the way it's usually first written. This is the "before"
// picture in "A threepp lesson, Part 0c: threepp is the view".
//
// It plays. But everything lives in main(), and the game's truth is the meshes:
// the rules compare float positions, a key press changes the direction at once,
// and drawing, timing and rules share one lambda. Two bugs hide in here, and the
// model's tests (../tests/test_game.cpp) pin both down:
//   - two quick key presses between moves can turn the snake back onto itself;
//   - a new apple can land on the snake.
// Neither can be tested here without opening the whole window.

#include "threepp/threepp.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace threepp;

int main() {
    Canvas canvas("snake", {{"size", WindowSize{1280, 720}}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1b1f24));

    Scene scene;

    PerspectiveCamera camera(50, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 20, 10);
    camera.lookAt(0, 0, 1.5f);

    scene.add(AmbientLight::create(0xffffff, 0.6f));
    auto light = DirectionalLight::create(0xffffff, 2.5f);
    light->position.set(5, 10, 7);
    scene.add(light);

    // the board: 20 x 20 cells of one unit, from -10 to 10
    auto boardMaterial = MeshStandardMaterial::create();
    boardMaterial->color = 0x2d3a2e;
    auto board = Mesh::create(PlaneGeometry::create(20, 20), boardMaterial);
    board->rotation.x = -math::PI / 2;
    scene.add(board);

    // the snake is its meshes; snake[0] is the head
    auto segmentGeometry = BoxGeometry::create(0.9f, 0.9f, 0.9f);
    auto snakeMaterial = MeshStandardMaterial::create();
    snakeMaterial->color = 0x7bd389;
    std::vector<std::shared_ptr<Mesh>> snake;
    for (int i = 0; i < 3; i++) {
        auto segment = Mesh::create(segmentGeometry, snakeMaterial);
        segment->position.set(0.5f - i, 0.45f, 0.5f);
        scene.add(segment);
        snake.push_back(segment);
    }

    auto appleMaterial = MeshStandardMaterial::create();
    appleMaterial->color = 0xe63946;
    auto apple = Mesh::create(SphereGeometry::create(0.4f), appleMaterial);
    apple->position.set(5.5f, 0.45f, -3.5f);
    scene.add(apple);

    std::mt19937 rng(7);
    Vector3 direction(1, 0, 0);

    KeyAdapter keys(KeyAdapter::KEY_PRESSED, [&](KeyEvent e) {
        if (e.key == Key::UP && direction.z != 1) direction.set(0, 0, -1);
        if (e.key == Key::DOWN && direction.z != -1) direction.set(0, 0, 1);
        if (e.key == Key::LEFT && direction.x != 1) direction.set(-1, 0, 0);
        if (e.key == Key::RIGHT && direction.x != -1) direction.set(1, 0, 0);
    });
    canvas.addKeyListener(keys);

    int score = 0;
    bool dead = false;
    float timer = 0;

    Clock clock;
    canvas.animate([&] {
        timer += clock.getDelta();

        if (!dead && timer > 0.125f) {
            timer = 0;

            Vector3 next = snake[0]->position + direction;
            if (std::abs(next.x) > 10 || std::abs(next.z) > 10) dead = true;
            for (auto& segment : snake) {
                if (segment->position.distanceTo(next) < 0.5f) dead = true;
            }

            if (!dead) {
                if (next.distanceTo(apple->position) < 0.5f) {
                    auto segment = Mesh::create(segmentGeometry, snakeMaterial);
                    segment->position.copy(snake.back()->position);
                    scene.add(segment);
                    snake.push_back(segment);

                    score++;
                    std::cout << "score " << score << std::endl;

                    apple->position.x = static_cast<float>(rng() % 20) - 9.5f;
                    apple->position.z = static_cast<float>(rng() % 20) - 9.5f;
                }

                for (auto i = snake.size() - 1; i > 0; i--) {
                    snake[i]->position.copy(snake[i - 1]->position);
                }
                snake[0]->position.copy(next);
            } else {
                snakeMaterial->color = 0x6c757d;
            }
        }

        renderer.render(scene, camera);
    });
}
