// trap/main.cpp: the first try at an object-oriented Snake. This is the
// inheritance trap in "A threepp lesson, Part 0c: threepp is the view".
//
// It looks tidy: a Snake class, an Apple class, a shorter main(). But Snake *is a*
// threepp::Group, so the rules still live inside a scene node:
//   - the snake's cells are float mesh positions, compared with distanceTo;
//   - everything a Group has is public: anyone can move, scale or empty the snake,
//     and then the snake you see and the snake the rules check drift apart;
//   - testing a rule means linking a 3D engine, drawing the game another way (text,
//     a minimap) means reading meshes, and changing the look means editing the rules.
// A snake isn't a scene node, it has a look. ../model is the way out.

#include "threepp/threepp.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace threepp;

class Snake: public Group {
public:
    Snake() {
        material_->color = 0x7bd389;
        for (int i = 0; i < 3; i++) addSegment({0.5f - i, 0.45f, 0.5f});
    }

    bool alive() const { return alive_; }

    // no turning back onto the neck
    void turn(const Vector3& direction) {
        if (direction.dot(direction_) > -0.5f) direction_ = direction;
    }

    Vector3 next() const {
        return segments_.front()->position + direction_;
    }

    bool occupies(const Vector3& cell) const {
        for (auto& segment : segments_) {
            if (segment->position.distanceTo(cell) < 0.5f) return true;
        }
        return false;
    }

    void advance(bool grow) {
        const Vector3 next = this->next();
        if (grow) addSegment(segments_.back()->position);
        for (auto i = segments_.size() - 1; i > 0; i--) {
            segments_[i]->position.copy(segments_[i - 1]->position);
        }
        segments_.front()->position.copy(next);
    }

    void die() {
        alive_ = false;
        material_->color = 0x6c757d;
    }

private:
    bool alive_ = true;
    Vector3 direction_{1, 0, 0};
    std::shared_ptr<BoxGeometry> geometry_ = BoxGeometry::create(0.9f, 0.9f, 0.9f);
    std::shared_ptr<MeshStandardMaterial> material_ = MeshStandardMaterial::create();
    std::vector<std::shared_ptr<Mesh>> segments_;

    void addSegment(const Vector3& position) {
        auto segment = Mesh::create(geometry_, material_);
        segment->position.copy(position);
        add(segment);
        segments_.push_back(segment);
    }
};

class Apple: public Mesh {
public:
    Apple(): Mesh(SphereGeometry::create(0.4f), red()) {
        position.set(5.5f, 0.45f, -3.5f);
    }

    void respawn(std::mt19937& rng) {
        position.x = static_cast<float>(rng() % 20) - 9.5f;
        position.z = static_cast<float>(rng() % 20) - 9.5f;
    }

private:
    static std::shared_ptr<MeshStandardMaterial> red() {
        auto material = MeshStandardMaterial::create();
        material->color = 0xe63946;
        return material;
    }
};

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

    auto snake = std::make_shared<Snake>();
    scene.add(snake);
    auto apple = std::make_shared<Apple>();
    scene.add(apple);

    std::mt19937 rng(7);

    KeyAdapter keys(KeyAdapter::KEY_PRESSED, [&](KeyEvent e) {
        if (e.key == Key::UP) snake->turn({0, 0, -1});
        if (e.key == Key::DOWN) snake->turn({0, 0, 1});
        if (e.key == Key::LEFT) snake->turn({-1, 0, 0});
        if (e.key == Key::RIGHT) snake->turn({1, 0, 0});
    });
    canvas.addKeyListener(keys);

    int score = 0;
    float timer = 0;

    Clock clock;
    canvas.animate([&] {
        timer += clock.getDelta();

        if (snake->alive() && timer > 0.125f) {
            timer = 0;

            const Vector3 next = snake->next();
            if (std::abs(next.x) > 10 || std::abs(next.z) > 10 || snake->occupies(next)) {
                snake->die();
            } else {
                const bool eats = next.distanceTo(apple->position) < 0.5f;
                snake->advance(eats);
                if (eats) {
                    score++;
                    std::cout << "score " << score << std::endl;
                    apple->respawn(rng);
                }
            }
        }

        renderer.render(scene, camera);
    });
}
