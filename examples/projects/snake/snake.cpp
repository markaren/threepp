
#include "threepp/threepp.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
};

bool move(Object3D &o, Direction dir, float limit) {

    Vector3 &pos = o.position;

    switch (dir) {
        case Direction::LEFT:
            if (pos.x - 1 < -limit) return true;
            pos.x -= 1;
            break;
        case Direction::RIGHT:
            if (pos.x + 1 > limit) return true;
            pos.x += 1;
            break;
        case Direction::UP:
            if (pos.y - 1 < -limit) return true;
            pos.y -= 1;
            break;
        case Direction::DOWN:
            if (pos.y + 1 > limit) return true;
            pos.y += 1;
            break;
    }

    return false;
}

struct GameState {

    bool running;
    bool canChangeDirection;
    float moveInterval;
    Direction direction;

    GameState() {
        reset();
    }

    void reset() {
        running = true;
        canChangeDirection = true;
        moveInterval = 0.5f;
        direction = Direction::RIGHT;
    }
};


class MyKeyListener : public KeyListener {

public:
    explicit MyKeyListener(const std::shared_ptr<Mesh> &box, GameState &state)
        : box(box), state(state) {}

    void onKeyPressed(KeyEvent evt) override {
        if (state.canChangeDirection) {
            if (evt.key == 265 && state.direction != Direction::DOWN) {
                state.direction = Direction::UP;
                state.canChangeDirection = false;
            }
            if (evt.key == 264 && state.direction != Direction::UP) {
                state.direction = Direction::DOWN;
                state.canChangeDirection = false;
            }
            if (evt.key == 263 && state.direction != Direction::RIGHT) {
                state.direction = Direction::LEFT;
                state.canChangeDirection = false;
            }
            if (evt.key == 262 && state.direction != Direction::LEFT) {
                state.direction = Direction::RIGHT;
                state.canChangeDirection = false;
            }
        }

        if (!state.running && evt.key == 82 /*r*/) {
            box->position.set(0.5, 0.5, 0);
            box->material<MaterialWithColor>()->color = Color::white;
            state.reset();
        }
    }

private:
    GameState &state;
    std::shared_ptr<Mesh> box;
};

void createAndAddGrid(int size, Scene &scene) {
    auto grid = GridHelper::create(size, size, 0x444444, 0x444444);
    grid->rotation.x = math::PI / 2;
    scene.add(grid);
}

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();

    int gridSize = 10;
    auto camera = OrthographicCamera::create(-gridSize / 2, gridSize / 2, -gridSize / 2, gridSize / 2);
    camera->position.z = static_cast<float>(gridSize);
    scene->add(camera);

    createAndAddGrid(gridSize, *scene);

    auto boxGeometry = BoxGeometry::create();
    auto boxMaterial = MeshBasicMaterial::create();
    auto box = Mesh::create(boxGeometry);
    box->position.set(0.5, 0.5, 0);
    scene->add(box);

    GameState state;
    canvas.addKeyListener(std::make_shared<MyKeyListener>(box, state));

    canvas.onWindowResize([&](WindowSize size) {
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    float t = 0;
    float limit = static_cast<float>(gridSize) / 2;
    canvas.animate([&](float dt) {
        if (state.running) {
            t += dt;

            if (t > state.moveInterval) {

                if (move(*box, state.direction, limit)) {
                    box->material<MaterialWithColor>()->color = Color::red;
                    state.running = false;
                } else {
                    state.canChangeDirection = true;
                }

                t = 0;
            }
        }

        renderer.render(scene, camera);
    });
}
