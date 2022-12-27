
#include "threepp/threepp.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Snake;


struct GameState {

    int gridSize;
    bool running;
    float moveInterval;
    Direction direction;
    Vector2 foodpos;

    explicit GameState(Scene &scene, int gridSize) : gridSize(gridSize) {

        auto foodGeometry = BoxGeometry::create();
        foodGeometry->translate(0.5, 0.5, 0);
        auto foodMaterial = MeshBasicMaterial::create();
        foodMaterial->color = Color::green;
        food = Mesh::create(foodGeometry, foodMaterial);
        scene.add(food);

        reset();
    }

    void spawnFood() {
        foodpos.set(math::randomInRange(0, gridSize - 1), math::randomInRange(0, gridSize - 1));
        food->position.set(foodpos.x, foodpos.y, 0);

        moveInterval -= 0.05f;
    }

    void reset() {
        running = true;
        moveInterval = 0.5f;
        direction = Direction::RIGHT;
        spawnFood();
    }

private:
    std::shared_ptr<Object3D> food;
};

class Snake : public Object3D {

public:
    bool canChangeDirection = true;

    Snake(int startRow, int startCol)
        : startRow(startRow), startCol(startCol) {

        boxGeometry->translate(0.5, 0.5, 0);
        snakeHead = Mesh::create(boxGeometry, boxMaterial);

        reset();
    }

    void reset() {

        boxMaterial->color = Color::white;
        snakeHead->position.set(static_cast<float>(startRow), static_cast<float>(startCol), 0);

        for (auto &o : body) {
            remove(o);
        }
        body.clear();

        add(snakeHead);
        body.emplace_back(snakeHead);

        movements.clear();
    }

    void move(GameState &state) {

        movements.insert(movements.begin(), Vector2{snakeHead->position.x, snakeHead->position.y});

        if (movements.size() >= body.size()) {
            movements.pop_back();
        }

        switch (state.direction) {
            case Direction::LEFT:
                if (snakeHead->position.x - 1 < 0) {
                    state.running = false;
                } else {
                    snakeHead->position.x -= 1;
                }
                break;
            case Direction::RIGHT:
                if (snakeHead->position.x + 2 > state.gridSize) {
                    state.running = false;
                } else {
                    snakeHead->position.x += 1;
                }
                break;
            case Direction::UP:
                if (snakeHead->position.y - 1 < 0) {
                    state.running = false;
                } else {
                    snakeHead->position.y -= 1;
                }
                break;
            case Direction::DOWN:
                if (snakeHead->position.y + 2 > state.gridSize) {
                    state.running = false;
                } else {
                    snakeHead->position.y += 1;
                }
                break;
        }

        if (!state.running) {
            boxMaterial->color = Color::red;
            return;
        }

        if (state.foodpos.distanceTo(Vector2(snakeHead->position.x, snakeHead->position.y)) < 0.1) {
            grow();

            state.spawnFood();
            std::cout << "yummy" << std::endl;
        }

        for (unsigned i = 1; i < body.size(); ++i) {
            auto p = movements[i - 1];
            body.at(i)->position.set(p.x, p.y, 0);
        }

    }


private:
    int startRow;
    int startCol;

    std::vector<Vector2> movements;
    std::vector<std::shared_ptr<Object3D>> body;

    std::shared_ptr<BoxGeometry> boxGeometry = BoxGeometry::create();
    std::shared_ptr<MeshBasicMaterial> boxMaterial = MeshBasicMaterial::create();
    std::shared_ptr<Mesh> snakeHead;

    void grow() {
        auto part = Mesh::create(boxGeometry, boxMaterial);
        part->position.copy(snakeHead->position);
        body.emplace_back(part);
        add(part);
    }
};


class MyKeyListener : public KeyListener {

public:
    explicit MyKeyListener(Snake &snake, GameState &state)
        : snake(snake), state(state) {}

    void onKeyPressed(KeyEvent evt) override {
        if (snake.canChangeDirection) {
            if (evt.key == 265 && state.direction != Direction::DOWN) {
                state.direction = Direction::UP;
                snake.canChangeDirection = false;
            }
            if (evt.key == 264 && state.direction != Direction::UP) {
                state.direction = Direction::DOWN;
                snake.canChangeDirection = false;
            }
            if (evt.key == 263 && state.direction != Direction::RIGHT) {
                state.direction = Direction::LEFT;
                snake.canChangeDirection = false;
            }
            if (evt.key == 262 && state.direction != Direction::LEFT) {
                state.direction = Direction::RIGHT;
                snake.canChangeDirection = false;
            }
        }

        if (!state.running && evt.key == 82 /*r*/) {
            snake.reset();
            state.reset();
        }
    }

private:
    GameState &state;
    Snake &snake;
};

void createAndAddGrid(int size, Scene &scene) {
    auto grid = GridHelper::create(size, size, 0x444444, 0x444444);
    grid->rotation.x = math::PI / 2;
    grid->position.set(static_cast<float>(size) / 2, static_cast<float>(size) / 2, 0);
    scene.add(grid);
}

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();

    int gridSize = 10;
    auto camera = OrthographicCamera::create(-gridSize / 2, gridSize / 2, -gridSize / 2, gridSize / 2);
    camera->position.set(static_cast<float>(gridSize) / 2, static_cast<float>(gridSize) / 2, static_cast<float>(gridSize));
    scene->add(camera);

    createAndAddGrid(gridSize, *scene);


    GameState state(*scene, gridSize);
    auto snake = std::make_shared<Snake>(gridSize / 2, gridSize / 2);
    scene->add(snake);

    canvas.addKeyListener(std::make_shared<MyKeyListener>(*snake, state));

    canvas.onWindowResize([&](WindowSize size) {
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    float t = 0;
    canvas.animate([&](float dt) {
        if (state.running) {
            t += dt;

            if (t > state.moveInterval) {

                snake->move(state);
                snake->canChangeDirection = state.running;

                t = 0;
            }
        }

        renderer.render(scene, camera);
    });
}
