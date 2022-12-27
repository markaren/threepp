
#include "threepp/threepp.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
};

struct GameState {

    int gridSize;
    bool running;
    float moveInterval;
    Direction direction;
    Direction nextDirection;
    Vector2 foodPos;

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
        foodPos.set(math::randomInRange(0, gridSize - 1), math::randomInRange(0, gridSize - 1));
        food->position.set(foodPos.x, foodPos.y, 0);

        moveInterval -= (moveInterval * 0.05f);
    }

    void reset() {
        running = true;
        moveInterval = 0.5f;
        nextDirection = Direction::RIGHT;
        direction = nextDirection;
        spawnFood();
    }

private:
    std::shared_ptr<Object3D> food;
};

class Snake : public Object3D {

public:
    Snake(GameState &state, int startRow, int startCol)
        : state(state), startRow(startRow), startCol(startCol) {

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

    void move() {

        movements.insert(movements.begin(), Vector2{snakeHead->position.x, snakeHead->position.y});

        if (movements.size() >= body.size()) {
            movements.pop_back();
        }

        Vector2 nextPos(snakeHead->position.x, snakeHead->position.y);
        switch (state.nextDirection) {
            case Direction::LEFT:
                nextPos.x -= 1;
                break;
            case Direction::RIGHT:
                nextPos.x += 1;
                break;
            case Direction::UP:
                nextPos.y -= 1;
                break;
            case Direction::DOWN:
                nextPos.y += 1;
                break;
        }

        state.running = !checkBorderCollision(nextPos) && !checkSelfCollision(nextPos);

        if (!state.running) {
            boxMaterial->color = Color::red;
            return;
        }

        state.direction = state.nextDirection;
        snakeHead->position.set(nextPos.x, nextPos.y, 0);

        if (state.foodPos.distanceTo(nextPos) < 0.1) {
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
    GameState &state;

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

    [[nodiscard]] bool checkBorderCollision(Vector2 pos) const {
        if (pos.x < 0 || pos.x >= state.gridSize) return true;
        if (pos.y < 0 || pos.y >= state.gridSize) return true;

        return false;
    }

    [[nodiscard]] bool checkSelfCollision(Vector2 pos) const {
        Vector3 tmp;
        for (auto &o : body) {
            if (o->position.distanceTo(tmp.set(pos.x, pos.y, 0)) < 0.1) {
                return true;
            }
        }

        return false;
    }
};


class MyKeyListener : public KeyListener {

public:
    explicit MyKeyListener(Snake &snake, GameState &state)
        : snake(snake), state(state) {}

    void onKeyPressed(KeyEvent evt) override {

        if (state.running) {

            if (evt.key == 265 && state.direction != Direction::DOWN) {
                state.nextDirection = Direction::UP;
            }
            if (evt.key == 264 && state.direction != Direction::UP) {
                state.nextDirection = Direction::DOWN;
            }
            if (evt.key == 263 && state.direction != Direction::RIGHT) {
                state.nextDirection = Direction::LEFT;
            }
            if (evt.key == 262 && state.direction != Direction::LEFT) {
                state.nextDirection = Direction::RIGHT;
            }

        } else if (evt.key == 82 /*r*/) {

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
    auto snake = std::make_shared<Snake>(state, gridSize / 2, gridSize / 2);
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

                snake->move();

                t = 0;
            }
        }

        renderer.render(scene, camera);
    });
}
