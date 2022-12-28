
#include "threepp/threepp.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
};


struct Snake {

    Snake(int xStart, int yStart) : xStart_(xStart), yStart_(yStart) {
        reset();
    }

    [[nodiscard]] size_t size() const {
        return positions_.size();
    }

    [[nodiscard]] const Vector2 &headPosition() const {
        return positions_.front();
    }

    bool move(const Vector2 &nextPos, const Vector2 food) {

        float foodDist = nextPos.distanceTo(food);
        if (foodDist < 0.1) {
            grow(nextPos);
            return true;
        }

        auto posCopy = positions_;

        positions_.front().copy(nextPos);
        for (unsigned i = 1; i < positions_.size(); ++i) {
            positions_[i].copy(posCopy[i-1]);
        }

        return false;
    }

    void grow(const Vector2& pos) {
        positions_.insert(positions_.begin(), pos);
    }

    void reset() {
        positions_.clear();
        positions_.emplace_back(xStart_, yStart_);
    }

    [[nodiscard]] const std::vector<Vector2> &positions() const {
        return positions_;
    }

    [[nodiscard]] bool checkSelfCollision(const Vector2& pos) const {
        return std::any_of(positions_.begin(), positions_.end(), [&](auto &p) {
            return pos.distanceTo(p) < 0.1;
        });
    }


private:
    int xStart_, yStart_;
    std::vector<Vector2> positions_;
};


struct SnakeGame {

    Direction direction;
    Direction nextDirection;

    explicit SnakeGame(int gridSize)
        : gridSize_(gridSize),
          snake_(gridSize / 2, gridSize / 2) {

        reset();
    }

    void update(float dt) {

        if (t_ < moveInterval_) {
            t_ += dt;
            return;
        }

        t_ = 0;// reset timer

        Vector2 nextMove;
        switch (nextDirection) {
            case Direction::LEFT:
                nextMove.x -= 1;
                break;
            case Direction::RIGHT:
                nextMove.x += 1;
                break;
            case Direction::UP:
                nextMove.y -= 1;
                break;
            case Direction::DOWN:
                nextMove.y += 1;
                break;
        }
        nextMove += snake_.headPosition();

        if (checkBorderCollision(nextMove) || snake_.checkSelfCollision(nextMove)) {
            running_ = false;
            return;
        }

        if (snake_.move(nextMove, foodPos_)) {
            spawnFood();
            moveInterval_ -= (moveInterval_ * 0.05f);
        }

        direction = nextDirection;
    }

    [[nodiscard]] bool isRunning() const {
        return running_;
    }

    [[nodiscard]] Vector2 foodPos() const {
        return foodPos_;
    }

    [[nodiscard]] int gridSize() const {
        return gridSize_;
    }

    [[nodiscard]] const Snake &snake() const {
        return snake_;
    }

    void reset() {
        t_ = 0;
        running_ = true;
        moveInterval_ = 0.5f;
        nextDirection = Direction::RIGHT;
        direction = nextDirection;
        snake_.reset();
        spawnFood();
    }

private:
    float t_;
    int gridSize_;
    bool running_;

    float moveInterval_;
    Vector2 foodPos_;
    Snake snake_;

    void spawnFood() {
        do {
            foodPos_.set(math::randomInRange(0, gridSize_ - 1), math::randomInRange(0, gridSize_ - 1));
        } while (snake_.checkSelfCollision(foodPos_));
    }

    [[nodiscard]] bool checkBorderCollision(Vector2 pos) const {
        if (pos.x < 0 || pos.x >= gridSize_) return true;
        if (pos.y < 0 || pos.y >= gridSize_) return true;

        return false;
    }
};

class SnakeScene : public Scene {

public:
    explicit SnakeScene(SnakeGame &game) : game_(game) {

        int size = game.gridSize();
        auto grid = GridHelper::create(size, size, 0x444444, 0x444444);
        grid->rotation.x = math::PI / 2;
        grid->position.set(static_cast<float>(size) / 2, static_cast<float>(size) / 2, 0);
        add(grid);

        boxGeometry_ = BoxGeometry::create();
        boxGeometry_->translate(0.5, 0.5, 0);

        auto foodMaterial = MeshBasicMaterial::create();
        foodMaterial->color = Color::green;

        food_ = Mesh::create(boxGeometry_, foodMaterial);
        add(food_);

        snakeMaterial_ = MeshBasicMaterial::create();
        snake_.emplace_back(Mesh::create(boxGeometry_, snakeMaterial_));
        add(snake_.back());

        camera_ = OrthographicCamera::create(-size / 2, size / 2, -size / 2, size / 2);
        camera_->position.set(static_cast<float>(size) / 2, static_cast<float>(size) / 2, static_cast<float>(size));
        add(camera_);
    }

    void update() {
        auto foodPos = game_.foodPos();
        food_->position.set(foodPos.x, foodPos.y, 0);

        auto &positions = game_.snake().positions();
        for (unsigned i = 0; i < positions.size(); ++i) {
            auto& pos = positions.at(i);
            if (positions.size() != snake_.size()) {
                snake_.emplace_back(Mesh::create(boxGeometry_, snakeMaterial_));
                add(snake_.back());
            }
            snake_.at(i)->position.set(pos.x, pos.y, 0);
        }

        if (!game_.isRunning()) {
            snakeMaterial_->color = Color::red;
        }

    }

    void reset() {
        // keep initial box
        auto head = snake_.front();
        for (unsigned i = 1; i < snake_.size(); ++i) {
            remove(snake_.at(i));
        }
        snake_.clear();
        snake_.emplace_back(head);

        snakeMaterial_->color = Color::white;
    }

    [[nodiscard]] OrthographicCamera &camera() const {
        return *camera_;
    }

private:

    SnakeGame &game_;
    std::shared_ptr<Mesh> food_;
    std::shared_ptr<OrthographicCamera> camera_;

    std::shared_ptr<BoxGeometry> boxGeometry_;
    std::shared_ptr<MeshBasicMaterial> snakeMaterial_;
    std::vector<std::shared_ptr<Mesh>> snake_;
};


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
