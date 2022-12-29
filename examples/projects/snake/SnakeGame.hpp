
#ifndef THREEPP_SNAKEGAME_HPP
#define THREEPP_SNAKEGAME_HPP

#include "threepp/threepp.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
};


class Snake {

public:

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


class SnakeGame {

public:
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
    float t_{};
    int gridSize_{};
    bool running_{};

    float moveInterval_{};
    Vector2 foodPos_;
    Snake snake_;

    void spawnFood() {
        do {
            auto x = static_cast<float>(math::randomInRange(0, gridSize_ - 1));
            auto y = static_cast<float>(math::randomInRange(0, gridSize_ - 1));
            foodPos_.set(x, y);
        } while (snake_.checkSelfCollision(foodPos_));
    }

    [[nodiscard]] bool checkBorderCollision(const Vector2& pos) const {
        auto size = static_cast<float>(gridSize_);
        if (pos.x < 0 || pos.x >= size) return true;
        if (pos.y < 0 || pos.y >= size) return true;

        return false;
    }
};

#endif//THREEPP_SNAKEGAME_HPP
