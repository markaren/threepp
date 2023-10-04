
#ifndef THREEPP_SNAKEGAME_HPP
#define THREEPP_SNAKEGAME_HPP

#include "Snake.hpp"

using namespace threepp;

enum class Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN
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

    [[nodiscard]] const Snake& snake() const {
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
            auto x = static_cast<float>(math::randInt(0, gridSize_ - 1));
            auto y = static_cast<float>(math::randInt(0, gridSize_ - 1));
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
