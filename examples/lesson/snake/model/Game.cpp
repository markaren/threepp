#include "Game.hpp"

#include <utility>

using namespace snake;

namespace {

    Snake startingSnake(int width, int height) {
        const int x = width / 2;
        const int y = height / 2;
        return Snake({{x, y}, {x - 1, y}, {x - 2, y}}, Direction::Right);
    }

}// namespace

Game::Game(int width, int height, unsigned seed)
    : Game(width, height, startingSnake(width, height), {}, seed) {

    placeApple();
}

Game::Game(int width, int height, Snake snake, Vec2i apple, unsigned seed)
    : width_(width), height_(height),
      snake_(std::move(snake)), apple_(apple),
      wanted_(snake_.heading()), rng_(seed) {}

void Game::steer(Direction d) {
    wanted_ = d;
}

void Game::tick() {
    if (!alive_ || won_) return;
    ++ticks_;

    snake_.turn(wanted_);
    const Vec2i next = snake_.next();

    // the tail moves out of the way this tick (it never holds the apple)
    const bool bitesItself = snake_.occupies(next) && next != snake_.tail();

    if (!inside(next) || bitesItself) {
        alive_ = false;
        for (const auto& f : dieListeners_) f(*this);
        return;
    }

    const bool eats = next == apple_;
    snake_.advance(eats);

    if (eats) {
        ++score_;
        placeApple();
        for (const auto& f : eatListeners_) f(*this);
    }
}

bool Game::inside(Vec2i cell) const {
    return cell.x >= 0 && cell.x < width_ && cell.y >= 0 && cell.y < height_;
}

bool Game::isFree(Vec2i cell) const {
    return inside(cell) && !snake_.occupies(cell);
}

void Game::onEat(Listener f) {
    eatListeners_.push_back(std::move(f));
}

void Game::onDie(Listener f) {
    dieListeners_.push_back(std::move(f));
}

void Game::placeApple() {
    std::vector<Vec2i> free;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            if (isFree({x, y})) free.push_back({x, y});
        }
    }

    if (free.empty()) {
        won_ = true;
        return;
    }

    // std::mt19937 gives the same numbers on every compiler, so a seed is a whole game
    apple_ = free[rng_() % free.size()];
}
