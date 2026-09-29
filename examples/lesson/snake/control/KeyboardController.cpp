#include "KeyboardController.hpp"

using namespace snake;
using threepp::Key;

Direction KeyboardController::next(const Game& game) {
    if (presses_.empty()) return game.snake().heading();

    const Direction d = presses_.front();
    presses_.pop_front();
    return d;
}

void KeyboardController::onKeyPressed(threepp::KeyEvent evt) {
    if (presses_.size() >= 3) return;

    switch (evt.key) {
        case Key::UP: presses_.push_back(Direction::Up); break;
        case Key::DOWN: presses_.push_back(Direction::Down); break;
        case Key::LEFT: presses_.push_back(Direction::Left); break;
        case Key::RIGHT: presses_.push_back(Direction::Right); break;
        default: break;
    }
}
