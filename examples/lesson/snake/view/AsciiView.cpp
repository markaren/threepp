#include "AsciiView.hpp"

using namespace snake;

void AsciiView::update(const Game& game) {
    if (game.ticks() == seenTick_) return;
    seenTick_ = game.ticks();

    out_ << draw(game) << std::flush;
}

std::string AsciiView::draw(const Game& game) {
    const std::string wall(game.width() + 2, '#');

    std::string text = wall + "\n";
    for (int y = game.height() - 1; y >= 0; y--) {// the top row first
        text += '#';
        for (int x = 0; x < game.width(); x++) {
            const Vec2i cell{x, y};
            if (cell == game.snake().head()) text += game.alive() ? 'O' : 'X';
            else if (game.snake().occupies(cell)) text += 'o';
            else if (cell == game.apple() && !game.won()) text += '@';
            else text += ' ';
        }
        text += "#\n";
    }
    text += wall + "  score " + std::to_string(game.score()) + "\n";
    return text;
}
