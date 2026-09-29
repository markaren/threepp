#include "GreedyAI.hpp"

#include <climits>
#include <cstdlib>
#include <queue>
#include <vector>

using namespace snake;

namespace {

    int distance(Vec2i a, Vec2i b) {
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }

    // how many free cells can be reached from here (a flood fill)
    int room(const Game& game, Vec2i start) {
        std::vector<bool> seen(static_cast<size_t>(game.width() * game.height()), false);
        std::queue<Vec2i> open;
        open.push(start);
        seen[start.y * game.width() + start.x] = true;

        int count = 0;
        while (!open.empty()) {
            const Vec2i cell = open.front();
            open.pop();
            count++;
            for (Direction d : {Direction::Up, Direction::Down, Direction::Left, Direction::Right}) {
                const Vec2i n = cell + step(d);
                if (!game.isFree(n) || seen[n.y * game.width() + n.x]) continue;
                seen[n.y * game.width() + n.x] = true;
                open.push(n);
            }
        }
        return count;
    }

}// namespace

Direction GreedyAI::next(const Game& game) {
    const Snake& snake = game.snake();

    Direction best = snake.heading();
    int bestScore = INT_MIN;
    for (Direction d : {Direction::Up, Direction::Down, Direction::Left, Direction::Right}) {
        if (d == opposite(snake.heading())) continue;

        const Vec2i cell = snake.head() + step(d);
        if (!game.isFree(cell) && cell != snake.tail()) continue;

        int score = -distance(cell, game.apple());
        if (room(game, cell) < snake.length()) score -= 1000;// a dead end

        if (score > bestScore) {
            bestScore = score;
            best = d;
        }
    }
    return best;
}
