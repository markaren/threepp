// GreedyAI.hpp: heads for the apple, and avoids walls, itself and dead ends.
// Greedy, so not clever: it can still box itself in on a long snake.

#ifndef SNAKE_GREEDYAI_HPP
#define SNAKE_GREEDYAI_HPP

#include "Controller.hpp"

namespace snake {

    class GreedyAI: public Controller {
    public:
        Direction next(const Game& game) override;
    };

}// namespace snake

#endif//SNAKE_GREEDYAI_HPP
