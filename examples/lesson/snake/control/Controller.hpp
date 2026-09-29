// Controller.hpp: anything that can play. A person at the keys, an AI, a replay:
// the game loop asks each the same question and doesn't care which it is.

#ifndef SNAKE_CONTROLLER_HPP
#define SNAKE_CONTROLLER_HPP

#include "model/Game.hpp"

namespace snake {

    class Controller {
    public:
        virtual ~Controller() = default;

        // asked once per tick: which way now?
        virtual Direction next(const Game& game) = 0;
    };

}// namespace snake

#endif//SNAKE_CONTROLLER_HPP
