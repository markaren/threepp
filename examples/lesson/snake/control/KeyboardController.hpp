// KeyboardController.hpp: the arrow keys. It inherits two contracts, one of ours
// (Controller) and one of threepp's (KeyListener), and no implementation.

#ifndef SNAKE_KEYBOARDCONTROLLER_HPP
#define SNAKE_KEYBOARDCONTROLLER_HPP

#include "Controller.hpp"

#include "threepp/input/KeyListener.hpp"

#include <deque>

namespace snake {

    class KeyboardController: public Controller, public threepp::KeyListener {
    public:
        Direction next(const Game& game) override;

        void onKeyPressed(threepp::KeyEvent evt) override;

    private:
        // presses wait their turn, one per tick: a quick "up, left" is two turns, not a U-turn
        std::deque<Direction> presses_;
    };

}// namespace snake

#endif//SNAKE_KEYBOARDCONTROLLER_HPP
