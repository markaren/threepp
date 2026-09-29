// AsciiView.hpp: the same game as text. No threepp in here at all.

#ifndef SNAKE_ASCIIVIEW_HPP
#define SNAKE_ASCIIVIEW_HPP

#include "model/Game.hpp"

#include <ostream>
#include <string>

namespace snake {

    class AsciiView {
    public:
        explicit AsciiView(std::ostream& out): out_(out) {}

        // prints the board once per tick
        void update(const Game& game);

        static std::string draw(const Game& game);

    private:
        std::ostream& out_;
        int seenTick_ = -1;
    };

}// namespace snake

#endif//SNAKE_ASCIIVIEW_HPP
