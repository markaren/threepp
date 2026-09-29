// Snake.hpp: the snake itself. Cells and a heading, nothing to draw.

#ifndef SNAKE_SNAKE_HPP
#define SNAKE_SNAKE_HPP

#include "Vec2i.hpp"

#include <deque>
#include <vector>

namespace snake {

    class Snake {
    public:
        // body[0] is the head
        Snake(std::vector<Vec2i> body, Direction heading);

        Vec2i head() const { return body_.front(); }
        Vec2i tail() const { return body_.back(); }
        const std::deque<Vec2i>& body() const { return body_; }
        int length() const { return static_cast<int>(body_.size()); }
        Direction heading() const { return heading_; }

        bool occupies(Vec2i cell) const;

        // where the head goes on the next move
        Vec2i next() const { return head() + step(heading_); }

        // a snake can't turn back onto its own neck: such a turn is ignored
        void turn(Direction d);

        // one cell forward; the tail stays put when the snake grows
        void advance(bool grow);

    private:
        std::deque<Vec2i> body_;
        Direction heading_;
    };

}// namespace snake

#endif//SNAKE_SNAKE_HPP
