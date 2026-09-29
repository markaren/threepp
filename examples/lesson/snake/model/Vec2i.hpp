// Vec2i.hpp: a cell on the board, and the four ways to leave it.

#ifndef SNAKE_VEC2I_HPP
#define SNAKE_VEC2I_HPP

namespace snake {

    struct Vec2i {
        int x = 0;
        int y = 0;

        bool operator==(const Vec2i&) const = default;

        Vec2i operator+(Vec2i other) const {
            return {x + other.x, y + other.y};
        }
    };

    // x grows to the right, y grows up
    enum class Direction { Up, Down, Left, Right };

    inline Vec2i step(Direction d) {
        switch (d) {
            case Direction::Up: return {0, 1};
            case Direction::Down: return {0, -1};
            case Direction::Left: return {-1, 0};
            case Direction::Right: return {1, 0};
        }
        return {};
    }

    inline Direction opposite(Direction d) {
        switch (d) {
            case Direction::Up: return Direction::Down;
            case Direction::Down: return Direction::Up;
            case Direction::Left: return Direction::Right;
            case Direction::Right: return Direction::Left;
        }
        return d;
    }

}// namespace snake

#endif//SNAKE_VEC2I_HPP
