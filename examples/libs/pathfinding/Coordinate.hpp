// https://cokeandcode.com/tutorials/tilemap2.html

#ifndef PATHFINDING_COORDINATE_HPP
#define PATHFINDING_COORDINATE_HPP

#include <ostream>

/**
 * Represents an xy coordinate in a grid
 */
class Coordinate {
public:
    int x, y;

    Coordinate(): Coordinate(0, 0) {}
    Coordinate(int x, int y): x(x), y(y) {}

    Coordinate operator+(const Coordinate& other) const {
        return {x + other.x, y + other.y};
    }

    Coordinate operator-(const Coordinate& other) const {
        return {x - other.x, y - other.y};
    }

    Coordinate& operator+=(const Coordinate& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Coordinate& operator-=(const Coordinate& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    bool operator==(const Coordinate& other) const {
        return x == other.x && y == other.y;
    }

    friend std::ostream& operator<<(std::ostream& os, const Coordinate& v) {
        os << "[" << v.x << ", " << v.y << "]";
        return os;
    }
};

#endif//PATHFINDING_COORDINATE_HPP
