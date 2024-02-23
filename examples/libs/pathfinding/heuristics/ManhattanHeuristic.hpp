// https://cokeandcode.com/tutorials/tilemap2.html

#ifndef PATHFINDING_MANHATTANHEURISTIC_HPP
#define PATHFINDING_MANHATTANHEURISTIC_HPP

#include "pathfinding/Heuristic.hpp"

#include <cmath>

/**
 * A heuristic that drives the search based on the Manhattan distance
 * between the current location and the target
 */
class ManhattanHeuristic: public Heuristic {

public:
    explicit ManhattanHeuristic(int minimumCost): minimumCost_(minimumCost) {}

    float getCost(TileBasedMap& map, const Coordinate& start, const Coordinate& target) override {
        return minimumCost_ * (std::abs(start.x - target.x) + std::abs(start.y - target.y));
    }

private:
    int minimumCost_;
};

#endif//PATHFINDING_MANHATTANHEURISTIC_HPP
