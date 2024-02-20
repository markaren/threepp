// https://cokeandcode.com/tutorials/tilemap2.html

#ifndef PATHFINDING_HEURISTIC_HPP
#define PATHFINDING_HEURISTIC_HPP

#include "Coordinate.hpp"
#include "TileBasedMap.hpp"

/**
 * The description of a class providing a cost for a given tile based
 * on a target location and entity being moved. This heuristic controls
 * what priority is placed on different tiles during the search for a path
 *
 */
class Heuristic {

public:
    /**
	 * Get the additional heuristic cost of the given tile. This controls the
	 * order in which tiles are searched while attempting to find a path to the
	 * target location. The lower the cost the more likely the tile will
	 * be searched.
	 *
	 * @param map The map on which the path is being found
	 * @param start The coordinates of the tile being evaluated
	 * @param target The coordinates of the target location
     *
	 * @return The cost associated with the given tile
	 */
    virtual float getCost(TileBasedMap& map, const Coordinate& start, const Coordinate& target) = 0;

    virtual ~Heuristic() = default;
};

#endif//PATHFINDING_HEURISTIC_HPP
