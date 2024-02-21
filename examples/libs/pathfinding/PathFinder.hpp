// https://cokeandcode.com/tutorials/tilemap2.html

#ifndef PATHFINDING_PATHFINDER_HPP
#define PATHFINDING_PATHFINDER_HPP

#include "Path.hpp"

#include <optional>

class Pathfinder {

public:
    /**
	 * Find a path from the starting location to the target
	 * location, avoiding blockages and attempting to honour costs
	 * provided by the tile map.
	 *
	 * @param start The coordinate of the start location
	 * @param target The coordinate of the target location
     *
	 * @return The path found from start to end, or null if no path can be found.
	 */
    [[nodiscard]] virtual std::optional<Path> findPath(const Coordinate& start, const Coordinate& target) = 0;

    virtual ~Pathfinder() = default;
};

#endif//PATHFINDING_PATHFINDER_HPP
