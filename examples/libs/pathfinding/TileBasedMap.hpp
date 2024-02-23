// https://cokeandcode.com/tutorials/tilemap2.html

#ifndef PATHFINDING_TILEBASEDMAP_HPP
#define PATHFINDING_TILEBASEDMAP_HPP

#include "Coordinate.hpp"

/**
 * The description for the data we're pathfinding over. This provides the contract
 * between the data being searched (i.e. the in game map) and the path finding
 * generic tools
 */
class TileBasedMap {

public:
    /**
	 * Get the width of the tile map.
	 *
	 * @return The number of tiles across the map
	 */
    [[nodiscard]] virtual unsigned int width() const = 0;

    /**
	 * Get the height of the tile map.
	 *
	 * @return The number of tiles down the map
	 */
    [[nodiscard]] virtual unsigned int height() const = 0;

    /**
     * Check if the given location is blocked, i.e. blocks movement of
     * the supplied mover.
     * @param c The coordinate of the tile to check
     *
     * @return True if the location is blocked
     */
    [[nodiscard]] virtual bool blocked(const Coordinate& c) const = 0;

    /**
	 * Get the cost of moving through the given tile. This can be used to
	 * make certain areas more desirable. A simple and valid implementation
	 * of this method would be to return 1 in all cases.
	 *
	 * @param start The coordinate of the tile we're moving from
	 * @param target The coordinate of the tile we're moving to
     *
	 * @return The relative cost of moving across the given tile
	 */
    virtual float getCost(const Coordinate& start, const Coordinate& target) = 0;

    virtual ~TileBasedMap() = default;
};

#endif//PATHFINDING_TILEBASEDMAP_HPP
