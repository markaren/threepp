
#ifndef PATHFINDING_ASTAR_HPP
#define PATHFINDING_ASTAR_HPP

#include "pathfinding/Heuristic.hpp"
#include "pathfinding/PathFinder.hpp"
#include "pathfinding/TileBasedMap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

class AStar: public Pathfinder {

private:
    struct Node;// forward declaration

public:
    explicit AStar(std::unique_ptr<TileBasedMap> map, std::unique_ptr<Heuristic> heuristic = nullptr)
        : map(std::move(map)), heuristic(std::move(heuristic)) {

        for (int x = 0; x < this->map->width(); x++) {
            nodes.emplace_back();
            for (int y = 0; y < this->map->height(); y++) {
                nodes[x].emplace_back(Coordinate{x, y});
            }
        }
    }

    AStar& setMaxSearchDistance(int distance) {
        maxSearchDistance = distance;
        return *this;
    }
    AStar& setAllowDiagMovement(bool allow) {
        allowDiagMovement = allow;
        return *this;
    }

    std::optional<Path> findPath(const Coordinate& s, const Coordinate& t) override {

        std::vector<Node*> closed;
        std::vector<Node*> open;

        // easy first check, if the destination is blocked, we can't get there
        if (map->blocked(s) || map->blocked(t)) {
            return std::nullopt;
        }

        // initial state for A*. The closed group is empty. Only the starting
        // tile is in the open list and it's cost is zero, i.e. we're already there
        nodes[s.x][s.y].cost = 0;
        nodes[s.x][s.y].depth = 0;

        open.push_back(&nodes[s.x][s.y]);

        nodes[t.x][t.y].parent = nullptr;

        // while we haven't found the goal and haven't exceeded our max search depth
        int maxDepth = 0;
        while ((maxDepth < maxSearchDistance) && (!open.empty())) {
            // pull out the first node in our open list, this is determined to
            // be the most likely to be the next step based on our heuristic
            Node* current = open.front();
            if (current == &nodes[t.x][t.y]) {
                break;
            }

            removeFrom(open, current);
            closed.push_back(current);

            // search through all the neighbours of the current node evaluating
            // them as next steps
            for (int x = -1; x < 2; x++) {
                for (int y = -1; y < 2; y++) {
                    // not a neighbour, it's the current tile
                    if ((x == 0) && (y == 0)) {
                        continue;
                    }

                    // if we're not allowing diagonal movement then only
                    // one of x or y can be set
                    if (!allowDiagMovement) {
                        if ((x != 0) && (y != 0)) {
                            continue;
                        }
                    }

                    // determine the location of the neighbour and evaluate it
                    Coordinate p{x + current->xy.x, y + current->xy.y};

                    if (isValidLocation(s, p)) {
                        // the cost to get to this node is cost the current plus the
                        // movement cost to reach this node. Note that the heuristic value
                        // is only used in the sorted open list
                        float nextStepCost = current->cost + getMovementCost(current->xy, p);
                        Node* neighbour = &nodes[p.x][p.y];

                        // if the new cost we've determined for this node is lower than
                        // it has been previously makes sure the node hasn't been discarded.
                        // We've determined that there might have been a better path to get
                        // to this node, so it needs to be re-evaluated
                        if (nextStepCost < neighbour->cost) {
                            if (contains(open, neighbour)) {
                                removeFrom(open, neighbour);
                            }
                            if (contains(closed, neighbour)) {
                                removeFrom(closed, neighbour);
                            }
                        }

                        // if the node hasn't already been processed and discarded then
                        // reset it's cost to our current cost and add it as a next possible
                        // step (i.e. to the open list)
                        if (!contains(open, neighbour) && !(contains(closed, neighbour))) {
                            neighbour->cost = nextStepCost;
                            neighbour->heuristic = getHeuristicCost(p, t);
                            maxDepth = std::max(maxDepth, neighbour->setParent(current));
                            open.push_back(neighbour);
                            std::sort(open.begin(), open.end());
                        }
                    }
                }
            }
        }

        // since we've got an empty open list, or we've run out of search
        // there was no path. Just return null
        if (nodes[t.x][t.y].parent == nullptr) {
            return std::nullopt;
        }

        // At this point we've definitely found a path so we can uses the parent
        // references of the nodes to find out way from the target location back
        // to the start recording the nodes on the way.
        Path path;
        Node* target = &nodes[t.x][t.y];
        while (target != &nodes[s.x][s.y]) {
            path.prependStep(target->xy);
            target = target->parent;
        }
        path.prependStep(s);

        // that's it, we have our path
        return path;
    }

private:
    /**
	 * A single node in the search graph
	 */
    struct Node {

        // The coordinate of the node
        Coordinate xy;

        float cost = 0;
        /** The parent of this node, how we reached it in the search */
        Node* parent = nullptr;
        /** The heuristic cost of this node */
        float heuristic = 0;
        /** The search depth of this node */
        int depth = 0;

        explicit Node(const Coordinate& c): xy(c) {}

        int setParent(Node* p) {
            depth = p->depth + 1;
            this->parent = p;

            return depth;
        }

        // used for sorting
        bool operator<(const Node& other) const {
            float f = heuristic + cost;
            float of = other.heuristic + other.cost;

            return f < of;
        }

        ~Node() = default;
    };

    std::unique_ptr<TileBasedMap> map;
    std::unique_ptr<Heuristic> heuristic;

    std::vector<std::vector<Node>> nodes;

    int maxSearchDistance = 100;
    bool allowDiagMovement = true;

    static bool contains(const std::vector<Node*>& list, Node* node) {

        return std::find(list.begin(), list.end(), node) != std::end(list);
    }

    static void removeFrom(std::vector<Node*>& list, Node* node) {

        list.erase(std::remove(list.begin(), list.end(), node), list.end());
    }

    /**
	 * Check if a given location is valid for the supplied mover
	 *
	 * @param c1 The starting x coordinate
	 * @param c2 The coordinate of the location to check
     *
	 * @return True if the location is valid
	 */
    [[nodiscard]] bool isValidLocation(const Coordinate& c1, const Coordinate& c2) const {

        bool invalid = (c2.x < 0) || (c2.y < 0) || (c2.x >= map->width()) ||
                       (c2.y >= map->height());

        if ((!invalid) && ((c1.x != c2.x) || (c1.y != c2.y))) {
            invalid = map->blocked(c2);
        }

        return !invalid;
    }

    /**
	 * Get the cost to move through a given location
	 *
	 * @param s The coordinate of the tile whose cost is being determined
	 * @param t The coordinate of the target location
     *
	 * @return The cost of movement through the given tile
	 */
    [[nodiscard]] float getMovementCost(const Coordinate& s, const Coordinate& t) const {

        return map->getCost(s, t);
    }

    /**
	 * Get the heuristic cost for the given location. This determines in which
	 * order the locations are processed.
	 *
	 * @param s The coordinate of the tile whose cost is being determined
	 * @param t The coordinate of the target location
     *
	 * @return The heuristic cost assigned to the tile
	 */
    float getHeuristicCost(const Coordinate& s, const Coordinate& t) {

        if (!heuristic) return 0;// Dijkstra's

        return heuristic->getCost(*map, s, t);
    }
};

#endif// PATHFINDING_ASTAR_HPP
