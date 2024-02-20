
set(headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Coordinate.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GameMap.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Heuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Path.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PathFinder.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TileBasedMap.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/astar/AStar.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/heuristics/ClosestHeuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/heuristics/ClosestSquaredHeuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/heuristics/ManhattanHeuristic.hpp"
)

add_example(NAME "Pathfinding" SOURCES main.cpp ${headers})
if (TARGET "Pathfinding")
    target_include_directories(Pathfinding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif ()
