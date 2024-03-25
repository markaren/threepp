
add_library(kine INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/Kine.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineComponent.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineLimit.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineLink.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/kine/ik/CCDSolver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/ik/IKSolver.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/KineJoint.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/PrismaticJoint.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/RevoluteJoint.hpp"
)
target_include_directories(kine INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(pathfinding INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/Coordinate.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/Heuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/Path.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/PathFinder.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/TileBasedMap.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/algorithm/AStar.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/heuristics/ClosestHeuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/heuristics/ClosestSquaredHeuristic.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pathfinding/heuristics/ManhattanHeuristic.hpp"
)
target_include_directories(pathfinding INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if (CURL_FOUND)
    add_library(geothreepp
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/lod/LODControl.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/lod/LODFrustum.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/lod/LODRadial.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/lod/LODRaycast.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/lod/LODRaycast.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/geo/nodes/MapNode.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/nodes/MapNode.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/nodes/MapPlaneNode.hpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/geo/geometries/MapNodeGeometry.hpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/geo/providers/MapProvider.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/providers/BingMapsProvider.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/providers/DebugMapsProvider.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/providers/OpenStreetMapsProvider.hpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/geo/utils/UnitUtils.hpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/geo/MapView.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/geo/MapView.cpp"
    )

    target_include_directories(geothreepp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(geothreepp PUBLIC threepp::threepp CURL::libcurl)

endif ()
