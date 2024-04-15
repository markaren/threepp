
set(headers
        "${CMAKE_CURRENT_SOURCE_DIR}/GameMap.hpp"
)

add_example(NAME "Pathfinding" SOURCES main.cpp ${headers})
if (TARGET "Pathfinding")
    target_link_libraries("Pathfinding" PRIVATE pathfinding)
endif ()
