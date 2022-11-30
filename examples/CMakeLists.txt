include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp)

add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PUBLIC threepp)

add_executable(points_demo points_demo.cpp)
target_link_libraries(points_demo PUBLIC threepp)

add_subdirectory(geometries)
add_subdirectory(materials)
add_subdirectory(shaders)
add_subdirectory(objects)
add_subdirectory(lights)
add_subdirectory(misc)
add_subdirectory(helpers)
add_subdirectory(loaders)

if (BULLET_FOUND)
    add_subdirectory(bullet)
endif()
