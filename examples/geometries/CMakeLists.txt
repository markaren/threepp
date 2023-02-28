
add_executable(geometries geometries.cpp)
target_link_libraries(geometries PUBLIC threepp)

add_executable(tube_geometry tube_geometry.cpp)
target_link_libraries(tube_geometry PUBLIC threepp)
