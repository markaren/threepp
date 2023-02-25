
add_executable(geometries geometries.cpp)
target_link_libraries(geometries PUBLIC threepp)

add_executable(cylinder_geometry cylinder_geometry.cpp)
target_link_libraries(cylinder_geometry PUBLIC threepp)

add_executable(ring_geometry ring_geometry.cpp)
target_link_libraries(ring_geometry PUBLIC threepp)

add_executable(tube_geometry tube_geometry.cpp)
target_link_libraries(tube_geometry PUBLIC threepp)
