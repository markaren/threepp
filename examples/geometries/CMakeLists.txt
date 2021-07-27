include_directories(test ${CMAKE_SOURCE_DIR}/include)


add_executable(box_geometry box_geometry.cpp)
target_link_libraries(box_geometry PUBLIC threepp)

add_executable(sphere_geometry sphere_geometry.cpp)
target_link_libraries(sphere_geometry PUBLIC threepp)

add_executable(tube_geometry tube_geometry.cpp)
target_link_libraries(tube_geometry PUBLIC threepp)
