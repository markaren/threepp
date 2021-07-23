include_directories(test ${CMAKE_SOURCE_DIR}/include)

add_executable(point_light point_light.cpp)
target_link_libraries(point_light PUBLIC threepp)

add_executable(directional directional.cpp)
target_link_libraries(directional PUBLIC threepp)
