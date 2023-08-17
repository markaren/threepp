
add_executable(hemi_light hemi_light.cpp)
target_link_libraries(hemi_light PUBLIC threepp glfw::glfw)

add_executable(point_light point_light.cpp)
target_link_libraries(point_light PUBLIC threepp glfw::glfw)

add_executable(spot_light spot_light.cpp)
target_link_libraries(spot_light PUBLIC threepp glfw::glfw)

add_executable(directional directional.cpp)
target_link_libraries(directional PUBLIC threepp glfw::glfw)
