
add_executable(helpers helpers.cpp)
target_link_libraries(helpers PUBLIC threepp glfw::glfw)

add_executable(camera_helper camera_helper.cpp)
target_link_libraries(camera_helper PUBLIC threepp glfw::glfw)
