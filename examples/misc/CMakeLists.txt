
add_executable(mouse_key_listener mouse_key_listener.cpp)
target_link_libraries(mouse_key_listener PUBLIC threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PUBLIC threepp)

add_executable(test_capabilites test_capabilites.cpp)
target_include_directories(test_capabilites PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_capabilites
        PUBLIC
        threepp
        PRIVATE
        glfw::glfw
        glad::glad)
