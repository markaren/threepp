
add_executable(test_event test_event.cpp)
target_link_libraries(test_event PUBLIC threepp)

add_executable(test_uniform test_uniform.cpp)
target_link_libraries(test_uniform PUBLIC threepp)

add_executable(key_listener key_listener.cpp)
target_link_libraries(key_listener PUBLIC threepp)

add_executable(mouse_listener mouse_listener.cpp)
target_link_libraries(mouse_listener PUBLIC threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PUBLIC threepp)

add_executable(test_capabilites test_capabilites.cpp)
target_link_libraries(test_capabilites
        PUBLIC
        threepp
        PRIVATE
        glfw::glfw
        glad::glad)
