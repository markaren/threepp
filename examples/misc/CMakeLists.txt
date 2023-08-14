
add_executable(key_listener key_listener.cpp)
target_link_libraries(key_listener PUBLIC threepp)

add_executable(mouse_listener mouse_listener.cpp)
target_link_libraries(mouse_listener PUBLIC threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PUBLIC threepp)
