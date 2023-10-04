
add_executable(mouse_key_listener mouse_key_listener.cpp)
target_link_libraries(mouse_key_listener PRIVATE threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PRIVATE threepp)

if (imgui_FOUND)
    add_executable(clipping clipping.cpp)
    target_link_libraries(clipping PRIVATE threepp imgui::imgui)
endif ()

add_executable(test_capabilites test_capabilites.cpp)
target_include_directories(test_capabilites PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_capabilites
        PRIVATE
        threepp
        glfw::glfw
        glad::glad)
