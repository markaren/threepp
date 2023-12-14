
add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PRIVATE threepp)

add_executable(texture3d texture3d.cpp)
target_link_libraries(texture3d PRIVATE threepp)

add_executable(data_texture data_texture.cpp)
target_link_libraries(data_texture PRIVATE threepp)

if (imgui_FOUND)

    add_executable(imgui_framebuffer.cpp imgui_framebuffer.cpp)
    target_link_libraries(imgui_framebuffer.cpp threepp imgui::imgui)

endif ()
