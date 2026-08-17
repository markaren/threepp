
add_library(imgui)
add_library(imgui::imgui ALIAS imgui)
target_sources(imgui PRIVATE
        "imgui/imgui.cpp"
        "imgui/imgui_draw.cpp"
        "imgui/imgui_impl_glfw.cpp"
        "imgui/imgui_impl_opengl3.cpp"
        "imgui/imgui_tables.cpp"
        "imgui/imgui_widgets.cpp"
)
target_link_libraries(imgui PRIVATE glfw)
target_include_directories(imgui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/imgui")

if (THREEPP_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    target_sources(imgui PRIVATE "imgui/imgui_impl_vulkan.cpp")
    target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
    # No THREEPP_WITH_VULKAN here: nothing in imgui reads it. This target used to
    # hand it out PUBLIC, which is how most examples and the editor came to see
    # the macro at all — a link edge to a UI helper deciding which backends a
    # binary supports. The threepp target states it PUBLIC itself now.
endif()
