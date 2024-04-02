
if (NOT imgui_FOUND)

    message("ImGui not found. Using bundled version")

    add_library(imgui OBJECT)
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

    target_compile_definitions(imgui INTERFACE HAS_IMGUI)

    set(imgui_FOUND 1 PARENT_SCOPE)

endif ()
