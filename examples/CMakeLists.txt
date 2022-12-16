
if (imgui_FOUND)
    add_compile_definitions("HAS_IMGUI")
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        set(imgui_bindings "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        add_library(imgui_glfw "${imgui_bindings}/imgui_impl_glfw.cpp" "${imgui_bindings}/imgui_impl_opengl3.cpp")
        target_link_libraries(imgui_glfw PUBLIC glad::glad glfw::glfw imgui::imgui)
        target_include_directories(imgui_glfw PUBLIC "${imgui_bindings}")

        set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_LINK_LIBRARIES imgui_glfw)
    endif()
endif ()

add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp imgui::imgui)

add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PUBLIC threepp)

add_executable(points_demo points_demo.cpp)
target_link_libraries(points_demo PUBLIC threepp)

add_subdirectory(geometries)
add_subdirectory(materials)
add_subdirectory(shaders)
add_subdirectory(objects)
add_subdirectory(lights)
add_subdirectory(misc)
add_subdirectory(helpers)
add_subdirectory(loaders)

if (BULLET_FOUND)
    add_subdirectory(bullet)
endif()
