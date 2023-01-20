cmake_minimum_required(VERSION 3.15)
project(threepp_vcpkg_test)

set(CMAKE_CXX_STANDARD 17)

find_package(threepp CONFIG REQUIRED)
find_package(imgui CONFIG)

if (imgui_FOUND)

    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        set(imgui_bindings "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        add_library(imgui_glfw "${imgui_bindings}/imgui_impl_glfw.cpp" "${imgui_bindings}/imgui_impl_opengl3.cpp")
        target_link_libraries(imgui_glfw PUBLIC glad::glad glfw::glfw imgui::imgui)
        target_include_directories(imgui_glfw PUBLIC "${imgui_bindings}")

        set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_LINK_LIBRARIES imgui_glfw)
    endif()

    set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS HAS_IMGUI)

endif ()

add_executable(test main.cpp)
target_link_libraries(test PRIVATE threepp::threepp imgui::imgui)
