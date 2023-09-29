
add_executable(stl_loader stl_loader.cpp)
target_link_libraries(stl_loader PRIVATE threepp glfw::glfw)

add_executable(obj_loader obj_loader.cpp)
target_link_libraries(obj_loader PRIVATE threepp glfw::glfw)

if (assimp_FOUND)
    add_executable(assimp_loader assimp_loader.cpp)
    target_link_libraries(assimp_loader PRIVATE threepp assimp::assimp glfw::glfw)
endif()

if (pugixml_FOUND)
    add_executable(svg_loader svg_loader.cpp)
    target_link_libraries(svg_loader PRIVATE threepp pugixml::pugixml glfw::glfw)
    if (imgui_FOUND)
        target_link_libraries(svg_loader PRIVATE imgui::imgui)
    endif ()
endif ()
