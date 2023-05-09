
add_executable(stl_loader stl_loader.cpp)
target_link_libraries(stl_loader PUBLIC threepp)

add_executable(obj_loader obj_loader.cpp)
target_link_libraries(obj_loader PUBLIC threepp)

if (assimp_FOUND)
    add_executable(assimp_loader assimp_loader.cpp)
    target_link_libraries(assimp_loader PUBLIC threepp assimp::assimp)
endif()

if (THREEPP_WITH_SVG_FEATURE AND PUGIXML_FOUND)
    add_executable(svg_loader svg_loader.cpp)
    target_link_libraries(svg_loader PUBLIC threepp)
    if (imgui_FOUND)
        target_link_libraries(svg_loader PRIVATE imgui::imgui)
    endif ()
endif ()
