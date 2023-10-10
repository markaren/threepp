
add_executable(stl_loader stl_loader.cpp)
target_link_libraries(stl_loader PRIVATE threepp)

add_executable(obj_loader obj_loader.cpp)
target_link_libraries(obj_loader PRIVATE threepp)

if (assimp_FOUND)
    add_executable(assimp_loader assimp_loader.cpp)
    target_link_libraries(assimp_loader PRIVATE threepp assimp::assimp)
endif ()

if (pugixml_FOUND)
    add_executable(svg_loader svg_loader.cpp)
    target_link_libraries(svg_loader PRIVATE threepp pugixml::pugixml)
    if (imgui_FOUND)
        target_link_libraries(svg_loader PRIVATE imgui::imgui)
    endif ()
endif ()
