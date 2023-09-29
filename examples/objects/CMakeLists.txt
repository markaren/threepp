
add_executable(instancing instancing.cpp)
target_link_libraries(instancing PRIVATE threepp glfw::glfw)

add_executable(lod lod.cpp)
target_link_libraries(lod PRIVATE threepp glfw::glfw)

add_executable(water water.cpp)
target_link_libraries(water PRIVATE threepp glfw::glfw)

add_executable(sprite sprite.cpp)
target_link_libraries(sprite PRIVATE threepp glfw::glfw)

add_executable(points points.cpp)
target_link_libraries(points PRIVATE threepp glfw::glfw)

if (assimp_FOUND)
    add_executable(decal decal.cpp)
    target_link_libraries(decal PRIVATE threepp assimp::assimp glfw::glfw)
    if (imgui_FOUND)
        target_link_libraries(decal PRIVATE imgui::imgui)
    endif()
endif()
