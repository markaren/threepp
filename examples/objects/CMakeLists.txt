
add_executable(instancing instancing.cpp)
target_link_libraries(instancing PUBLIC threepp glfw::glfw)

add_executable(lod lod.cpp)
target_link_libraries(lod PUBLIC threepp glfw::glfw)

add_executable(water water.cpp)
target_link_libraries(water PUBLIC threepp glfw::glfw)

add_executable(sprite sprite.cpp)
target_link_libraries(sprite PUBLIC threepp glfw::glfw)

add_executable(points points.cpp)
target_link_libraries(points PUBLIC threepp glfw::glfw)

if (assimp_FOUND)
    add_executable(decal decal.cpp)
    target_link_libraries(decal PUBLIC threepp assimp::assimp glfw::glfw)
    if (imgui_FOUND)
        target_link_libraries(decal PUBLIC imgui::imgui)
    endif()
endif()
