
add_executable(instancing instancing.cpp)
target_link_libraries(instancing PUBLIC threepp)

add_executable(lod lod.cpp)
target_link_libraries(lod PUBLIC threepp)

add_executable(reflector reflector.cpp)
target_link_libraries(reflector PUBLIC threepp)

add_executable(water water.cpp)
target_link_libraries(water PUBLIC threepp)

add_executable(sprite sprite.cpp)
target_link_libraries(sprite PUBLIC threepp)

add_executable(points points.cpp)
target_link_libraries(points PUBLIC threepp)

if (assimp_FOUND)
    add_executable(decal decal.cpp)
    target_link_libraries(decal PUBLIC threepp assimp::assimp)
    if (imgui_FOUND)
        target_link_libraries(decal PUBLIC imgui::imgui)
    endif()
endif()
