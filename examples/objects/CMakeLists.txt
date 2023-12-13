
add_executable(instancing instancing.cpp)
target_link_libraries(instancing PRIVATE threepp)
target_include_directories(instancing PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
if (imgui_FOUND)
    target_link_libraries(instancing PRIVATE imgui::imgui)
endif ()

add_executable(lod lod.cpp)
target_link_libraries(lod PRIVATE threepp)

add_executable(water water.cpp)
target_link_libraries(water PRIVATE threepp)

add_executable(sprite sprite.cpp)
target_link_libraries(sprite PRIVATE threepp)

add_executable(points points.cpp)
target_link_libraries(points PRIVATE threepp)

if (imgui_FOUND)
    add_executable(bones bones.cpp)
    target_link_libraries(bones PRIVATE threepp imgui::imgui)

    target_link_libraries(water PRIVATE imgui::imgui)
endif ()

if (assimp_FOUND)
    add_executable(decal decal.cpp)
    target_link_libraries(decal PRIVATE threepp assimp::assimp)
    if (imgui_FOUND)
        target_link_libraries(decal PRIVATE imgui::imgui)
    endif ()
endif ()
