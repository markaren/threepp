
include(LibConfig.cmake)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE threepp)
if (imgui_FOUND)
    target_link_libraries(demo PRIVATE imgui::imgui)
endif ()

add_subdirectory(animation)
add_subdirectory(audio)
add_subdirectory(bullet)
add_subdirectory(controls)
add_subdirectory(extras)
add_subdirectory(geometries)
add_subdirectory(helpers)
add_subdirectory(lights)
add_subdirectory(loaders)
add_subdirectory(misc)
add_subdirectory(objects)
add_subdirectory(shaders)
add_subdirectory(textures)

add_subdirectory(projects)
