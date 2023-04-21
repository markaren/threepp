
include(LibConfig.cmake)

add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp)
if (imgui_FOUND)
    target_link_libraries(demo PUBLIC imgui::imgui)
endif()

add_subdirectory(geometries)
add_subdirectory(shaders)
add_subdirectory(objects)
add_subdirectory(lights)
add_subdirectory(misc)
add_subdirectory(helpers)
add_subdirectory(loaders)
add_subdirectory(extras)
add_subdirectory(textures)

if (BULLET_FOUND)
    add_subdirectory(bullet)
endif()

if(THREEPP_BUILD_EXAMPLE_PROJECTS)
    add_subdirectory(projects)
else()
    message(AUTHOR_WARNING "Building examples, but NOT example projects. Re-run with -DTHREEPP_BUILD_EXAMPLE_PROJECTS=ON to change this behaviour.")
endif()

