
add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp)
if (imgui_FOUND)
    target_link_libraries(demo PUBLIC imgui::imgui)
    target_compile_definitions(demo PRIVATE "HAS_IMGUI")
endif()

add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PUBLIC threepp)

add_executable(points_demo points_demo.cpp)
target_link_libraries(points_demo PUBLIC threepp)

add_subdirectory(geometries)
add_subdirectory(materials)
add_subdirectory(shaders)
add_subdirectory(objects)
add_subdirectory(lights)
add_subdirectory(misc)
add_subdirectory(helpers)
add_subdirectory(loaders)

add_subdirectory(projects)

if (BULLET_FOUND)
    add_subdirectory(bullet)
endif()
