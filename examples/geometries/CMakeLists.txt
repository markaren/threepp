
add_executable(geometries geometries.cpp)
target_link_libraries(geometries PRIVATE threepp)

if (DEFINED EMSCRIPTEN)
    set_target_properties(geometries
            PROPERTIES SUFFIX ".html"
            LINK_FLAGS " --bind -s USE_GLFW=3 -s MIN_WEBGL_VERSION=2 -s MAX_WEBGL_VERSION=2 -s FULL_ES3 -s -s WASM=1 --embed-file ${CMAKE_SOURCE_DIR}/data/textures/uv_grid_opengl.jpg@/data/textures/uv_grid_opengl.jpg")
endif (DEFINED EMSCRIPTEN)

add_executable(lathe_geometry lathe_geometry.cpp)
target_link_libraries(lathe_geometry PRIVATE threepp)

add_executable(tube_geometry tube_geometry.cpp)
target_link_libraries(tube_geometry PRIVATE threepp)

add_executable(shape_geometry shape_geometry.cpp)
target_link_libraries(shape_geometry PRIVATE threepp)

add_executable(convex_geometry convex_geometry.cpp)
target_link_libraries(convex_geometry PRIVATE threepp)

add_executable(heightmap heightmap.cpp)
target_link_libraries(heightmap PRIVATE threepp)

add_executable(dynamic dynamic.cpp)
target_link_libraries(dynamic PRIVATE threepp)

if (imgui_FOUND)
    add_executable(box_geometry box_geometry.cpp)
    target_link_libraries(box_geometry PRIVATE threepp imgui::imgui)

    add_executable(cylinder_geometry cylinder_geometry.cpp)
    target_link_libraries(cylinder_geometry PRIVATE threepp imgui::imgui)

    add_executable(plane_geometry plane_geometry.cpp)
    target_link_libraries(plane_geometry PRIVATE threepp imgui::imgui)

    add_executable(sphere_geometry sphere_geometry.cpp)
    target_link_libraries(sphere_geometry PRIVATE threepp imgui::imgui)
endif ()
