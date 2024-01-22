
add_example(NAME "geometries" WEB WEB_EMBED "${CMAKE_SOURCE_DIR}/data/textures/uv_grid_opengl.jpg@/data/textures/uv_grid_opengl.jpg")
add_example(NAME "lathe_geometry")
add_example(NAME "tube_geometry")
add_example(NAME "shape_geometry")
add_example(NAME "convex_geometry")
add_example(NAME "heightmap")
add_example(NAME "dynamic")

add_example(NAME "box_geometry" LINK_IMGUI)
add_example(NAME "cylinder_geometry" LINK_IMGUI)
add_example(NAME "plane_geometry" LINK_IMGUI)
add_example(NAME "sphere_geometry" LINK_IMGUI)
