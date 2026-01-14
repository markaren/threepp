
add_example(NAME "stl_loader" WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/models/stl/pr2_head_pan.stl@data/models/stl/pr2_head_pan.stl"
)
add_example(NAME "obj_loader" WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/models/obj/female02@data/models/obj/female02"
        "${PROJECT_SOURCE_DIR}/data/textures/uv_grid_opengl.jpg@data/textures/uv_grid_opengl.jpg"
)
add_example(NAME "assimp_loader" LINK_ASSIMP)
add_example(NAME "urdf_loader" LINK_ASSIMP LINK_IMGUI)
add_example(NAME "urdf_loader_obj" LINK_IMGUI)
add_example(NAME "urdf_loader_simple" LINK_IMGUI)

if (THREEPP_WITH_SVG)
    add_example(NAME "svg_loader" LINK_IMGUI)
endif ()
