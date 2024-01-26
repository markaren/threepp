
add_example(NAME "stl_loader" WEB WEB_EMBED
        "../data/models/stl/pr2_head_pan.stl@data/models/stl/pr2_head_pan.stl"
)
add_example(NAME "obj_loader" WEB WEB_EMBED
        "../data/models/obj/female02@data/models/obj/female02"
        "../data/textures/uv_grid_opengl.jpg@data/textures/uv_grid_opengl.jpg"
)
add_example(NAME "assimp_loader" LINK_ASSIMP)

if (THREEPP_WITH_SVG)
    add_example(NAME "svg_loader" TRY_LINK_IMGUI)
endif ()
