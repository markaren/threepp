
add_example(NAME "cubemap")
add_example(NAME "hdr_envmap")
add_example(NAME "texture3d" LINK_IMGUI)
add_example(NAME "texture2d" WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/textures/crate.gif@data/textures/crate.gif"
        "${THREEPP_DATA_DIR}/textures/checker.png@data/textures/checker.png"
        "${THREEPP_DATA_DIR}/textures/brick_bump.jpg@data/textures/brick_bump.jpg"
)
add_example(NAME "data_texture" WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/textures/checker.png@data/textures/checker.png"
        "${THREEPP_DATA_DIR}/textures/crate.gif@data/textures/crate.gif"
)
add_example(NAME "imgui_framebuffer" LINK_IMGUI WEB)

add_example(NAME "depth_texture" WEB)
