
add_example(NAME "cubemap")
add_example(NAME "texture3d")
add_example(NAME "texture2d" WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/textures/crate.gif@data/textures/crate.gif"
        "${PROJECT_SOURCE_DIR}/data/textures/checker.png@data/textures/checker.png"
        "${PROJECT_SOURCE_DIR}/data/textures/brick_bump.jpg@data/textures/brick_bump.jpg"
)
add_example(NAME "data_texture" WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/textures/checker.png@data/textures/checker.png"
        "${PROJECT_SOURCE_DIR}/data/textures/crate.gif@data/textures/crate.gif"
)
add_example(NAME "imgui_framebuffer" LINK_IMGUI WEB)
