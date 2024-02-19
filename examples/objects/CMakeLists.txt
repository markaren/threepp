
add_example(NAME "instancing" TRY_LINK_IMGUI WEB WEB_EMBED
        "../data/fonts@data/fonts"
)
if (TARGET "instancing")
    target_include_directories("instancing" PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()

add_example(NAME "sprite")
add_example(NAME "lod" WEBWEB_EMBED
        "../data/fonts@data/fonts"
)
add_example(NAME "points" WEB)

add_example(NAME "bones" LINK_IMGUI)
add_example(NAME "decal" LINK_IMGUI LINK_ASSIMP)
add_example(NAME "water" TRY_LINK_IMGUI WEB WEB_EMBED
        "../data/textures/waternormals.jpg@data/textures/waternormals.jpg"
)

add_example(NAME "particle_system" TRY_LINK_IMGUI)
