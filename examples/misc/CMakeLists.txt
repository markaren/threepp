
add_example(NAME "lut" WEB LINK_IMGUI)
add_example(NAME "raycast" WEB)
add_example(NAME "clipping" WEB LINK_IMGUI)
add_example(NAME "morphtargets" LINK_IMGUI)
add_example(NAME "morphtargets_sphere")
add_example(NAME "multiple_scenes" WEB)
add_example(NAME "transmission" LINK_IMGUI WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr@data/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr"
)
add_example(NAME "gltf_dump")
add_example(NAME "showpiece" WEB)
