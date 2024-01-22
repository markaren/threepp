
add_example(NAME "instancing" LINK_IMGUI)
target_include_directories(instancing PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")

add_example(NAME "lod")
add_example(NAME "sprite")
add_example(NAME "points")
add_example(NAME "water" TRY_LINK_IMGUI)
add_example(NAME "bones" LINK_IMGUI)
add_example(NAME "decal" LINK_IMGUI LINK_ASSIMP)

