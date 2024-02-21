
include(AddExample.cmake)
include(LibConfig.cmake)

add_subdirectory(libs)

add_example(NAME "demo" TRY_LINK_IMGUI WEB WEB_EMBED
        "../data/fonts@data/fonts"
)

add_subdirectory(animation)
add_subdirectory(audio)
add_subdirectory(controls)
add_subdirectory(extras)
add_subdirectory(geometries)
add_subdirectory(helpers)
add_subdirectory(lights)
add_subdirectory(loaders)
add_subdirectory(misc)
add_subdirectory(objects)
add_subdirectory(shaders)
add_subdirectory(textures)

add_subdirectory(projects)
