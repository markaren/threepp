
include(AddExample.cmake)
include(LibConfig.cmake)

add_subdirectory(libs)

add_example(NAME "demo" LINK_IMGUI WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/fonts@data/fonts"
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

if (THREEPP_WITH_WGPU)
    add_subdirectory(wgpu)
endif ()

if (THREEPP_WITH_VULKAN)
    add_subdirectory(vulkan)
endif ()

if (THREEPP_BUILD_EXAMPLE_PROJECTS)
    add_subdirectory(projects)
endif ()

if (DEFINED EMSCRIPTEN)
    # Build the example listing from the names registered by add_example(), not
    # from a glob of the output dir: the *.html files are produced at build time,
    # so a configure-time glob is empty on a clean build (e.g. in CI).
    get_property(WEB_EXAMPLES GLOBAL PROPERTY THREEPP_WEB_EXAMPLES)
    list(SORT WEB_EXAMPLES)

    set(EXAMPLE_LINKS "")
    foreach (NAME ${WEB_EXAMPLES})
        string(APPEND EXAMPLE_LINKS "<li><a href=\"${NAME}.html\">${NAME}</a></li>\n")
    endforeach ()

    configure_file(${PROJECT_SOURCE_DIR}/cmake/index.html.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/index.html @ONLY)
endif ()
