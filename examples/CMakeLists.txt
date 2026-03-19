
include(AddExample.cmake)
include(LibConfig.cmake)

add_subdirectory(libs)

add_example(NAME "demo" LINK_IMGUI WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/fonts@data/fonts"
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

if (THREEPP_BUILD_EXAMPLE_PROJECTS)
    add_subdirectory(projects)
endif ()

if (DEFINED EMSCRIPTEN)
    file(GLOB HTML_FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/*.html")
    list(REMOVE_ITEM HTML_FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/index.html")

    set(EXAMPLE_LINKS "")
    foreach (FILE ${HTML_FILES})
        get_filename_component(NAME ${FILE} NAME_WE)
        string(APPEND EXAMPLE_LINKS "<li><a href=\"${NAME}.html\">${NAME}</a></li>\n")
    endforeach ()

    configure_file(${PROJECT_SOURCE_DIR}/cmake/index.html.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/index.html @ONLY)
endif ()
