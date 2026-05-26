
if (THREEPP_WITH_WGPU)
    add_example(NAME "webtide" SOURCES main.cpp LINK_IMGUI WEB WEB_EMBED
            "${THREEPP_DATA_DIR}/textures/sand.jpg@data/textures/"
    )
    if (NOT EMSCRIPTEN)
        target_include_directories(webtide PRIVATE "${WGPU_INCLUDE_DIR}")
        target_link_libraries(webtide PRIVATE ${WGPU_LIBRARY})
    endif ()
endif ()
