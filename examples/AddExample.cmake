
function(add_example)

    set(flags LINK_IMGUI LINK_ASSIMP LINK_XML WEB)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES WEB_EMBED)

    cmake_parse_arguments(arg "${flags}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (EMSCRIPTEN AND NOT arg_WEB)
        return()
    endif ()

    if (arg_LINK_ASSIMP AND (NOT TARGET assimp::assimp))
        message(AUTHOR_WARNING "assimp not found, skipping '${arg_NAME}' example..")
        return()
    endif ()

    if (NOT arg_SOURCES)
        add_executable("${arg_NAME}" "${arg_NAME}.cpp")
    else ()
        add_executable("${arg_NAME}" "${arg_SOURCES}")
    endif ()

    target_link_libraries("${arg_NAME}" PRIVATE threepp)

    if (arg_LINK_IMGUI)
        target_link_libraries("${arg_NAME}" PRIVATE imgui::imgui)
    endif ()

    if (arg_LINK_ASSIMP)
        target_link_libraries("${arg_NAME}" PRIVATE assimp::assimp)
    endif ()

    target_compile_definitions(${arg_NAME} PRIVATE DATA_FOLDER="${PROJECT_SOURCE_DIR}/data")

    if (DEFINED EMSCRIPTEN)

        set(LINK_FLAGS " --bind -sUSE_GLFW=3 -sGL_DEBUG=1 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sFULL_ES3 -sASSERTIONS -sALLOW_MEMORY_GROWTH -sNO_DISABLE_EXCEPTION_CATCHING -sWASM=1")
        if (arg_WEB_EMBED)
            foreach (path ${arg_WEB_EMBED})
                set(LINK_FLAGS "${LINK_FLAGS} --embed-file \"${path}\"")
            endforeach ()
        endif ()

        set_target_properties("${arg_NAME}"
                PROPERTIES SUFFIX ".html"
                LINK_FLAGS "${LINK_FLAGS}")

    endif (DEFINED EMSCRIPTEN)

endfunction()
