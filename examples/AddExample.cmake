
function(add_example)

    set(flags TRY_LINK_IMGUI LINK_IMGUI LINK_ASSIMP LINK_XML LINK_PHYSX WEB)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES WEB_EMBED)

    cmake_parse_arguments(arg "${flags}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (EMSCRIPTEN AND NOT arg_WEB)
        return()
    endif ()

    if (arg_LINK_IMGUI AND NOT imgui_FOUND)
        message(AUTHOR_WARNING "imgui not found, skipping '${arg_NAME}' example..")
        return()
    endif ()

    if (arg_LINK_ASSIMP AND NOT assimp_FOUND)
        message(AUTHOR_WARNING "assimp not found, skipping '${arg_NAME}' example..")
        return()
    endif ()

    if (arg_LINK_PHYSX AND NOT unofficial-omniverse-physx-sdk_FOUND)
        message(AUTHOR_WARNING "physx not found, skipping '${arg_NAME}' example..")
        return()
    endif ()


    if (NOT arg_SOURCES)
        add_executable("${arg_NAME}" "${arg_NAME}.cpp")
    else ()
        add_executable("${arg_NAME}" "${arg_SOURCES}")
    endif ()

    target_link_libraries("${arg_NAME}" PRIVATE threepp)

    if ((arg_TRY_LINK_IMGUI OR arg_LINK_IMGUI) AND imgui_FOUND)
        target_link_libraries("${arg_NAME}" PRIVATE imgui::imgui)
    endif ()

    if (arg_LINK_ASSIMP AND assimp_FOUND)
        target_link_libraries("${arg_NAME}" PRIVATE assimp::assimp)
    endif ()

    if (arg_LINK_PHYSX AND unofficial-omniverse-physx-sdk_FOUND)
        target_link_libraries("${arg_NAME}" PRIVATE unofficial::omniverse-physx-sdk::sdk)
    endif ()

    if (DEFINED EMSCRIPTEN)

        set(LINK_FLAGS " --bind -sUSE_GLFW=3 -sGL_DEBUG=1 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sFULL_ES3 -sASSERTIONS -sALLOW_MEMORY_GROWTH -sNO_DISABLE_EXCEPTION_CATCHING -sWASM=1")
        if (arg_WEB_EMBED)
            foreach (path ${arg_WEB_EMBED})
                set(LINK_FLAGS "${LINK_FLAGS} --embed-file ${path}")
            endforeach ()
        endif ()

        set_target_properties("${arg_NAME}"
                PROPERTIES SUFFIX ".html"
                LINK_FLAGS "${LINK_FLAGS}")

    endif (DEFINED EMSCRIPTEN)

endfunction()
