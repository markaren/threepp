
function(add_example)

    set(flags LINK_IMGUI LINK_ASSIMP LINK_XML LINK_PHYSX LINK_MESHOPT WEB)
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

    if (arg_LINK_PHYSX AND (NOT TARGET unofficial::omniverse-physx-sdk::sdk))
        message(AUTHOR_WARNING "physx not found, skipping '${arg_NAME}' example..")
        return()
    endif ()

    if (arg_LINK_MESHOPT AND (NOT THREEPP_WITH_MESHOPT))
        message(AUTHOR_WARNING "meshoptimizer not enabled, skipping '${arg_NAME}' example..")
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

    if (arg_LINK_PHYSX)
        target_link_libraries("${arg_NAME}" PRIVATE unofficial::omniverse-physx-sdk::sdk)
        if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-library)
            add_custom_command(TARGET "${arg_NAME}" POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
                    $<TARGET_FILE_DIR:${arg_NAME}>)
        endif ()
        if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-device-library)
            add_custom_command(TARGET "${arg_NAME}" POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
                    $<TARGET_FILE_DIR:${arg_NAME}>)
        endif ()
    endif ()



    if (DEFINED EMSCRIPTEN)

        target_compile_definitions(${arg_NAME} PRIVATE DATA_FOLDER="data")

        set(LINK_FLAGS " --bind -sUSE_GLFW=3 -sASSERTIONS -sALLOW_MEMORY_GROWTH -sNO_DISABLE_EXCEPTION_CATCHING -sWASM=1 -sEXPORTED_RUNTIME_METHODS=[requestFullscreen]")
        if (THREEPP_WITH_WGPU)
            set(LINK_FLAGS "${LINK_FLAGS} --use-port=emdawnwebgpu -sASYNCIFY")
        else ()
            set(LINK_FLAGS "${LINK_FLAGS} -sGL_DEBUG=1 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sFULL_ES3")
        endif ()
        set(LINK_FLAGS "${LINK_FLAGS} --shell-file \"${PROJECT_SOURCE_DIR}/examples/emshell.html\"")
        if (arg_WEB_EMBED)
            foreach (path ${arg_WEB_EMBED})
                set(LINK_FLAGS "${LINK_FLAGS} --embed-file \"${path}\"")
            endforeach ()
        endif ()

        set_target_properties("${arg_NAME}"
                PROPERTIES SUFFIX ".html"
                LINK_FLAGS "${LINK_FLAGS}")

    else ()
        target_compile_definitions(${arg_NAME} PRIVATE DATA_FOLDER="${THREEPP_DATA_DIR}")
        target_compile_definitions(${arg_NAME} PRIVATE PROJECT_FOLDER="${PROJECT_SOURCE_DIR}")
    endif (DEFINED EMSCRIPTEN)

endfunction()
