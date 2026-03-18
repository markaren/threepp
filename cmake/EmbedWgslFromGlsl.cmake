# cmake/EmbedWgslFromGlsl.cmake
# Provides embed_wgsl_from_glsl() -- a function that translates Vulkan GLSL
# shaders to WGSL via naga at build time, then embeds the results as C++
# string literals.
#
# Requires: NAGA_EXECUTABLE to be set (from FetchNaga.cmake).
#
# Usage:
#   embed_wgsl_from_glsl(<target> <name> <vert.glsl> <frag.glsl>)

function(embed_wgsl_from_glsl TARGET NAME VERT_GLSL FRAG_GLSL)
    if(NOT NAGA_EXECUTABLE)
        message(FATAL_ERROR "embed_wgsl_from_glsl: NAGA_EXECUTABLE not set. Include FetchNaga.cmake first.")
    endif()

    set(_wgsl_dir "${CMAKE_BINARY_DIR}/wgsl_intermediate")
    set(_gen_dir  "${CMAKE_BINARY_DIR}/generated/threepp/renderers/dawn/wgsl")
    file(MAKE_DIRECTORY "${_wgsl_dir}")
    file(MAKE_DIRECTORY "${_gen_dir}")

    set(_vert_wgsl "${_wgsl_dir}/${NAME}.vert.wgsl")
    set(_frag_wgsl "${_wgsl_dir}/${NAME}.frag.wgsl")
    set(_out_hpp   "${_gen_dir}/${NAME}_wgsl.hpp")
    set(_out_cpp   "${_gen_dir}/${NAME}_wgsl.cpp")

    # Step 1: Translate vertex GLSL -> WGSL via naga
    add_custom_command(
        OUTPUT "${_vert_wgsl}"
        COMMAND "${NAGA_EXECUTABLE}" "${VERT_GLSL}" "${_vert_wgsl}"
        DEPENDS "${VERT_GLSL}" "${NAGA_EXECUTABLE}"
        COMMENT "naga: ${NAME}.vert -> WGSL"
        VERBATIM
    )

    # Step 2: Translate fragment GLSL -> WGSL via naga
    add_custom_command(
        OUTPUT "${_frag_wgsl}"
        COMMAND "${NAGA_EXECUTABLE}" "${FRAG_GLSL}" "${_frag_wgsl}"
        DEPENDS "${FRAG_GLSL}" "${NAGA_EXECUTABLE}"
        COMMENT "naga: ${NAME}.frag -> WGSL"
        VERBATIM
    )

    # Step 3: Combine + embed the two WGSL files into C++ source/header
    add_custom_command(
        OUTPUT "${_out_hpp}" "${_out_cpp}"
        COMMAND "${CMAKE_COMMAND}"
            -DVERT_WGSL=${_vert_wgsl}
            -DFRAG_WGSL=${_frag_wgsl}
            -DOUT_HPP=${_out_hpp}
            -DOUT_CPP=${_out_cpp}
            -DSHADER_NAME=${NAME}
            -P "${PROJECT_SOURCE_DIR}/cmake/EmbedWgslFile.cmake"
        DEPENDS "${_vert_wgsl}" "${_frag_wgsl}"
                "${PROJECT_SOURCE_DIR}/cmake/EmbedWgslFile.cmake"
        COMMENT "Embedding ${NAME} WGSL as C++ string literals"
        VERBATIM
    )

    # Add the generated source to the target
    target_sources(${TARGET} PRIVATE "${_out_cpp}")

    # Set compile definition so code can conditionally use the pre-translated WGSL
    string(TOUPPER "${NAME}" _name_upper)
    target_compile_definitions(${TARGET} PRIVATE "THREEPP_WGSL_${_name_upper}=1")

    message(STATUS "embed_wgsl_from_glsl: ${NAME} (${VERT_GLSL} + ${FRAG_GLSL})")
endfunction()
