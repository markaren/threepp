# cmake/EmbedWgslPathTracer.cmake
# Embeds the WGPU path tracer's .wgsl shader sources as chunked C++ raw
# string literals at build time. Produces a single auto-generated .cpp
# file that defines the fragment globals consumed by
# WgpuPathTracerShaders.cpp's build functions.
#
# Usage:
#   include(${PROJECT_SOURCE_DIR}/cmake/EmbedWgslPathTracer.cmake)
#   embed_wgsl_path_tracer(<target>)
#
# The .wgsl source files live in
#   src/threepp/renderers/wgpu/pathtracer/shaders/
# and the basename->symbol mapping is hardcoded in
#   cmake/EmbedWgslPathTracerScript.cmake.

include_guard(GLOBAL)

function(embed_wgsl_path_tracer TARGET)
    set(_wgsl_dir "${PROJECT_SOURCE_DIR}/src/threepp/renderers/wgpu/pathtracer/shaders")
    set(_gen_dir  "${CMAKE_BINARY_DIR}/generated/threepp/renderers/wgpu/pathtracer")
    set(_script   "${PROJECT_SOURCE_DIR}/cmake/EmbedWgslPathTracerScript.cmake")
    set(_out_cpp  "${_gen_dir}/WgpuPathTracerShadersGen.cpp")

    file(MAKE_DIRECTORY "${_gen_dir}")
    file(GLOB _wgsl_files "${_wgsl_dir}/*.wgsl")

    add_custom_command(
        OUTPUT  "${_out_cpp}"
        COMMAND "${CMAKE_COMMAND}"
                "-DWGSL_DIR=${_wgsl_dir}"
                "-DOUT_CPP=${_out_cpp}"
                -P "${_script}"
        DEPENDS ${_wgsl_files} "${_script}"
        COMMENT "Embedding WGPU path-tracer WGSL shaders"
        VERBATIM
    )

    target_sources(${TARGET} PRIVATE "${_out_cpp}")
endfunction()
