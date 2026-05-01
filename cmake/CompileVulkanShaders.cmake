# compile_vulkan_shader(<target> <shader_src> <var_name> <out_header_var>)
#
# Compile a GLSL shader to SPIR-V using glslangValidator (from VULKAN_SDK
# or vcpkg's glslang port) and emit a C++ header with the SPIR-V embedded
# as a `static const uint32_t <var_name>[]` array via glslangValidator's
# --vn option. The generated header is placed under the project build dir
# at threepp/renderers/vulkan/shaders/<basename>.spv.h and added to
# <target> as a generated source dependency.
#
# Out arg <out_header_var> receives the generated header's absolute path
# so callers can pin it as a target dependency.

if (NOT GLSLANG_VALIDATOR)
    find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator glslang-validator
        HINTS
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VULKAN_SDK}/bin"
        DOC "Path to glslangValidator (Vulkan SDK or vcpkg glslang)")
    if (NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR
            "glslangValidator not found. Install the Vulkan SDK or `vcpkg install glslang` "
            "and ensure glslangValidator is on PATH or VULKAN_SDK is set.")
    endif ()
    message(STATUS "Vulkan shader compiler: ${GLSLANG_VALIDATOR}")
endif ()

function(compile_vulkan_shader target shader_src var_name out_header_var)
    get_filename_component(_name "${shader_src}" NAME)
    set(_gen_dir "${CMAKE_BINARY_DIR}/generated/threepp/renderers/vulkan/shaders")
    set(_out_header "${_gen_dir}/${_name}.spv.h")

    file(MAKE_DIRECTORY "${_gen_dir}")

    add_custom_command(
        OUTPUT  "${_out_header}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3
                --vn "${var_name}"
                "${shader_src}"
                -o   "${_out_header}"
        DEPENDS "${shader_src}"
        COMMENT "Compiling Vulkan shader ${_name} -> ${var_name}"
        VERBATIM)

    target_sources(${target} PRIVATE "${_out_header}")
    target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")

    set(${out_header_var} "${_out_header}" PARENT_SCOPE)
endfunction()
