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
    # Optional 5th positional arg: variant suffix used in the output filename
    # so two compiles of the same source (e.g. raygen.rgen with/without SER)
    # don't collide. Optional 6th+ args become `-D<MACRO>` flags to
    # glslangValidator.
    cmake_parse_arguments(ARG "" "VARIANT_SUFFIX" "DEFINES;INCLUDE_DIRS" ${ARGN})

    get_filename_component(_name "${shader_src}" NAME)
    get_filename_component(_src_dir "${shader_src}" DIRECTORY)
    set(_gen_dir "${CMAKE_BINARY_DIR}/generated/threepp/renderers/vulkan/shaders")
    if (ARG_VARIANT_SUFFIX)
        set(_out_header "${_gen_dir}/${_name}.${ARG_VARIANT_SUFFIX}.spv.h")
    else()
        set(_out_header "${_gen_dir}/${_name}.spv.h")
    endif()
    set(_shared_header "${_src_dir}/vulkan_shared.h")
    # raygen.rgen #includes shade_primary.glsl; we list it here as a global
    # dep so edits to shade_primary trigger a raygen rebuild. Cheap (<1s
    # extra glslangValidator invocations on the unrelated shaders that
    # don't include it).
    set(_shade_primary "${_src_dir}/shade_primary.glsl")

    file(MAKE_DIRECTORY "${_gen_dir}")

    set(_define_flags "")
    foreach(_d ${ARG_DEFINES})
        list(APPEND _define_flags "-D${_d}")
    endforeach()

    # extra include dirs for #include inside the shader (e.g. shared single-source
    # env-dynamics files that also compile as C++). Listed as DEPENDS too so edits
    # to an included file trigger a recompile.
    set(_include_flags "")
    set(_include_deps "")
    foreach(_dir ${ARG_INCLUDE_DIRS})
        list(APPEND _include_flags "-I${_dir}")
        file(GLOB _dir_files "${_dir}/*")
        list(APPEND _include_deps ${_dir_files})
    endforeach()

    # SPIR-V optimization. -Os runs glslang's spirv-opt size recipe (dead-code
    # elimination, CSE, function inlining, loop-invariant code motion, dead-
    # branch elimination). These passes are FP-bit-preserving — they do not
    # reassociate or reorder floating-point arithmetic — so the rendered image
    # is unchanged; they shrink the embedded SPIR-V and hand the driver a
    # cleaner starting point. Skipped for Debug builds so shader stepping in
    # Nsight / RenderDoc sees unoptimized SPIR-V; enabled for the optimized
    # configs (Release, RelWithDebInfo, MinSizeRel). Empty/unset build type
    # keeps the prior unoptimized behaviour.
    set(_opt_flags "")
    if (CMAKE_BUILD_TYPE STREQUAL "Release"
            OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
            OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        list(APPEND _opt_flags "-Os")
    endif ()

    # Only depend on the shared headers when they actually sit beside the shader
    # source. Core shaders live next to vulkan_shared.h / shade_primary.glsl;
    # out-of-tree shaders (e.g. example inference kernels) don't, and a DEPENDS
    # on a missing file would break the build graph.
    set(_extra_deps "")
    if (EXISTS "${_shared_header}")
        list(APPEND _extra_deps "${_shared_header}")
    endif ()
    if (EXISTS "${_shade_primary}")
        list(APPEND _extra_deps "${_shade_primary}")
    endif ()

    add_custom_command(
        OUTPUT  "${_out_header}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3
                ${_opt_flags}
                "-I${_src_dir}"
                ${_include_flags}
                ${_define_flags}
                --vn "${var_name}"
                "${shader_src}"
                -o   "${_out_header}"
        DEPENDS "${shader_src}" ${_extra_deps} ${_include_deps}
        COMMENT "Compiling Vulkan shader ${_name} -> ${var_name}"
        VERBATIM)

    target_sources(${target} PRIVATE "${_out_header}")
    target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")

    set(${out_header_var} "${_out_header}" PARENT_SCOPE)
endfunction()
