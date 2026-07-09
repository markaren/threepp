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
    cmake_parse_arguments(ARG "" "VARIANT_SUFFIX" "DEFINES" ${ARGN})

    get_filename_component(_name "${shader_src}" NAME)
    get_filename_component(_src_dir "${shader_src}" DIRECTORY)
    set(_gen_dir "${CMAKE_BINARY_DIR}/generated/threepp/renderers/vulkan/shaders")
    if (ARG_VARIANT_SUFFIX)
        set(_out_header "${_gen_dir}/${_name}.${ARG_VARIANT_SUFFIX}.spv.h")
    else()
        set(_out_header "${_gen_dir}/${_name}.spv.h")
    endif()
    file(MAKE_DIRECTORY "${_gen_dir}")

    set(_define_flags "")
    foreach(_d ${ARG_DEFINES})
        list(APPEND _define_flags "-D${_d}")
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

    # Depend on every .glsl/.h that sits beside the shader source, whatever its
    # name. Shaders in this dir #include each other freely (vulkan_shared.h,
    # probe_common.glsl, the deferred_shade_NN_*.glsl split, ...) and any of
    # them can be transitively pulled into any shader here, so a glob is the
    # only dependency list that can't go stale as files are added/renamed.
    # Cheap (<1s extra glslangValidator invocations on the unrelated shaders
    # that don't actually include the touched file). CONFIGURE_DEPENDS makes
    # the generator re-run the glob at build time (ninja re-checks it), so
    # adding a new .glsl/.h picks it up without a manual reconfigure.
    # Out-of-tree shaders (e.g. example inference kernels) have no .glsl/.h
    # neighbours, so the glob returns empty there — no DEPENDS on anything,
    # same as before.
    file(GLOB _extra_deps CONFIGURE_DEPENDS "${_src_dir}/*.glsl" "${_src_dir}/*.h")

    add_custom_command(
        OUTPUT  "${_out_header}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3
                ${_opt_flags}
                "-I${_src_dir}"
                ${_define_flags}
                --vn "${var_name}"
                "${shader_src}"
                -o   "${_out_header}"
        DEPENDS "${shader_src}" ${_extra_deps}
        COMMENT "Compiling Vulkan shader ${_name} -> ${var_name}"
        VERBATIM)

    target_sources(${target} PRIVATE "${_out_header}")
    target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")

    set(${out_header_var} "${_out_header}" PARENT_SCOPE)
endfunction()
