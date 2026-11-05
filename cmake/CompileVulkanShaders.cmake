# compile_vulkan_shader(<target> <shader_src> <var_name> <out_header_var>)
#
# Compile a GLSL shader to SPIR-V with glslangValidator (from VULKAN_SDK or
# vcpkg's glslang port), optimize it with spirv-opt (same SDK / vcpkg's
# spirv-tools port) and emit a C++ header with the SPIR-V embedded as a
# `const uint32_t <var_name>[]` array (cmake/EmbedSpirv.cmake). The generated
# header is placed under the project build dir at
# threepp/renderers/vulkan/shaders/<basename>.spv.h and added to <target> as a
# generated source dependency.
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

if (NOT SPIRV_OPT)
    get_filename_component(_glslang_dir "${GLSLANG_VALIDATOR}" DIRECTORY)
    find_program(SPIRV_OPT
        NAMES spirv-opt
        HINTS
            "${_glslang_dir}"
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VULKAN_SDK}/bin"
        DOC "Path to spirv-opt (Vulkan SDK or vcpkg spirv-tools[tools])")
    if (NOT SPIRV_OPT)
        message(FATAL_ERROR
            "spirv-opt not found. Install the Vulkan SDK or `vcpkg install spirv-tools[tools]` "
            "and ensure spirv-opt is on PATH or VULKAN_SDK is set.")
    endif ()
    message(STATUS "Vulkan shader optimizer: ${SPIRV_OPT}")
endif ()

# Captured here, at include time: inside the function CMAKE_CURRENT_LIST_DIR
# is the caller's directory.
set(_THREEPP_EMBED_SPIRV "${CMAKE_CURRENT_LIST_DIR}/EmbedSpirv.cmake")

# The optimization recipe, spelled out pass by pass instead of asking
# glslangValidator for -Os.
#
# This is the pass list glslang's -Os ran up to glslang 16.4 (SpvTools.cpp,
# SpirvToolsTransform), reproduced through spirv-opt's command line, with
# one addition explained below. glslang 16.5.0 (2026-07-31, in Vulkan SDK
# 1.4.363) replaced that list with spirv-opt's own size recipe
# (RegisterSizePasses), which fully unrolls loops and runs SSA construction
# on the whole inlined module before any dead-code pass. On
# deferred_shade.comp that recipe runs out of SPIR-V ids ("ID overflow. Try
# running compact-ids." at the universal limit of 4,194,303) and the build
# stops; with the limit lifted it takes three minutes per compile. Pinning
# the recipe here makes the embedded SPIR-V the same whichever SDK builds it.
#
# The addition, --private-to-local: spirv-opt's command line hands every
# pass preserve_interface = true with no flag to turn it off (tools/opt/
# opt.cpp), so its aggressive DCE keeps every global the entry point lists
# and every store into one, where glslang's library call (preserve_interface
# = false) dropped the unused ones. --private-to-local, which spirv-opt's own
# -O and -Os recipes run at the same point, turns the shaders' Private
# globals (the gXxx state of the deferred shade) into function-scope
# variables the local passes then promote to SSA values or delete, and
# --remove-unused-interface-variables drops the bindings nothing references.
# Against the old -Os the shade comes out 2% fewer instructions and 80%
# fewer stores, the no-ray-query variant marginally smaller, and the rest
# of the shaders byte-identical or down to the interface list's order.
#
# Why optimize at all: the passes are FP-bit-preserving (no reassociation,
# no reordering of floating-point arithmetic), so the image is unchanged,
# and the inlined module is what the driver compiles fastest. Measured
# 2026-10-14 on deferred_shade.comp (RTX 4070, cold driver cache): optimized
# 3.7 MB, 160 s; unoptimized 0.6 MB, 405 s; the output is byte-identical
# either way. So the smaller module is not the cheaper one, and -Os stays.
#
# Skipped for Debug builds so shader stepping in Nsight / RenderDoc sees
# unoptimized SPIR-V (spirv-opt with no passes writes the module unchanged);
# enabled for the optimized configs (Release, RelWithDebInfo, MinSizeRel).
# Gated with $<CONFIG>, not CMAKE_BUILD_TYPE, so multi-config generators
# (Visual Studio, which scikit-build-core uses for Windows wheels) also
# optimize; a single-config build with empty/unset build type stays
# unoptimized as before. Note the output header is shared across configs, so
# switching config in an existing build tree reuses the last-built SPIR-V
# until a dependency touches it.
#
# --skip-validation: glslang's -Os never validated either, and the validator
# rejects the lights UBO's scalar-layout arrays unless told the device
# enables uniformBufferStandardLayout (which the renderer does).
set(_THREEPP_SPIRV_OPT_PASSES
    --wrap-opkill
    --eliminate-dead-branches
    --merge-return
    --inline-entry-points-exhaustive
    --eliminate-dead-functions
    --private-to-local
    --remove-unused-interface-variables
    --scalar-replacement=100
    --convert-local-access-chains
    --eliminate-local-single-block
    --eliminate-local-single-store
    --simplify-instructions
    --eliminate-dead-code-aggressive
    --vector-dce
    --eliminate-dead-inserts
    --eliminate-dead-code-aggressive
    --eliminate-dead-branches
    --merge-blocks
    --eliminate-local-multi-store
    --if-conversion
    --simplify-instructions
    --eliminate-dead-code-aggressive
    --vector-dce
    --eliminate-dead-inserts
    --redundancy-elimination
    --eliminate-dead-input-components
    --remove-unused-interface-variables
    --eliminate-dead-code-aggressive
    --cfg-cleanup)

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
        set(_stem "${_gen_dir}/${_name}.${ARG_VARIANT_SUFFIX}")
    else()
        set(_stem "${_gen_dir}/${_name}")
    endif()
    set(_out_header "${_stem}.spv.h")
    set(_raw_spv "${_stem}.raw.spv")
    set(_opt_spv "${_stem}.spv")
    file(MAKE_DIRECTORY "${_gen_dir}")

    set(_define_flags "")
    foreach(_d ${ARG_DEFINES})
        list(APPEND _define_flags "-D${_d}")
    endforeach()

    # $<SEMICOLON> keeps the list inside the genex until it is evaluated;
    # a bare ';' is split by COMMAND_EXPAND_LISTS first and tears the genex
    # apart ("$<1:--wrap-opkill" ... "--cfg-cleanup>").
    string(REPLACE ";" "$<SEMICOLON>" _opt_pass_list "${_THREEPP_SPIRV_OPT_PASSES}")
    set(_opt_passes "$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:${_opt_pass_list}>")

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
        BYPRODUCTS "${_raw_spv}" "${_opt_spv}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3
                "-I${_src_dir}"
                ${_define_flags}
                "${shader_src}"
                -o   "${_raw_spv}"
        COMMAND "${SPIRV_OPT}" --target-env=vulkan1.3 --skip-validation
                ${_opt_passes}
                "${_raw_spv}"
                -o   "${_opt_spv}"
        COMMAND "${CMAKE_COMMAND}"
                "-DIN=${_opt_spv}"
                "-DOUT=${_out_header}"
                "-DVAR=${var_name}"
                -P "${_THREEPP_EMBED_SPIRV}"
        DEPENDS "${shader_src}" ${_extra_deps} "${_THREEPP_EMBED_SPIRV}"
        COMMENT "Compiling Vulkan shader ${_name} -> ${var_name}"
        VERBATIM
        # Required so the _opt_passes genex expands to one argument per pass,
        # and vanishes (instead of becoming a literal "" argument) in configs
        # where it evaluates empty.
        COMMAND_EXPAND_LISTS)

    target_sources(${target} PRIVATE "${_out_header}")
    target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")

    set(${out_header_var} "${_out_header}" PARENT_SCOPE)
endfunction()
