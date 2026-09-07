# RLtools (https://rl.tools) reinforcement-learning demos.
# Enabled by configuring with -DTHREEPP_WITH_RLTOOLS=ON, which fetches the
# header-only library and defines the `rltools` interface target.
#
#   rltools_pendulum — one SAC policy, single env, visualized field (preview).
#   rltools_swarm    — one SAC learner fed by 64 environments collected in PARALLEL
#                      (the field IS the training data); Stage 1 of the parallel-RL
#                      direction (CPU; a Vulkan-compute rollout is the planned Stage 2).

# Optional aggressive optimization of the (template-heavy, compute-bound) RLtools
# trainer. RLtools' own CMake only adds -O3/-ffast-math/-march=native for GNU/Clang
# and NOTHING for MSVC. Turning this ON applies the best flags for the detected
# compiler. NOTE: -march=native bakes the build machine's ISA in, so it is OFF by
# default. The big win is the LLVM frontend (clang-cl): configure with
# -DCMAKE_CXX_COMPILER=clang-cl from a vcvars prompt.
option(THREEPP_RLTOOLS_FAST "Aggressively optimize the RLtools trainer TUs (march=native; build-machine specific)" OFF)

# Apply RLtools linkage + the MSVC inlining fix (and optional FAST flags) to a target.
function(_rltools_configure tgt)
    target_link_libraries("${tgt}" PRIVATE rltools)
    find_package(Threads REQUIRED)
    target_link_libraries("${tgt}" PRIVATE Threads::Threads)

    set(_clang_cl FALSE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(_clang_cl TRUE)
    endif ()

    # RLtools leans ENTIRELY on the compiler inlining its tiny templated matrix kernels.
    #   * /bigobj — heavy template instantiation (too many sections otherwise).
    #   * /Ob2    — full inlining. RelWithDebInfo defaults to /Ob1, which leaves the
    #               kernels un-inlined and makes TRAINING ~10x SLOWER (measured:
    #               85 ms/step vs 7.9 ms/step). Force /Ob2 in optimized configs.
    if (MSVC)
        target_compile_options("${tgt}" PRIVATE /bigobj /fp:fast)
        target_compile_options("${tgt}" PRIVATE "$<$<NOT:$<CONFIG:Debug>>:/Ob2>")
    endif ()

    if (THREEPP_RLTOOLS_FAST)
        if (_clang_cl)
            target_compile_options("${tgt}" PRIVATE /clang:-O3 /clang:-ffast-math /clang:-march=native)
            message(STATUS "${tgt}: FAST — clang-cl -O3 -ffast-math -march=native")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options("${tgt}" PRIVATE -O3 -ffast-math -march=native)
            message(STATUS "${tgt}: FAST — -O3 -ffast-math -march=native")
        elseif (MSVC)
            message(STATUS "${tgt}: FAST has little effect under cl.exe — configure with "
                           "-DCMAKE_CXX_COMPILER=clang-cl for real gains")
        endif ()
    endif ()
endfunction()

add_example(NAME "rltools_pendulum" SOURCES main.cpp RLPendulumTrainer.cpp LINK_IMGUI)
if (TARGET "rltools_pendulum")
    _rltools_configure(rltools_pendulum)
endif ()

add_example(NAME "rltools_swarm" SOURCES swarm.cpp RLSwarmTrainer.cpp LINK_IMGUI)
if (TARGET "rltools_swarm")
    _rltools_configure(rltools_swarm)
endif ()

# Debug disables optimization entirely (/Od), so RLtools trains painfully slowly there
# no matter what. Warn on single-config generators set to Debug / no type.
if (NOT CMAKE_CONFIGURATION_TYPES AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(WARNING
        "rltools examples: CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}' — RLtools' training kernels "
        "are UNOPTIMIZED in Debug and train ~10x slower. Configure with "
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo (or Release) for the intended speed.")
endif ()
