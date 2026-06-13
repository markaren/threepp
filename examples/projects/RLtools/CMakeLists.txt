# RLtools (https://rl.tools) reinforcement-learning demo.
# Enabled by configuring with -DTHREEPP_WITH_RLTOOLS=ON, which fetches the
# header-only library and defines the `rltools` interface target.

# Optional aggressive optimization of the (template-heavy, compute-bound) RLtools
# trainer. RLtools' own CMake only adds -O3/-ffast-math/-march=native for GNU/Clang
# and NOTHING for MSVC, so a plain cl.exe build leaves a lot of training throughput
# on the table. Turning this ON applies the best flags for the detected compiler.
#
# NOTE: this bakes the BUILD MACHINE's instruction set (-march=native) into the
# binary, so it is OFF by default. The big win is with the LLVM frontend
# (clang-cl): configure with -DCMAKE_CXX_COMPILER=clang-cl from a vcvars prompt.
option(THREEPP_RLTOOLS_FAST "Aggressively optimize the RLtools trainer TU (march=native; build-machine specific)" OFF)

add_example(NAME "rltools_pendulum" SOURCES main.cpp RLPendulumTrainer.cpp LINK_IMGUI)

if (TARGET "rltools_pendulum")
    target_link_libraries("rltools_pendulum" PRIVATE rltools)

    find_package(Threads REQUIRED)
    target_link_libraries("rltools_pendulum" PRIVATE Threads::Threads)

    # clang-cl reports MSVC=TRUE (MSVC-compatible driver); distinguish it from
    # cl.exe via the compiler id + frontend variant.
    set(_rltools_clang_cl FALSE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(_rltools_clang_cl TRUE)
    endif ()

    # Base flags: RLtools' template instantiation is heavy, so MSVC / clang-cl
    # need /bigobj; /fp:fast matches the math the policy was validated with.
    if (MSVC)
        target_compile_options("rltools_pendulum" PRIVATE /bigobj /fp:fast)
    endif ()

    if (THREEPP_RLTOOLS_FAST)
        if (_rltools_clang_cl)
            # LLVM frontend, MSVC-style driver: pass clang-native flags via /clang:
            target_compile_options("rltools_pendulum" PRIVATE /clang:-O3 /clang:-ffast-math /clang:-march=native)
            message(STATUS "rltools_pendulum: FAST — clang-cl -O3 -ffast-math -march=native")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            # GNU-style clang/gcc (e.g. MinGW, Linux, macOS)
            target_compile_options("rltools_pendulum" PRIVATE -O3 -ffast-math -march=native)
            message(STATUS "rltools_pendulum: FAST — -O3 -ffast-math -march=native")
        elseif (MSVC)
            # cl.exe doesn't vectorize RLtools' kernels; /arch is no help (measured).
            message(STATUS "rltools_pendulum: FAST has little effect under cl.exe — "
                           "configure with -DCMAKE_CXX_COMPILER=clang-cl for real gains")
        endif ()
    endif ()
endif ()
