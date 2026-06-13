# RLtools (https://rl.tools) reinforcement-learning demo.
# Enabled by configuring with -DTHREEPP_WITH_RLTOOLS=ON, which fetches the
# header-only library and defines the `rltools` interface target.

add_example(NAME "rltools_pendulum" SOURCES main.cpp RLPendulumTrainer.cpp LINK_IMGUI)

if (TARGET "rltools_pendulum")
    target_link_libraries("rltools_pendulum" PRIVATE rltools)

    find_package(Threads REQUIRED)
    target_link_libraries("rltools_pendulum" PRIVATE Threads::Threads)

    # RLtools' template instantiation is heavy; MSVC needs /bigobj. /fp:fast
    # matches the math the policy was validated with.
    if (MSVC)
        target_compile_options("rltools_pendulum" PRIVATE /bigobj /fp:fast)
    endif ()
endif ()
