
# threepp_player — the second front end over the editor's play runtime.
#
# Pulled in AFTER apps/editor (see the top-level CMakeLists), because the Python
# scripting library is declared there: threepp_editor_scripting is what makes a
# document's scripts actually run, and hover_arena_author is the precedent for a
# sibling executable that links it. The player links no part of the editor APP —
# no ImGui, no panels — only the play core in libthreepp and that scripting
# library.
#
# Every optional half is gated the way the editor gates it, and the player builds
# and runs without any of them: no PhysX means a document plays without physics,
# no Python means its scripts are loaded and not run.

# The player minus the window. Its own library for the same reason
# threepp_editor_scripting and threepp_editor_examples are: the headless test
# links the code the binary runs, rather than a second copy of it that can drift.
add_library(threepp_player_core STATIC
        PlayerCore.cpp)

target_link_libraries(threepp_player_core PUBLIC threepp)
target_include_directories(threepp_player_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(threepp_player_core PUBLIC cxx_std_20)

# PhysX is an optional vcpkg dependency (feature "physx"). Found here rather than
# relying on apps/editor having looked for it, because find_package's imported
# targets are directory-scoped — the same dance tests/extras/CMakeLists.txt does.
if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
endif ()

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    # PUBLIC throughout: PlayerCore.hpp does not include the PhysX headers, but
    # PlayerCore.cpp includes PhysicsPlaySession.hpp, whose inline methods branch
    # on THREEPP_EDITOR_WITH_VHACD. Every target that shares a binary with this
    # one must compile that header identically or they disagree within one
    # program (ODR). Linking V-HACD carries its INTERFACE macro out to all of
    # them, which is what keeps the definitions in step without a per-target
    # macro to get wrong.
    target_link_libraries(threepp_player_core PUBLIC unofficial::omniverse-physx-sdk::sdk)
    target_compile_definitions(threepp_player_core PUBLIC THREEPP_EDITOR_WITH_PHYSX=1)
    if (TARGET threepp_convex_decomp)
        target_link_libraries(threepp_player_core PUBLIC threepp_convex_decomp)
    endif ()
    message(STATUS "threepp_player: PhysX play session enabled")
else ()
    message(STATUS "threepp_player: physx not found, documents play without physics")
endif ()

# Python scripting. Declared by apps/editor, which is where the embedded
# interpreter and the bind_*.cpp translation units live; PUBLIC on that target
# carries THREEPP_EDITOR_WITH_PYTHON and the scripting include directory here.
if (TARGET threepp_editor_scripting)
    target_link_libraries(threepp_player_core PUBLIC threepp_editor_scripting)
    message(STATUS "threepp_player: Python scripting enabled")
else ()
    message(STATUS "threepp_player: scripting not built, a document's scripts will not run")
endif ()

add_executable(threepp_player
        main.cpp
        PlayerApp.cpp
        DebugDrawOverlay.cpp)

target_link_libraries(threepp_player PRIVATE threepp_player_core)

# The GPU libraries PhysX loads at runtime, beside the binary — the same
# POST_BUILD copies threepp_editor does, so the player can be run from its own
# output directory without one.
if (TARGET unofficial::omniverse-physx-sdk::sdk)
    if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-library)
        add_custom_command(TARGET threepp_player POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
                $<TARGET_FILE_DIR:threepp_player>)
    endif ()
    if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-device-library)
        add_custom_command(TARGET threepp_player POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
                $<TARGET_FILE_DIR:threepp_player>)
    endif ()
endif ()
