cmake_minimum_required(VERSION 3.21)
project(threepp)

include(CMakeDependentOption)

# Examples and tests are for people working *on* threepp: they add 100+ executables,
# Catch2 and the (large) asset repo to the build. Consumers who add_subdirectory or
# FetchContent threepp get just the library unless they ask for more.
option(THREEPP_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(THREEPP_BUILD_TESTS "Build test suite" ${PROJECT_IS_TOP_LEVEL})
option(THREEPP_WITH_AUDIO "Build with Audio" ON)

option(THREEPP_WITH_FBX "Build with FBXLoader (OpenFBX)" OFF)
option(THREEPP_WITH_USD "Build with USDLoader (tinyusdz)" OFF)
option(THREEPP_WITH_RLTOOLS "Fetch RLtools (header-only deep RL) for the reinforcement-learning example" OFF)

option(THREEPP_WITH_PYTHON "Build the pybind11 Python bindings (threepp module)" OFF)

option(THREEPP_TREAT_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(THREEPP_WITH_VULKAN "Build with Vulkan backend (requires Vulkan SDK)" OFF)
# The backend builds on macOS but cannot run there: its shading path is ray-query
# based (shadows, reflections and GI all trace) and MoltenVK implements none of
# the KHR ray-tracing extensions. A warning rather than a hard error — building
# it is still useful as a compile check, and the runtime failure now says so
# plainly (VulkanContext::pickPhysicalDevice).
if (THREEPP_WITH_VULKAN AND APPLE)
    message(WARNING
        "THREEPP_WITH_VULKAN builds on macOS but will not run: the renderer requires "
        "KHR ray tracing, which MoltenVK does not implement. Use GLRenderer on macOS.")
endif ()
# AMD FidelityFX FSR 3.1 temporal upscaler for the Vulkan backend. Windows-only:
# the FidelityFX SDK bundles a Windows shader compiler + builds a Vulkan backend
# with MSVC. When OFF (or off-Windows) the renderer uses its built-in TAA
# temporal upsampler (taa_resolve.comp) instead — FSR is a compile-time swap of
# that one pass, with a runtime fallback to TAA if FSR context creation fails.
cmake_dependent_option(THREEPP_WITH_FSR
    "Build with AMD FidelityFX FSR 3.1 upscaler (Windows, Vulkan)"
    OFF "THREEPP_WITH_VULKAN AND WIN32" OFF)
# NVIDIA DLSS Super Resolution for the Vulkan backend (RTX GPUs; runtime no-op
# elsewhere). Same compile-time-swap shape as FSR: replaces the TAA temporal
# upsampler when the NGX feature creates successfully, with a runtime fallback
# to FSR (if built + available) and then the built-in TAA. When both DLSS and
# FSR are active, DLSS wins; setDlss(false) hands back to FSR.
cmake_dependent_option(THREEPP_WITH_DLSS
    "Build with NVIDIA DLSS Super Resolution upscaler (Windows, Vulkan, RTX)"
    OFF "THREEPP_WITH_VULKAN AND WIN32" OFF)

cmake_dependent_option(THREEPP_WITH_INFERENCE "Build with Inference support" OFF "THREEPP_WITH_VULKAN" OFF)

# Force THREEPP_WITH_GLFW ON when targeting Emscripten
cmake_dependent_option(THREEPP_WITH_GLFW "Build with GLFW frontend" ON "NOT DEFINED EMSCRIPTEN" ON)
# Force THREEPP_USE_EXTERNAL_GLFW OFF if NOT THREEPP_WITH_GLFW
cmake_dependent_option(THREEPP_USE_EXTERNAL_GLFW "Use externally supplied GLFW" OFF "THREEPP_WITH_GLFW" OFF)

# The scene editor application (apps/editor). Needs a window, so it follows
# THREEPP_WITH_GLFW; otherwise it defaults on for anyone building the project
# itself, exactly like the examples.
cmake_dependent_option(THREEPP_BUILD_EDITOR "Build the threepp scene editor"
        ${PROJECT_IS_TOP_LEVEL} "THREEPP_WITH_GLFW AND NOT DEFINED EMSCRIPTEN" OFF)

cmake_dependent_option(THREEPP_BUILD_EXAMPLE_PROJECTS "Build example projects" ON "THREEPP_BUILD_EXAMPLES" OFF)


# ==============================================================================
# Global internal configuration
# ==============================================================================

# Give a single-config generator (Ninja, Makefiles) a build type when nothing
# else has. An empty CMAKE_BUILD_TYPE means no optimisation flags at all, which
# is what a bare `cmake . -B build` produces on Linux and macOS: a silently
# unoptimised build. MSVC is unaffected either way — its platform module already
# defaults to Debug during project(), so this is a no-op on Windows.
#
# Only when we are the top-level project: as a subproject (FetchContent /
# add_subdirectory) the build type belongs to whoever consumed us.
#
# Evaluating CMAKE_BUILD_TYPE here has a second effect, which is worth keeping
# even where the assignment never fires. Multi-config generators (Visual Studio,
# Xcode) choose the configuration at build time and ignore the variable, so a
# preset that sets it — as ours do, for the single-config case — would otherwise
# draw a "Manually-specified variables were not used by the project" warning on
# every configure. Reading it in the test above is what marks it as used.
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (single-config generators)" FORCE)
endif ()
# Offers the usual four as a dropdown in cmake-gui / ccmake. Guarded because a
# multi-config generator leaves no CMAKE_BUILD_TYPE cache entry to annotate, and
# set_property(CACHE) on a missing entry is an error.
if (DEFINED CACHE{CMAKE_BUILD_TYPE})
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


# ==============================================================================
# Dependencies
# ==============================================================================


if (DEFINED EMSCRIPTEN)
    if ("${CMAKE_SYSTEM}" MATCHES "Linux" OR "${CMAKE_SYSTEM}" MATCHES "Darwin-*")
        find_package(OpenGL REQUIRED)
    endif ()
else ()

    if (THREEPP_USE_EXTERNAL_GLFW)
        find_package(glfw3 CONFIG REQUIRED)
    endif ()

endif ()


# ==============================================================================
# Targets
# ==============================================================================

include(cmake/FetchOptionalDeps.cmake)

# Static PhysX + GNU ld needs the archives wrapped in a rescan group (see the
# module for the full story). Included here so the function is defined for
# every subdirectory; each find_package site calls it — the port's imported
# targets are directory-scoped, so there is no central place to patch them.
include(cmake/PhysxStaticLinkFix.cmake)

add_subdirectory(src)

# The editor/player document model and play runtime — a static library outside
# libthreepp and outside the install, because its headers change meaning under
# per-consumer macros the installed library cannot own (see the file's header
# comment). Before the editor gate on purpose: the Python bindings and the
# tests consume it whether or not the apps are built.
add_subdirectory(editor-core)

# V-HACD convex decomposition, a small static library the editor and the PhysX
# tests share. Self-gates on the PhysX SDK being present, so a build without
# PhysX skips it entirely.
include(cmake/ConvexDecomposition.cmake)

# Python bindings (pybind11). Built only when explicitly enabled; pulls in
# pybind11 via FetchContent and produces the `threepp` Python extension module.
if (THREEPP_WITH_PYTHON)
    add_subdirectory(python)
endif ()

# Example/test assets live in a separate
# repo, fetched only when building examples or tests so library consumers stay
# lean. Point at a local checkout with -DFETCHCONTENT_SOURCE_DIR_THREEPP_DATA=<dir>.
if (THREEPP_BUILD_EXAMPLES OR THREEPP_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
            threepp_data
            GIT_REPOSITORY https://github.com/markaren/threepp_data.git
            GIT_TAG 93069e76bfd6cd75e785d0334738cb303d4f555e
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(threepp_data)
    set(THREEPP_DATA_DIR "${threepp_data_SOURCE_DIR}" CACHE INTERNAL "threepp-data asset path")
endif ()

if (THREEPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (THREEPP_BUILD_EDITOR)
    add_subdirectory(apps/editor)
    # threepp_player: the same play runtime with no editing machinery, for
    # policy validation and data generation. After apps/editor, not beside it —
    # the Python scripting library it links is declared there.
    add_subdirectory(apps/player)
endif ()

if (NOT DEFINED EMSCRIPTEN AND THREEPP_BUILD_TESTS)

    include(FetchContent)
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    enable_testing()
    add_subdirectory(tests)
endif ()


# ==============================================================================
# Exports and remaining installation
# ==============================================================================

configure_package_config_file(cmake/config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/threepp-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/threepp"
        PATH_VARS THREEPP_USE_EXTERNAL_GLFW
        NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/threepp-config-version.cmake"
        VERSION 0.0.0
        COMPATIBILITY AnyNewerVersion)
install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/threepp-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/threepp-config-version.cmake"
        DESTINATION
        "${CMAKE_INSTALL_DATADIR}/threepp")

if (BUILD_SHARED_LIBS AND NOT THREEPP_USE_EXTERNAL_GLFW)
    install(TARGETS glfw EXPORT threepp-targets)
endif ()
if (THREEPP_WITH_USD)
    install(TARGETS tinyusdz_static EXPORT threepp-targets)
    install(DIRECTORY "${tinyusdz_SOURCE_DIR}/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tinyusdz"
            FILES_MATCHING PATTERN "*.hh" PATTERN "*.h" PATTERN "*.inc")
endif ()
install(TARGETS threepp EXPORT threepp-targets)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT threepp-targets
        NAMESPACE threepp::
        DESTINATION "${CMAKE_INSTALL_DATADIR}/threepp")
