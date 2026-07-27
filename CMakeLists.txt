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
cmake_dependent_option(THREEPP_FETCH_ASSIMP "Make assimp available for examples" OFF "THREEPP_BUILD_EXAMPLES" OFF)


# ==============================================================================
# Global internal configuration
# ==============================================================================

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

add_subdirectory(src)

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
            GIT_TAG 6a99ee82f9ce68eddcad6e3d0b5cbb5ea889e5c5
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
