cmake_minimum_required(VERSION 3.19)
project(threepp)

include(CMakeDependentOption)

option(THREEPP_BUILD_EXAMPLES "Build examples" ON)
option(THREEPP_BUILD_TESTS "Build test suite" ON)
option(THREEPP_WITH_SVG "Build with SVGLoader" ON)
option(THREEPP_WITH_AUDIO "Build with Audio" ON)

option(THREEPP_WITH_FBX "Build with FBXLoader (OpenFBX)" OFF)
option(THREEPP_WITH_USD "Build with USDLoader (tinyusdz)" OFF)

option(THREEPP_TREAT_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)


# Force THREEPP_WITH_GLFW ON when targeting Emscripten
cmake_dependent_option(THREEPP_WITH_GLFW "Build with GLFW frontend" ON "NOT DEFINED EMSCRIPTEN" ON)
# Force THREEPP_USE_EXTERNAL_GLFW OFF if NOT THREEPP_WITH_GLFW
cmake_dependent_option(THREEPP_USE_EXTERNAL_GLFW "Use externally supplied GLFW" OFF "THREEPP_WITH_GLFW" OFF)

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

if (THREEPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
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
