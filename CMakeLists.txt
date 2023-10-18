cmake_minimum_required(VERSION 3.19)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/version.txt" projectVersion)
project(threepp VERSION ${projectVersion})
message("Current ${PROJECT_NAME} version: ${CMAKE_PROJECT_VERSION}\n")

option(THREEPP_BUILD_EXAMPLES "Build examples" ON)
option(THREEPP_BUILD_EXAMPLE_PROJECTS "Build example projects" OFF)
option(THREEPP_BUILD_TESTS "Build test suite" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)


# ==============================================================================
# Global internal configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


# ==============================================================================
# Dependencies
# ==============================================================================

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(stb CONFIG QUIET)

if (NOT TARGET "glfw::glfw" AND TARGET "glfw")
    add_library(glfw::glfw ALIAS glfw)
endif()

if (NOT stb_INCLUDE_DIR)
    find_path(stb_INCLUDE_DIR "stb_c_lexer.h" REQUIRED)
endif()

include(FetchContent)
FetchContent_Declare(
        glText
        GIT_REPOSITORY https://github.com/vallentin/glText.git
        GIT_TAG 8200fa70e32acec0a3cd777d404f41ee0c203ca4
)
FetchContent_Populate(glText)
set(gltext_INCLUDE_DIR "${gltext_SOURCE_DIR}")


# ==============================================================================
# Targets
# ==============================================================================

add_subdirectory(src)

if (THREEPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (THREEPP_BUILD_TESTS)

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
# Application resources
# ==============================================================================

# Data files
file(COPY "${CMAKE_CURRENT_LIST_DIR}/data/models" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/data)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/data/textures" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/data)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/data/fonts" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/data)
file(COPY "${CMAKE_CURRENT_LIST_DIR}/data/sounds" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/data)


# ==============================================================================
# Exports and remaining installation
# ==============================================================================

configure_package_config_file(cmake/config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/threepp-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/threepp
        NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/threepp-config-version.cmake
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/threepp-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/threepp-config-version.cmake
        DESTINATION
        ${CMAKE_INSTALL_DATADIR}/threepp)

install(TARGETS threepp EXPORT threepp-targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT threepp-targets
        NAMESPACE threepp::
        DESTINATION ${CMAKE_INSTALL_DATADIR}/threepp)
