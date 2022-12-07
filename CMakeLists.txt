cmake_minimum_required(VERSION 3.15)
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" projectVersion)
project(threepp VERSION ${projectVersion})
message("Current ${PROJECT_NAME} version: ${CMAKE_PROJECT_VERSION}\n")

option(THREEPP_BUILD_EXAMPLES "Build examples" ON)
option(THREEPP_BUILD_TESTS "Build test suite" ON)

# ==============================================================================
# Global internal configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# ==============================================================================
# Dependencies
# ==============================================================================

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(stb CONFIG QUIET)
find_package(assimp CONFIG)

if (NOT TARGET "glfw::glfw" AND TARGET "glfw")
    add_library(glfw::glfw ALIAS glfw)
endif()

if (NOT TARGET "stb::stb")
    find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")
    add_library(stb::stb INTERFACE IMPORTED)
endif ()

target_include_directories(stb::stb INTERFACE ${STB_INCLUDE_DIRS})

if (THREEPP_BUILD_EXAMPLES)

    find_package(Bullet CONFIG)

    if (BULLET_FOUND AND NOT TARGET Bullet::Bullet)
        add_library(Bullet::Bullet INTERFACE IMPORTED)
        if("${BULLET_INCLUDE_DIRS}" MATCHES ".*/bullet$")
            target_include_directories(Bullet::Bullet INTERFACE "${BULLET_INCLUDE_DIRS}/..")
        endif()
        target_link_libraries(Bullet::Bullet INTERFACE LinearMath Bullet3Common BulletCollision BulletDynamics)
    endif ()

    add_subdirectory(examples)

endif ()

add_subdirectory(src)

if (THREEPP_BUILD_TESTS)

    include(FetchContent)
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v2.13.8
    )
    FetchContent_MakeAvailable(Catch2)

    enable_testing()
    add_subdirectory(tests)
endif ()


# ==============================================================================
# Application resources
# ==============================================================================

# Shaders
file(COPY "${CMAKE_SOURCE_DIR}/src/threepp/renderers/shaders/ShaderChunk" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY "${CMAKE_SOURCE_DIR}/src/threepp/renderers/shaders/ShaderLib" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Data files
file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

# ==============================================================================
# Exports and remaining installation
# ==============================================================================


include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


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
