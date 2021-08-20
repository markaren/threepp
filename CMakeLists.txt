cmake_minimum_required(VERSION 3.10)
project(threepp)

# ==============================================================================
# Global internal configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# ==============================================================================
# Conan setup
# ==============================================================================

# Download automatically, you can also just copy the conan.cmake file
if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake")
endif ()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(
        REQUIRES
        glfw/3.3.4
        glad/0.1.34
        stb/20200203
        OPTIONS
        glad:gl_version=4.1
        GENERATORS
        cmake
        BUILD missing
        SETTINGS build_type=${CMAKE_BUILD_TYPE}
        BASIC_SETUP CMAKE_TARGETS)


# ==============================================================================
# Source code
# ==============================================================================

add_subdirectory(src)
add_subdirectory(examples)


# ==============================================================================
# Application resources
# ==============================================================================

# Shaders
file(COPY "${CMAKE_SOURCE_DIR}/src/threepp/renderers/shaders/ShaderChunk" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY "${CMAKE_SOURCE_DIR}/src/threepp/renderers/shaders/ShaderLib" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Textures
file(COPY "${CMAKE_SOURCE_DIR}/data/textures" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

# favicon
file(COPY "${CMAKE_SOURCE_DIR}/data/favicon.bmp" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
