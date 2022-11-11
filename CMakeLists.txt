cmake_minimum_required(VERSION 3.15)
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" projectVersion)
project(threepp VERSION ${projectVersion})
message("Current ${PROJECT_NAME} version: ${CMAKE_PROJECT_VERSION}\n")

option(THREEPP_BUILD_EXAMPLES "Build examples" ON)

# ==============================================================================
# Global internal configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Prepare for export and installation.
set(THREEPP_HEADER_INSTALL_DIR "include")
if (WIN32)
    set(THREEPP_CMAKE_INSTALL_DIR "cmake")
else ()
    set(THREEPP_CMAKE_INSTALL_DIR "share/${PROJECT_NAME}/cmake")
endif ()
set(THREEPP_INSTALL_DESTINATIONS
        ARCHIVE DESTINATION "lib"
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        INCLUDES DESTINATION "${THREEPP_HEADER_INSTALL_DIR}")
set(THREEPP_EXPORT_TARGET "${PROJECT_NAME}-targets")


# ==============================================================================
# Conan setup
# ==============================================================================

if(CONAN_EXPORTED) # in conan local cache
    # standard conan installation, deps will be defined in conanfile.py
    # and not necessary to call conan again, conan is already running
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else()

    include(cmake/conan.cmake)

    conan_cmake_run(
            CONANFILE conanfile.py
            BUILD missing
            SETTINGS build_type=${CMAKE_BUILD_TYPE}
            BASIC_SETUP CMAKE_TARGETS)

endif()


# ==============================================================================
# Source code
# ==============================================================================

add_subdirectory(src)

if (THREEPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


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


# ==============================================================================
# Exports and remaining installation
# ==============================================================================

install(
        EXPORT "${THREEPP_EXPORT_TARGET}"
        DESTINATION "${THREEPP_CMAKE_INSTALL_DIR}"
        NAMESPACE "${PROJECT_NAME}::"
)

include(CMakePackageConfigHelpers)

# Generate and install package-config file.
set(configFile "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake")
set(targetsFile "${THREEPP_CMAKE_INSTALL_DIR}/${THREEPP_EXPORT_TARGET}.cmake")
configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/project-config.cmake.in"
        "${configFile}"
        INSTALL_DESTINATION "${THREEPP_CMAKE_INSTALL_DIR}"
        PATH_VARS targetsFile
)
install(FILES "${configFile}" DESTINATION "${THREEPP_CMAKE_INSTALL_DIR}")

# Generate and install package-version file
set(versionFile "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake")
write_basic_package_version_file(
        "${versionFile}"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY "SameMajorVersion")
install(FILES "${versionFile}" DESTINATION "${THREEPP_CMAKE_INSTALL_DIR}")
