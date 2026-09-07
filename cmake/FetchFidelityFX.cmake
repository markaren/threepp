# cmake/FetchFidelityFX.cmake
# Fetches the AMD FidelityFX SDK and wires up the prebuilt, signed Vulkan
# ffx-api runtime (amd_fidelityfx_vk.dll + import lib). No SDK-from-source build
# is needed: the DLL already contains every backend and precompiled shader
# permutation, so we only link the import library, add the ffx-api headers, and
# copy the DLL next to executables.
#
# Windows-only (the prebuilt VK runtime is a Windows binary); THREEPP_WITH_FSR
# is already gated on WIN32 in the top-level CMakeLists.
#
# NOTE: the release archive is large (~470 MB — it bundles all samples + media).
# It is downloaded once per build tree. To reuse a local extraction, configure
# with -DFETCHCONTENT_SOURCE_DIR_FIDELITYFX_SDK=<path-to-extracted-sdk>.

include(FetchContent)

set(FIDELITYFX_SDK_VERSION "v1.1.4" CACHE STRING "FidelityFX SDK release tag")

set(_ffx_url
    "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download/${FIDELITYFX_SDK_VERSION}/FidelityFX-SDK-${FIDELITYFX_SDK_VERSION}.zip")

message(STATUS "FidelityFX SDK: fetching ${FIDELITYFX_SDK_VERSION} (large ~470 MB archive; "
               "override with -DFETCHCONTENT_SOURCE_DIR_FIDELITYFX_SDK=<dir>)")

FetchContent_Declare(
    fidelityfx_sdk
    URL "${_ffx_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Populate the source only — do NOT add_subdirectory(). The SDK ships a root
# CMakeLists that builds the whole Cauldron sample framework; we only want the
# ffx-api headers + the prebuilt signed Vulkan runtime out of the archive.
# CMP0169 OLD lets us call FetchContent_Populate directly on CMake 4.x (same
# pattern as the RLtools / OpenFBX fetches in FetchOptionalDeps.cmake).
cmake_policy(SET CMP0169 OLD)
FetchContent_GetProperties(fidelityfx_sdk)
if (NOT fidelityfx_sdk_POPULATED)
    FetchContent_Populate(fidelityfx_sdk)
endif ()

# ffx-api C headers: <ffx_api/ffx_api.h>, <ffx_api/ffx_upscale.h>,
# <ffx_api/vk/ffx_api_vk.h> all resolve from this include root.
set(_ffx_inc "${fidelityfx_sdk_SOURCE_DIR}/ffx-api/include")
if (NOT EXISTS "${_ffx_inc}/ffx_api/ffx_api.h")
    message(FATAL_ERROR
        "FidelityFX SDK: ffx-api headers not found at ${_ffx_inc}. "
        "Archive layout may have changed. URL was: ${_ffx_url}")
endif ()

# Prebuilt signed Vulkan runtime: import lib + DLL.
find_file(_ffx_vk_lib
    NAMES "amd_fidelityfx_vk.lib"
    PATHS "${fidelityfx_sdk_SOURCE_DIR}/PrebuiltSignedDLL"
          "${fidelityfx_sdk_SOURCE_DIR}/ffx-api/bin"
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_file(_ffx_vk_dll
    NAMES "amd_fidelityfx_vk.dll"
    PATHS "${fidelityfx_sdk_SOURCE_DIR}/PrebuiltSignedDLL"
          "${fidelityfx_sdk_SOURCE_DIR}/ffx-api/bin"
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

if (NOT _ffx_vk_lib OR NOT _ffx_vk_dll)
    message(FATAL_ERROR
        "FidelityFX SDK: prebuilt Vulkan runtime (amd_fidelityfx_vk.lib/.dll) "
        "not found under ${fidelityfx_sdk_SOURCE_DIR}. URL was: ${_ffx_url}")
endif ()

# Export for src/CMakeLists.txt.
set(FFX_INCLUDE_DIR "${_ffx_inc}"     CACHE INTERNAL "FidelityFX ffx-api include dir")
set(FFX_LIBRARY     "${_ffx_vk_lib}"  CACHE INTERNAL "FidelityFX Vulkan import library")
set(FFX_DLL         "${_ffx_vk_dll}"  CACHE INTERNAL "FidelityFX Vulkan runtime DLL")
set(FFX_LICENSE     "${fidelityfx_sdk_SOURCE_DIR}/sdk/LICENSE.txt" CACHE INTERNAL "FidelityFX license")

message(STATUS "FidelityFX SDK: include = ${FFX_INCLUDE_DIR}")
message(STATUS "FidelityFX SDK: library = ${FFX_LIBRARY}")
message(STATUS "FidelityFX SDK: dll     = ${FFX_DLL}")
