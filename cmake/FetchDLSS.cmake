# cmake/FetchDLSS.cmake
# Fetches the NVIDIA DLSS Super Resolution SDK (github.com/NVIDIA/DLSS) and
# wires up the NGX static import library (nvsdk_ngx_*.lib) + the signed
# production DLSS runtime (nvngx_dlss.dll). Unlike FidelityFX there is no
# dllexport trap: the NGX entry points live in the static lib and the lib
# LoadLibrary's nvngx_dlss.dll itself at NVSDK_NGX_VULKAN_Init time (we pass the
# module directory as a search path — see DlssUpscaler.cpp).
#
# Windows-only (NGX Vulkan needs the NVIDIA driver's NGX loader);
# THREEPP_WITH_DLSS is already gated on WIN32 in the top-level CMakeLists.
#
# NOTE: the archive is a full repo snapshot (~hundreds of MB — it bundles the
# dev/rel DLLs for every platform + docs). Downloaded once per build tree; to
# reuse a local extraction, configure with
# -DFETCHCONTENT_SOURCE_DIR_DLSS_SDK=<path-to-extracted-repo>.

include(FetchContent)

set(DLSS_SDK_VERSION "v310.7.0" CACHE STRING "NVIDIA DLSS SDK release tag")

set(_dlss_url "https://github.com/NVIDIA/DLSS/archive/refs/tags/${DLSS_SDK_VERSION}.zip")

message(STATUS "DLSS SDK: fetching ${DLSS_SDK_VERSION} (large archive; "
               "override with -DFETCHCONTENT_SOURCE_DIR_DLSS_SDK=<dir>)")

FetchContent_Declare(
    dlss_sdk
    URL "${_dlss_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Populate only — the repo has no consumable CMake build; we want the headers,
# one import lib and one DLL out of the snapshot. Same CMP0169 pattern as
# FetchFidelityFX.cmake.
cmake_policy(SET CMP0169 OLD)
FetchContent_GetProperties(dlss_sdk)
if (NOT dlss_sdk_POPULATED)
    FetchContent_Populate(dlss_sdk)
endif ()

# NGX C headers: <nvsdk_ngx.h>, <nvsdk_ngx_vk.h>, <nvsdk_ngx_helpers_vk.h>
# (+ the dlssd/dlssg variants for Ray Reconstruction / Frame Generation).
set(_ngx_inc "${dlss_sdk_SOURCE_DIR}/include")
if (NOT EXISTS "${_ngx_inc}/nvsdk_ngx_vk.h")
    message(FATAL_ERROR
        "DLSS SDK: NGX headers not found at ${_ngx_inc}. "
        "Archive layout may have changed. URL was: ${_dlss_url}")
endif ()

# Static import libs. _d = dynamic CRT (/MD, threepp's default), _d_dbg = /MDd.
set(_ngx_lib_dir "${dlss_sdk_SOURCE_DIR}/lib/Windows_x86_64/x64")
set(_ngx_lib_rel "${_ngx_lib_dir}/nvsdk_ngx_d.lib")
set(_ngx_lib_dbg "${_ngx_lib_dir}/nvsdk_ngx_d_dbg.lib")
# Signed production DLSS runtime (rel/). dev/ carries the watermarked debug
# runtime — deliberately not used.
set(_ngx_dll "${dlss_sdk_SOURCE_DIR}/lib/Windows_x86_64/rel/nvngx_dlss.dll")

foreach (_f IN ITEMS "${_ngx_lib_rel}" "${_ngx_lib_dbg}" "${_ngx_dll}")
    if (NOT EXISTS "${_f}")
        message(FATAL_ERROR
            "DLSS SDK: expected file missing: ${_f}. URL was: ${_dlss_url}")
    endif ()
endforeach ()

# Export for src/CMakeLists.txt.
set(NGX_INCLUDE_DIR "${_ngx_inc}"     CACHE INTERNAL "NVIDIA NGX include dir")
set(NGX_LIBRARY_REL "${_ngx_lib_rel}" CACHE INTERNAL "NGX static import lib (/MD)")
set(NGX_LIBRARY_DBG "${_ngx_lib_dbg}" CACHE INTERNAL "NGX static import lib (/MDd)")
set(NGX_DLSS_DLL    "${_ngx_dll}"     CACHE INTERNAL "Signed DLSS runtime DLL")
set(NGX_LICENSE     "${dlss_sdk_SOURCE_DIR}/LICENSE.txt" CACHE INTERNAL "DLSS SDK license")

message(STATUS "DLSS SDK: include = ${NGX_INCLUDE_DIR}")
message(STATUS "DLSS SDK: lib     = ${NGX_LIBRARY_REL}")
message(STATUS "DLSS SDK: dll     = ${NGX_DLSS_DLL}")
