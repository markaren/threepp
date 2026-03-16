# cmake/FetchWgpuNative.cmake
# Downloads pre-built wgpu-native binaries for the current platform.

include(FetchContent)

set(WGPU_NATIVE_VERSION "v27.0.4.0" CACHE STRING "wgpu-native release version")

# Determine platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_wgpu_os "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_wgpu_os "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_wgpu_os "windows")
else()
    message(FATAL_ERROR
        "wgpu-native: unsupported platform '${CMAKE_SYSTEM_NAME}'. "
        "Supported: Linux, Darwin (macOS), Windows.")
endif()

# Determine architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_wgpu_arch "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_wgpu_arch "aarch64")
else()
    message(FATAL_ERROR
        "wgpu-native: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
        "Supported: x86_64/AMD64, aarch64/arm64.")
endif()

# Construct archive filename
if(_wgpu_os STREQUAL "windows")
    set(_wgpu_archive "wgpu-${_wgpu_os}-${_wgpu_arch}-msvc-release.zip")
else()
    set(_wgpu_archive "wgpu-${_wgpu_os}-${_wgpu_arch}-release.zip")
endif()

set(_wgpu_url
    "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}/${_wgpu_archive}")

message(STATUS "wgpu-native: fetching ${_wgpu_archive} (${WGPU_NATIVE_VERSION})")

FetchContent_Declare(
    wgpu_native
    URL "${_wgpu_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(wgpu_native)

# Locate the library inside the fetched content
# On Windows, use the DLL import library to avoid LNK1190 errors from
# Rust-compiled objects in the static lib. On other platforms, use the static lib.
if(WIN32)
    set(_wgpu_lib_name "wgpu_native.dll.lib")
else()
    set(_wgpu_lib_name "libwgpu_native.a")
endif()

find_file(_wgpu_native_lib
    NAMES "${_wgpu_lib_name}"
    PATHS "${wgpu_native_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(NOT _wgpu_native_lib)
    message(FATAL_ERROR
        "wgpu-native: library not found in fetched content at "
        "${wgpu_native_SOURCE_DIR}/lib. "
        "Archive may have an unexpected layout. URL was: ${_wgpu_url}")
endif()

# Export variables for the rest of the build
set(WGPU_INCLUDE_DIR "${wgpu_native_SOURCE_DIR}/include" CACHE INTERNAL
    "wgpu-native include directory")

# On Windows, link against the DLL import lib; on other platforms the static lib
# needs platform system libraries.
set(_wgpu_libs "${_wgpu_native_lib}")
if(WIN32)
    list(APPEND _wgpu_libs
        ws2_32 ntdll d3dcompiler opengl32 userenv bcrypt
        ole32 propsys dxgi runtimeobject)
endif()
set(WGPU_LIBRARY "${_wgpu_libs}" CACHE INTERNAL
    "wgpu-native library path plus required system libraries")

# On Windows, copy the DLL next to built executables so they can find it at runtime
if(WIN32)
    set(WGPU_NATIVE_DLL "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll"
        CACHE INTERNAL "Path to wgpu_native.dll")
endif()

message(STATUS "wgpu-native: include = ${WGPU_INCLUDE_DIR}")
message(STATUS "wgpu-native: library = ${WGPU_LIBRARY}")
