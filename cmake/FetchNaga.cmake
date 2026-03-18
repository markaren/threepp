# cmake/FetchNaga.cmake
# Downloads the naga CLI binary for GLSL -> WGSL offline translation.
# naga is part of the wgpu project; pre-built binaries are published on GitHub.
#
# Exposes: NAGA_EXECUTABLE (cache variable pointing to the naga binary)

include(FetchContent)

set(NAGA_VERSION "wgpu-25.0.2" CACHE STRING "naga release tag (from wgpu releases)")

# Check if naga is already available on PATH or in Cargo bin
find_program(NAGA_EXECUTABLE naga
    HINTS "$ENV{HOME}/.cargo/bin" "$ENV{USERPROFILE}/.cargo/bin"
)

if(NAGA_EXECUTABLE)
    message(STATUS "naga: found existing installation at ${NAGA_EXECUTABLE}")
    return()
endif()

# Determine platform triple
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(_naga_triple "x86_64-unknown-linux-gnu")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_naga_triple "aarch64-apple-darwin")
    else()
        set(_naga_triple "x86_64-apple-darwin")
    endif()
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(_naga_triple "x86_64-pc-windows-msvc")
else()
    message(FATAL_ERROR
        "naga: unsupported host platform '${CMAKE_HOST_SYSTEM_NAME}'. "
        "Install naga manually via 'cargo install naga-cli' and re-run CMake.")
endif()

# Construct filename and URL
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(_naga_filename "naga-${_naga_triple}.exe")
else()
    set(_naga_filename "naga-${_naga_triple}")
endif()

set(_naga_url
    "https://github.com/gfx-rs/wgpu/releases/download/${NAGA_VERSION}/${_naga_filename}")

message(STATUS "naga: fetching ${_naga_filename} (${NAGA_VERSION})")

# Download the single binary (no archive extraction)
FetchContent_Declare(
    naga_cli
    URL "${_naga_url}"
    DOWNLOAD_NO_EXTRACT TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(naga_cli)

# The downloaded file lands in the subbuild download dir.
# FetchContent with DOWNLOAD_NO_EXTRACT puts it at:
#   <FETCHCONTENT_BASE_DIR>/naga_cli-src/<filename>
set(_naga_download_dir "${FETCHCONTENT_BASE_DIR}/naga_cli-src")
set(_naga_bin "${_naga_download_dir}/${_naga_filename}")

if(NOT EXISTS "${_naga_bin}")
    message(FATAL_ERROR
        "naga: downloaded binary not found at expected path: ${_naga_bin}\n"
        "URL was: ${_naga_url}")
endif()

# Make executable on Unix
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    file(CHMOD "${_naga_bin}"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
endif()

set(NAGA_EXECUTABLE "${_naga_bin}" CACHE FILEPATH "Path to naga CLI binary" FORCE)
message(STATUS "naga: executable = ${NAGA_EXECUTABLE}")
