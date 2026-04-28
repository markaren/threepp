# cmake/FetchNaga.cmake
# Locates the naga CLI binary for GLSL -> WGSL offline translation.
# naga is part of the wgpu project; install via: cargo install naga-cli
#
# Exposes: NAGA_EXECUTABLE (cache variable pointing to the naga binary)

find_program(NAGA_EXECUTABLE naga
    HINTS "$ENV{HOME}/.cargo/bin" "$ENV{USERPROFILE}/.cargo/bin"
)

if(NAGA_EXECUTABLE)
    message(STATUS "naga: found at ${NAGA_EXECUTABLE}")
else()
    message(FATAL_ERROR
        "naga: not found on PATH or in ~/.cargo/bin.\n"
        "Install it with: cargo install naga-cli\n"
        "Then re-run CMake.")
endif()