include(FetchContent)

if (THREEPP_WITH_USD)
    message(STATUS "Fetching tinyusdz...")

    set(TINYUSDZ_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_AUDIO         OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_EXR           OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_COLORIO       OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_PYTHON        OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDMTLX       OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDVOX        OFF CACHE BOOL "" FORCE)

    # Local patches applied to the pinned tinyusdz tag.
    # PATCH_COMMAND runs once after FetchContent populates the source tree.
    # `git reset --hard HEAD` first makes re-runs idempotent should the source
    # tree need to be re-patched.
    find_package(Git REQUIRED)
    set(_TINYUSDZ_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/patches/tinyusdz")

    FetchContent_Declare(
        tinyusdz
        GIT_REPOSITORY https://github.com/lighttransport/tinyusdz.git
        GIT_TAG        v0.9.1
        GIT_SHALLOW    TRUE
        PATCH_COMMAND  ${GIT_EXECUTABLE} reset --hard HEAD
                COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace
                        "${_TINYUSDZ_PATCH_DIR}/0001-real-world-asset-compat.patch"
    )
    FetchContent_MakeAvailable(tinyusdz)

    # tinyusdz_static's INTERFACE_INCLUDE_DIRECTORIES may contain raw source
    # paths, which CMake rejects when the target is exported. Clear them here —
    # threepp links to tinyusdz_static privately and adds the paths itself via
    # target_include_directories in src/CMakeLists.txt. The install() rule
    # below deploys the headers so installed consumers can find them.
    set_target_properties(tinyusdz_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tinyusdz>"
    )
endif ()

if (THREEPP_WITH_RLTOOLS)
    message(STATUS "Fetching rl-tools...")
    FetchContent_Declare(
        rl_tools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG        b32d9985c65a5e098a6bbf190fd994962d288b99  # pinned for reproducible builds
    )
    # Populate the source only — do NOT add_subdirectory(rl-tools). RLtools is
    # header-only, and its CMakeLists runs an autodetect pass that would pull in
    # optional BLAS/HDF5/CUDA backends (changing the numerics and adding link
    # dependencies). We want the pure, dependency-free CPU build, so we expose a
    # minimal INTERFACE target pointing at its include/ directory ourselves.
    # CMP0169 OLD lets us call FetchContent_Populate directly on CMake 4.x.
    # For a local checkout: -DFETCHCONTENT_SOURCE_DIR_RL_TOOLS=<dir>.
    cmake_policy(SET CMP0169 OLD)
    FetchContent_GetProperties(rl_tools)
    if (NOT rl_tools_POPULATED)
        FetchContent_Populate(rl_tools)
    endif ()
    add_library(rltools INTERFACE)
    target_include_directories(rltools INTERFACE "${rl_tools_SOURCE_DIR}/include")
    target_compile_features(rltools INTERFACE cxx_std_17)
    add_library(RLtools::headers ALIAS rltools)
endif ()

if (THREEPP_WITH_FBX)
    message(STATUS "Fetching OpenFBX...")
    FetchContent_Declare(
        openfbx
        GIT_REPOSITORY https://github.com/nem0/OpenFBX.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    # Populate source only — do NOT call add_subdirectory (OpenFBX's CMakeLists
    # has an unresolvable install-export dependency on libdeflate_static).
    # CMP0169 OLD lets us call FetchContent_Populate directly without CMake 4.x
    # routing through add_subdirectory.
    cmake_policy(SET CMP0169 OLD)
    FetchContent_GetProperties(openfbx)
    if (NOT openfbx_POPULATED)
        FetchContent_Populate(openfbx)
    endif ()
endif ()
