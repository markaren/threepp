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

    FetchContent_Declare(
        tinyusdz
        GIT_REPOSITORY https://github.com/lighttransport/tinyusdz.git
        GIT_TAG        v0.9.1
        GIT_SHALLOW    TRUE
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
