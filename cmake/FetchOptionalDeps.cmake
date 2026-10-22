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
    # PATCH_COMMAND runs after FetchContent populates the source tree.
    # `git reset --hard HEAD` first makes re-runs idempotent should the source
    # tree need to be re-patched.
    #
    # It must not run on every configure, though: the reset and the apply
    # rewrite the patched files, and their fresh mtimes recompile those and
    # everything that includes crate-format.hh, then relink tinyusdz_static,
    # after ANY CMake regeneration. FetchContent chains the patch step to the
    # git update step, which is always dirty unless updates are disconnected.
    # A pinned tag needs no update check, so UPDATE_DISCONNECTED takes the step
    # out of the chain. The patch's hash rides in the command line because the
    # patch step re-runs exactly when that changes, so editing the patch still
    # re-applies it (nothing else tells the step the file changed).
    find_package(Git REQUIRED)
    set(_TINYUSDZ_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/patches/tinyusdz")
    set(_TINYUSDZ_PATCH "${_TINYUSDZ_PATCH_DIR}/0001-real-world-asset-compat.patch")
    file(SHA256 "${_TINYUSDZ_PATCH}" _TINYUSDZ_PATCH_SHA256)

    FetchContent_Declare(
        tinyusdz
        GIT_REPOSITORY https://github.com/lighttransport/tinyusdz.git
        GIT_TAG        v0.9.1
        GIT_SHALLOW    TRUE
        UPDATE_DISCONNECTED TRUE
        PATCH_COMMAND  ${GIT_EXECUTABLE} reset --hard HEAD
                COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace "${_TINYUSDZ_PATCH}"
                COMMAND ${CMAKE_COMMAND} -E echo "tinyusdz patch sha256 ${_TINYUSDZ_PATCH_SHA256}"
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
