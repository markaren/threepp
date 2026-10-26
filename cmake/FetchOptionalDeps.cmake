include(FetchContent)

if (THREEPP_WITH_USD)
    message(STATUS "Fetching LightUSD...")

    set(LIGHTUSD_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_BUILD_TOOLS        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_BUILD_BENCHMARKS   OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_USE_CCACHE         OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_PRODUCTION_BUILD   ON  CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_AUDIO         OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_EXR           OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_PYTHON        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_C_API         OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_USD_TO_GLTF   OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_USDOBJ        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_USDFBX        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_USDVOX        OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_USDVOL        OFF CACHE BOOL "" FORCE)
    # USDLoader reads through the `next` pipeline (next::pcp composition +
    # tydra::next): the classic flatten path does not compose the layered
    # Isaac Sim / SimReady assets (references of references, sublayer-only
    # wrappers, internal references inside a referenced layer).
    set(LIGHTUSD_USE_NEXT_PCP_LARGE_SCENE ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        lightusd
        GIT_REPOSITORY https://github.com/lighttransport/LightUSD.git
        GIT_TAG        v1.0.0-rc4
        GIT_SHALLOW    TRUE
        UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(lightusd)

    # LightUSD only builds tydra::next for its own examples. It also uses the
    # classic library (image loading, MaterialX DOM, tangent generation)
    # without declaring it.
    if (NOT TARGET tydra_next)
        add_subdirectory("${lightusd_SOURCE_DIR}/src/tydra/next" "${lightusd_BINARY_DIR}/tydra_next")
    endif ()
    target_link_libraries(tydra_next PUBLIC lightusd_static)

    # lightusd_static and its texture tools publish raw source paths as
    # INTERFACE_INCLUDE_DIRECTORIES, which CMake rejects when the targets are
    # exported. Keep them for the build only; the install() rule in the
    # top-level CMakeLists.txt deploys the headers for installed consumers.
    foreach (_lightusd_target lightusd_static lightusd_textools)
        if (TARGET ${_lightusd_target})
            get_target_property(_lightusd_includes ${_lightusd_target} INTERFACE_INCLUDE_DIRECTORIES)
            if (NOT _lightusd_includes)
                set(_lightusd_includes "")
            endif ()
            list(TRANSFORM _lightusd_includes PREPEND "$<BUILD_INTERFACE:")
            list(TRANSFORM _lightusd_includes APPEND ">")
            set_target_properties(${_lightusd_target} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES
                "${_lightusd_includes};$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/lightusd>"
            )
        endif ()
    endforeach ()
endif ()
