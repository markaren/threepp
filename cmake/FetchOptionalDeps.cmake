include(FetchContent)

if (THREEPP_WITH_USD)
    message(STATUS "Fetching LightUSD...")

    # USDLoader reads through LightUSD's `next` product, its default since
    # 1.0.0: the next core (lightusd_next: USDA/USDC/USDZ, next::pcp
    # composition, which composes the layered Isaac Sim / SimReady assets) and
    # its render scene converter (tydra_next). Its CLI tools, tests, examples,
    # MCP server and the optional GPU preview kernels (which want glslang and
    # the Vulkan SDK at build time) are not needed here.
    set(LIGHTUSD_NATIVE_PRODUCT      next CACHE STRING "" FORCE)
    set(LIGHTUSD_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_BUILD_TOOLS         OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_NEXT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_NEXT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_NEXT_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_USE_CCACHE          OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_TYDRA          ON  CACHE BOOL "" FORCE)
    set(LIGHTUSD_WITH_MCP_SERVER     OFF CACHE BOOL "" FORCE)
    set(LIGHTUSD_PREVIEW_GPU_COMPUTE OFF CACHE BOOL "" FORCE)
    # The next core's parallel paths (on by default for native builds in 1.0.0)
    # compose the same file differently from run to run: the SimReady
    # RackLarge_A1 came back with 8 to 11 distinct meshes and cartons missing
    # their labels, where the serial path gives the same scene every time,
    # and no faster here (the warehouse module loads in 17.5 s either way).
    set(LIGHTUSD_NEXT_ENABLE_THREAD  OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        lightusd
        GIT_REPOSITORY https://github.com/lighttransport/LightUSD.git
        GIT_TAG        v1.0.0
        GIT_SHALLOW    TRUE
        UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(lightusd)

    # The LightUSD libraries threepp links: tydra_next and what it links,
    # transitively. A static threepp exports them alongside itself (see the
    # install rules in the top-level CMakeLists.txt).
    set(_lightusd_targets "")
    set(_lightusd_queue tydra_next)
    while (_lightusd_queue)
        list(POP_FRONT _lightusd_queue _lightusd_target)
        if (NOT TARGET "${_lightusd_target}" OR _lightusd_target IN_LIST _lightusd_targets)
            continue()
        endif ()
        get_target_property(_lightusd_imported ${_lightusd_target} IMPORTED)
        if (_lightusd_imported)
            continue()
        endif ()
        get_target_property(_lightusd_aliased ${_lightusd_target} ALIASED_TARGET)
        if (_lightusd_aliased)
            list(APPEND _lightusd_queue ${_lightusd_aliased})
            continue()
        endif ()
        list(APPEND _lightusd_targets ${_lightusd_target})
        foreach (_lightusd_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_lightusd_deps ${_lightusd_target} ${_lightusd_property})
            if (_lightusd_deps)
                list(APPEND _lightusd_queue ${_lightusd_deps})
            endif ()
        endforeach ()
    endwhile ()
    set(THREEPP_LIGHTUSD_TARGETS "${_lightusd_targets}" CACHE INTERNAL "LightUSD libraries threepp links")

    # An exported target may not publish raw source or build paths as include
    # directories: keep any such path for the build only; the install() rule
    # in the top-level CMakeLists.txt deploys the headers for installed
    # consumers.
    foreach (_lightusd_target IN LISTS _lightusd_targets)
        get_target_property(_lightusd_includes ${_lightusd_target} INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT _lightusd_includes)
            continue()
        endif ()
        set(_lightusd_fixed "")
        set(_lightusd_changed FALSE)
        foreach (_lightusd_dir IN LISTS _lightusd_includes)
            if (_lightusd_dir MATCHES "^\\$<")
                list(APPEND _lightusd_fixed "${_lightusd_dir}")
            else ()
                list(APPEND _lightusd_fixed "$<BUILD_INTERFACE:${_lightusd_dir}>")
                set(_lightusd_changed TRUE)
            endif ()
        endforeach ()
        if (_lightusd_changed)
            list(APPEND _lightusd_fixed "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/lightusd>")
            set_target_properties(${_lightusd_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_lightusd_fixed}")
        endif ()
    endforeach ()
endif ()
