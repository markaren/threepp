# V-HACD convex decomposition, as its own tiny static library.
#
# The only consumers are the editor's physics play session and the PhysX-backed
# tests, both of which are themselves gated on the SDK. V-HACD itself does NOT
# depend on PhysX, so this target does not find_package or link the SDK — doing
# so at the top level created a non-GLOBAL imported target that later collided
# with the examples' own unguarded find_package. Instead it self-gates on
# VHACD.h being installed, which only happens when the vcpkg `physx` feature was
# requested (that feature is what carries v-hacd; see vcpkg.json). So a build
# without PhysX never installs the header and this library is never built.
#
# It is a separate target, not part of libthreepp, for two reasons: libthreepp
# stays free of the heavy VHACD.h implementation (which the `#define
# ENABLE_VHACD_IMPLEMENTATION` in ConvexDecomposition.cpp emits into exactly one
# TU), and the target can be shared by the editor and every PhysX test without
# either having to compile the source itself.

if (NOT TARGET threepp_convex_decomp)

    # v-hacd is header-only and ships no CMake config; vcpkg just installs the
    # header. Locate it the way vcpkg's own usage note prescribes.
    find_path(V_HACD_INCLUDE_DIRS "VHACD.h")

    if (V_HACD_INCLUDE_DIRS)
        add_library(threepp_convex_decomp STATIC
                "${PROJECT_SOURCE_DIR}/src/threepp/extras/physx/ConvexDecomposition.cpp")
        target_include_directories(threepp_convex_decomp
                PUBLIC "${PROJECT_SOURCE_DIR}/include"
                PRIVATE "${V_HACD_INCLUDE_DIRS}")
        target_compile_features(threepp_convex_decomp PUBLIC cxx_std_20)
        if (MSVC)
            # VHACD.h's implementation is one enormous object; it exceeds the
            # default COFF section limit exactly like the pybind TUs do.
            target_compile_options(threepp_convex_decomp PRIVATE /bigobj)
        endif ()
        message(STATUS "threepp: convex decomposition (V-HACD) enabled")
    else ()
        message(STATUS "threepp: VHACD.h not found, convex decomposition disabled")
    endif ()

endif ()
