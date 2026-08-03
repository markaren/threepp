# Fix the static-PhysX link order for single-pass linkers (GNU ld).
#
# The vcpkg port's imported target puts the core archive first: the ::sdk
# target IS libPhysX_static_64.a, and the seven component archives hang off
# its INTERFACE_LINK_LIBRARIES, so every consumer's link line reads
#
#   libPhysX_static_64.a ... libPhysXExtensions_static_64.a ... libPhysXVehicle_static_64.a
#
# GNU ld walks that line once. PhysXExtensions' RepX serializers reference the
# Px*GeneratedInfo metadata constructors compiled into the CORE archive, which
# ld has already left behind — undefined reference, at the link of the first
# executable that pulls a serializer object in. PhysXVehicle sits after
# Common/Foundation and fails the same way once vehicle objects are pulled.
# MSVC's linker resolves across all inputs regardless of order, which is why
# Windows never sees this.
#
# The fix is the one NVIDIA's own Linux sample builds use: wrap the archives
# in --start-group/--end-group so ld re-scans until fixed point. CMake's
# spelling of that is the $<LINK_GROUP:RESCAN,...> generator expression
# (3.24+). We rewrite the ::sdk target's INTERFACE_LINK_LIBRARIES to a single
# RESCAN group holding the core archive plus the components; the core then
# appears twice on the line (once as the target's own location, once in the
# group), which is harmless — the group copy is the one that resolves the
# backward references.
#
# Call threepp_fix_physx_static_link_order() immediately after EVERY
# find_package(unofficial-omniverse-physx-sdk) site. It cannot be centralized:
# the port's imported targets are directory-scoped (not GLOBAL), so each
# guarded find_package creates its own per-directory copy that a root-level
# call can never see — a deferred root call finds no target at all and
# silently does nothing (learned the hard way). The function is idempotent
# and a no-op when PhysX is absent, dynamic, or the linker is not single-pass.

function(threepp_fix_physx_static_link_order)
    if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
        return()
    endif ()

    # Idempotent: a target inherited from an ancestor scope is already fixed.
    get_target_property(current unofficial::omniverse-physx-sdk::sdk INTERFACE_LINK_LIBRARIES)
    if (NOT current)
        return()
    endif ()
    if (current MATCHES "LINK_GROUP")
        return()
    endif ()

    # Only single-pass linkers need this. MSVC and Apple ld resolve across
    # all inputs; MinGW would need it but the port does not support MinGW.
    if (MSVC OR APPLE)
        return()
    endif ()

    # Dynamic PhysX resolves at load time — only the static archives care.
    get_target_property(core unofficial::omniverse-physx-sdk::sdk IMPORTED_LOCATION_RELEASE)
    if (NOT core MATCHES "_static_")
        return()
    endif ()

    if (CMAKE_VERSION VERSION_LESS 3.24)
        message(WARNING
                "threepp: static PhysX on a GNU-ld platform needs CMake >= 3.24 "
                "($<LINK_GROUP:RESCAN>) to fix the archive link order; expect "
                "undefined Px*GeneratedInfo references at executable link time.")
        return()
    endif ()

    set(components "${current}")

    # The port's config appends PhysXVehicle2 to the component list on WIN32
    # only, but the x64-linux build installs the archive too — and the PxVehicle2
    # API that PhysxVehicleBase drives (PxVehicleEngineDrivetrainUpdate and
    # friends) lives in it. Without this, every vehicle-linking executable dies
    # with undefined vehicle2:: references on Linux. No imported target exists
    # for it here, so the raw archive path joins the group.
    get_filename_component(libdir "${core}" DIRECTORY)
    if (EXISTS "${libdir}/libPhysXVehicle2_static_64.a")
        list(APPEND components "${libdir}/libPhysXVehicle2_static_64.a")
    endif ()

    list(JOIN components "," joined)
    set_target_properties(unofficial::omniverse-physx-sdk::sdk PROPERTIES
            INTERFACE_LINK_LIBRARIES
            "$<LINK_GROUP:RESCAN,$<TARGET_FILE:unofficial::omniverse-physx-sdk::sdk>,${joined}>")
endfunction()
