# CMake package for the physx-wasm overlay port. It defines the same targets as
# the upstream physx port's config, so find_package(unofficial-omniverse-physx-sdk)
# and unofficial::omniverse-physx-sdk::sdk work unchanged on wasm32. Differences:
# release libraries only (the headers do not depend on the library's build type
# outside Windows), the 32-bit library names wasm32 produces, PhysXVehicle2 in
# the component list on every platform, SIMD compile options for consumers, and
# no GPU library targets (there is no CUDA backend).

if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    get_filename_component(z_physx_wasm_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    set(z_physx_wasm_libdir "${z_physx_wasm_prefix}/lib")

    find_library(OMNIVERSE-PHYSX-SDK_LIBRARY_RELEASE
            NAMES PhysX_static PhysX_static_32 PhysX_static_64
            PATHS "${z_physx_wasm_libdir}" NO_DEFAULT_PATH REQUIRED)

    add_library(unofficial::omniverse-physx-sdk::sdk STATIC IMPORTED)
    set_target_properties(unofficial::omniverse-physx-sdk::sdk PROPERTIES
            IMPORTED_LOCATION "${OMNIVERSE-PHYSX-SDK_LIBRARY_RELEASE}"
            INTERFACE_INCLUDE_DIRECTORIES "${z_physx_wasm_prefix}/include/physx"
            # PX_SSE2 in the public headers follows __SSE2__, and the libraries
            # were built with it; consumers must agree.
            INTERFACE_COMPILE_OPTIONS "-msimd128;-msse2")

    foreach (name IN ITEMS
            PhysXExtensions
            PhysXPvdSDK
            PhysXCharacterKinematic
            PhysXCooking
            PhysXCommon
            PhysXFoundation
            PhysXVehicle
            PhysXVehicle2)
        find_library(OMNIVERSE_${name}_LIBRARY_RELEASE
                NAMES ${name}_static ${name}_static_32 ${name}_static_64
                PATHS "${z_physx_wasm_libdir}" NO_DEFAULT_PATH REQUIRED)
        add_library(unofficial::omniverse-physx-sdk::${name} STATIC IMPORTED)
        set_target_properties(unofficial::omniverse-physx-sdk::${name} PROPERTIES
                IMPORTED_LOCATION "${OMNIVERSE_${name}_LIBRARY_RELEASE}")
        set_property(TARGET unofficial::omniverse-physx-sdk::sdk APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES unofficial::omniverse-physx-sdk::${name})
    endforeach ()

    unset(z_physx_wasm_prefix)
    unset(z_physx_wasm_libdir)
endif ()
