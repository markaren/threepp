
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)

# GPU PBD granular material (PxPBDParticleSystem). Builds everywhere PhysX is
# found — the PBD headers ship with the SDK whether or not CUDA does — and
# reports + exits 0 at runtime when GPU dynamics is unavailable.
add_example(NAME "physx_granular_conveyor" SOURCES granular_conveyor.cpp LINK_IMGUI LINK_PHYSX)

# This demo reaches VulkanRenderer's own knobs (pin auto-exposure for a
# repeatable capture, read per-pass frame timings for the perf table) behind
# THREEPP_WITH_VULKAN. It needs no flag of its own: that macro is PUBLIC on the
# threepp target, so linking the library carries it.

# The same bridge for the GPU PBD grains (plans/particle-atmosphere.md F6):
# PhysX's solver positions are copied device→device into the Vulkan renderer's
# exported ParticleField position buffer — no DtoH pull, no host submit. Needs
# the Vulkan backend + the CUDA toolkit (driver API) at build time; without
# either, --feed dtod declines out loud and the demo keeps the pull path.
if (TARGET physx_granular_conveyor AND THREEPP_WITH_VULKAN)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_compile_definitions(physx_granular_conveyor PRIVATE THREEPP_PHYSX_CUDA_VK_INTEROP)
        target_link_libraries(physx_granular_conveyor PRIVATE CUDA::cuda_driver)
        message(STATUS "physx_granular_conveyor: CUDA->Vulkan particle interop enabled")
    endif ()
endif ()

# CUDA→Vulkan zero-copy interop for the soft-body tet positions (optional):
# PhysX's deformed positions are copied device→device into the Vulkan
# renderer's exported tet-skinning buffer — no host round-trip. Needs the
# Vulkan backend + the CUDA toolkit (driver API) at build time; without
# either, the example silently keeps the pinned-memory CPU bridge.
if (TARGET physx_softbody AND THREEPP_WITH_VULKAN)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_compile_definitions(physx_softbody PRIVATE THREEPP_PHYSX_CUDA_VK_INTEROP)
        target_link_libraries(physx_softbody PRIVATE CUDA::cuda_driver)
        message(STATUS "physx_softbody: CUDA->Vulkan zero-copy interop enabled")
    endif ()
endif ()
