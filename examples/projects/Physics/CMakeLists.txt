
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)

# GPU PBD granular material (PxPBDParticleSystem). Builds everywhere PhysX is
# found — the PBD headers ship with the SDK whether or not CUDA does — and
# reports + exits 0 at runtime when GPU dynamics is unavailable.
add_example(NAME "physx_granular_conveyor" SOURCES granular_conveyor.cpp LINK_PHYSX)

# THREEPP_WITH_VULKAN is PRIVATE to the library target, so an example that wants
# to reach VulkanRenderer's own knobs (pin auto-exposure for a repeatable
# capture, read per-pass frame timings for the perf table) has to be told
# separately. Without it the demo still runs on the Vulkan backend — it just
# cannot see the backend-specific API, and falls back to wall-clock timing.
if (TARGET physx_granular_conveyor AND THREEPP_WITH_VULKAN)
    target_compile_definitions(physx_granular_conveyor PRIVATE THREEPP_WITH_VULKAN=1)
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
