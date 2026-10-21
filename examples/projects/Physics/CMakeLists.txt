
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)

# GPU PBD granular material (PxPBDParticleSystem). Builds everywhere PhysX is
# found — the PBD headers ship with the SDK whether or not CUDA does — and
# reports + exits 0 at runtime when GPU dynamics is unavailable. Where the SDK
# has no CUDA backend at all (PX_SUPPORT_GPU_PHYSX is 0: macOS) this demo and
# physx_softbody compile to a main() that says so; see THREEPP_PHYSX_GPU in
# PhysxWorld.hpp.
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

# CUDA zero-copy interop for the soft-body tet positions (optional): PhysX's
# deformed positions are copied device→device into the renderer's own storage,
# with no host round-trip. On GL that is the tet texture, registered with CUDA;
# on Vulkan the renderer's exported tet-skinning buffer (needs the Vulkan
# backend). Both need the CUDA toolkit (driver API) at build time; without it
# the example silently keeps the pinned-memory CPU bridge.
if (TARGET physx_softbody)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(physx_softbody PRIVATE CUDA::cuda_driver)
        target_compile_definitions(physx_softbody PRIVATE THREEPP_PHYSX_CUDA_GL_INTEROP)
        message(STATUS "physx_softbody: CUDA->GL zero-copy interop enabled")
        if (THREEPP_WITH_VULKAN)
            target_compile_definitions(physx_softbody PRIVATE THREEPP_PHYSX_CUDA_VK_INTEROP)
            message(STATUS "physx_softbody: CUDA->Vulkan zero-copy interop enabled")
        endif ()
    endif ()
endif ()
