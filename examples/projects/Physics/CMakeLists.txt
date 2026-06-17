
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)

find_package(CUDAToolkit QUIET)

# CUDA→Vulkan zero-copy interop for the soft-body tet positions (optional):
# PhysX's deformed positions are copied device→device into the Vulkan
# renderer's exported tet-skinning buffer — no host round-trip. Needs the
# Vulkan backend + the CUDA toolkit (driver API) at build time; without
# either, the example silently keeps the pinned-memory CPU bridge.
if (TARGET physx_softbody AND THREEPP_WITH_VULKAN)
    if (CUDAToolkit_FOUND)
        target_compile_definitions(physx_softbody PRIVATE THREEPP_PHYSX_CUDA_VK_INTEROP)
        target_link_libraries(physx_softbody PRIVATE CUDA::cuda_driver)
        message(STATUS "physx_softbody: CUDA->Vulkan zero-copy interop enabled")
    endif ()
endif ()
