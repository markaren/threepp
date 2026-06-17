
find_package(CUDAToolkit QUIET)

# CUDA→Vulkan zero-copy interop for the soft-body tet positions (optional):
# PhysX's deformed positions are copied device→device into the Vulkan
# renderer's exported tet-skinning buffer — no host round-trip. Needs the
# Vulkan backend + the CUDA toolkit (driver API) at build time; without
# either, the example silently keeps the pinned-memory CPU bridge.

add_example(NAME "physx_fish" SOURCES fish_softbody.cpp LINK_IMGUI LINK_PHYSX LINK_MESHOPT)
if (TARGET physx_fish)
    target_include_directories(physx_fish PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
    if (CUDAToolkit_FOUND)
        target_link_libraries(physx_fish PRIVATE CUDA::cuda_driver)
        target_compile_definitions(physx_fish PRIVATE THREEPP_PHYSX_CUDA_GL_INTEROP THREEPP_PHYSX_CUDA_VK_INTEROP)
        message(STATUS "physx_fish: CUDA interop enabled")
    endif ()
endif ()

add_example(NAME "conveyor_designer" SOURCES conveyor_designer.cpp LINK_IMGUI)
if (TARGET conveyor_designer)
    target_include_directories(conveyor_designer PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
    if (WIN32)
        target_link_libraries(conveyor_designer PRIVATE opengl32)
    endif ()
endif ()
