
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)

add_example(NAME "physx_fish" SOURCES fish_softbody.cpp LINK_IMGUI LINK_PHYSX LINK_MESHOPT)
if (TARGET physx_fish)
    target_include_directories(physx_fish PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
    # Optional CUDA-GL interop: feed deformed soft-body tets straight into the GL
    # tet texture device-side, skipping the GPU->CPU->GPU bridge. Falls back to the
    # CPU bridge when the CUDA toolkit isn't found.
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_include_directories(physx_fish PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(physx_fish PRIVATE CUDA::cuda_driver)
        target_compile_definitions(physx_fish PRIVATE THREEPP_PHYSX_CUDA_GL_INTEROP)
        message(STATUS "physx_fish: CUDA-GL interop enabled")
    else ()
        message(STATUS "physx_fish: CUDAToolkit not found - CUDA-GL interop disabled (CPU bridge)")
    endif ()
endif ()

add_example(NAME "conveyor_designer" SOURCES conveyor_designer.cpp LINK_IMGUI)
if (TARGET conveyor_designer)
    target_include_directories(conveyor_designer PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
    if (WIN32)
        target_link_libraries(conveyor_designer PRIVATE opengl32)
    endif ()
endif ()