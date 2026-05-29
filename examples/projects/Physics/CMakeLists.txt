
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)

add_example(NAME "physx_fish" SOURCES fish_softbody.cpp LINK_IMGUI LINK_PHYSX)
if (TARGET physx_fish)
    target_include_directories(physx_fish PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
endif ()

add_example(NAME "conveyor_designer" SOURCES conveyor_designer.cpp LINK_IMGUI)
if (TARGET conveyor_designer)
    target_include_directories(conveyor_designer PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
    if (WIN32)
        target_link_libraries(conveyor_designer PRIVATE opengl32)
    endif ()
endif ()