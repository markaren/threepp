
add_example(NAME "physx_demo" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_softbody" SOURCES softbody.cpp LINK_IMGUI LINK_PHYSX)
add_example(NAME "physx_fish_softbody" SOURCES fish_softbody.cpp LINK_IMGUI LINK_PHYSX LINK_MESHOPT)
