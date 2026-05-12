
add_example(NAME "vulkan_gltf_samples" LINK_IMGUI)
add_example(NAME "vulkan_showcase" LINK_IMGUI)
add_example(NAME "vulkan_lights" LINK_IMGUI)
add_example(NAME "vulkan_fog" LINK_IMGUI)
add_example(NAME "vulkan_ocean" LINK_IMGUI)
add_example(NAME "vulkan_denoise" LINK_IMGUI)
add_example(NAME "vulkan_animation" LINK_IMGUI)

if (THREEPP_WITH_FBX)
    add_example(NAME "vulkan_bistro" LINK_IMGUI)
endif ()