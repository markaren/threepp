
add_example(NAME "fps_demo" SOURCES main.cpp LINK_PHYSX)

# THREEPP_WITH_VULKAN is PRIVATE on the threepp lib and only re-exported via
# the imgui helper target, which this demo doesn't link — mirror it here so
# main.cpp can guard the VulkanRenderer-only code (motion blur, frame timings).
if (TARGET fps_demo AND THREEPP_WITH_VULKAN)
    target_compile_definitions(fps_demo PRIVATE FPS_DEMO_WITH_VULKAN=1)
endif ()
