
add_example(NAME "tps_shooter" SOURCES main.cpp LINK_PHYSX)

# THREEPP_WITH_VULKAN is PRIVATE on the threepp lib and only re-exported via
# the imgui helper target, which this demo doesn't link — mirror it here so
# main.cpp can guard the VulkanRenderer-only code (motion blur, screenshots).
if (TARGET tps_shooter AND THREEPP_WITH_VULKAN)
    target_compile_definitions(tps_shooter PRIVATE TPS_SHOOTER_WITH_VULKAN=1)
endif ()
