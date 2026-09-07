add_example(NAME "robot_cell" SOURCES main.cpp LINK_PHYSX)

# On Vulkan builds the perception sensor can be ray-traced through the
# renderer's TLAS (PathTracedLidarSensor camera mode) instead of rasterised.
if (TARGET robot_cell AND THREEPP_WITH_VULKAN)
    target_compile_definitions(robot_cell PRIVATE ROBOT_CELL_WITH_VULKAN=1)
endif ()
