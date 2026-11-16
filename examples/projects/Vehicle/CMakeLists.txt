
add_example(NAME "physx_vehicle" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)

# The browser-sized Vehicle2 demo: primitives for the course, so the car model
# (~5 MB) is the only file the page embeds.
add_example(NAME "vehicle_playground" SOURCES playground.cpp LINK_IMGUI LINK_PHYSX WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/models/gltf/2015_land-rover_range_rover_evoque_coupe@data/models/gltf/2015_land-rover_range_rover_evoque_coupe"
)
