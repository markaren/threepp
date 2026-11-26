
add_example(NAME "helpers" WEB)
add_example(NAME "camera_helper" WEB)
add_example(NAME "depth_sensor" LINK_IMGUI WEB)
add_example(NAME "lidar" LINK_IMGUI WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr@data/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr"
        "${THREEPP_DATA_DIR}/textures/pbr/forrest_ground_01@data/textures/pbr/forrest_ground_01"
        "${THREEPP_DATA_DIR}/textures/pbr/rock_wall_10@data/textures/pbr/rock_wall_10"
        "${THREEPP_DATA_DIR}/models/gltf/rock_moss_set_02@data/models/gltf/rock_moss_set_02"
        "${THREEPP_DATA_DIR}/models/gltf/ford_mustang_1967/1967_ford_mustang_shelby_cobra_gt500.glb@data/models/gltf/ford_mustang_1967/1967_ford_mustang_shelby_cobra_gt500.glb"
        "${THREEPP_DATA_DIR}/models/gltf/Soldier.glb@data/models/gltf/Soldier.glb"
)
add_example(NAME "lidar_slam" LINK_IMGUI WEB)
