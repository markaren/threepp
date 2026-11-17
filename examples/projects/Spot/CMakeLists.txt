
# The landscape's assets come from threepp_data (all CC0, Poly Haven) and the robot
# from its urdf/spot, which must travel with its LICENSE and NOTICE.md: embedding
# the whole directory carries both.
add_example(NAME "spot_play" SOURCES spot_play.cpp LINK_PHYSX LINK_IMGUI WEB WEB_EMBED
        "${CMAKE_CURRENT_SOURCE_DIR}/spot_policy.tpnn@spot_policy.tpnn"
        "${THREEPP_DATA_DIR}/urdf/spot@data/urdf/spot"
        "${THREEPP_DATA_DIR}/textures/pbr@data/textures/pbr"
        "${THREEPP_DATA_DIR}/models/gltf/rock_moss_set_02@data/models/gltf/rock_moss_set_02"
        "${THREEPP_DATA_DIR}/models/fjord/trees@data/models/fjord/trees"
        "${THREEPP_DATA_DIR}/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr@data/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr"
)

# Parity self-test for the exported policy: pure std (no threepp/PhysX), checks
# SpotPolicy reproduces the torch forward against spot_policy_ref.bin. Run as
# `spot_policy_selftest <dir-with-tpnn-and-ref>`. Both files are version-controlled
# (see .gitignore for why), so this runs from a fresh checkout; regenerate them with
# export_spot_policy.py after retraining. Desktop only: it reads files from a path.
if (NOT EMSCRIPTEN)
    add_executable(spot_policy_selftest spot_policy_selftest.cpp)
endif ()
