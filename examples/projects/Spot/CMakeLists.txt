add_example(NAME "spot_play" SOURCES spot_play.cpp LINK_PHYSX)

# Parity self-test for the exported policy: pure std (no threepp/PhysX), checks
# SpotPolicy reproduces the torch forward against spot_policy_ref.bin. Run as
# `spot_policy_selftest <dir-with-tpnn-and-ref>`. Both files are version-controlled
# (see .gitignore for why), so this runs from a fresh checkout; regenerate them with
# export_spot_policy.py after retraining.
add_executable(spot_policy_selftest spot_policy_selftest.cpp)
