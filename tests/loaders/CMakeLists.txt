
add_test_executable(ColladaLoader_test)
add_test_executable(ObjectAssetReference_test)
add_test_executable(DDSLoader_test)
add_test_executable(EXRLoader_test)
add_test_executable(Fontloader_test)
add_test_executable(GLTFLoader_test)
add_test_executable(ImageLoader_test)
add_test_executable(ModelLoader_test)
add_test_executable(OBJLoader_test)
add_test_executable(ObjectLoader_test)
add_test_executable(ObjectExporterSplat_test)
add_test_executable(SogLoader_test)
add_test_executable(SplatLoader_test)
add_test_executable(STLLoader_test)
add_test_executable(URDFLoader_test)
add_test_executable(Xacro_test)
add_test_executable(XacroUr_test)
add_test_executable(XacroFranka_test)
# Every case needs a description clone to expand — a Universal_Robots_ROS2_Description for
# the one (THREEPP_UR_DESCRIPTION), a franka_description for the other
# (THREEPP_FRANKA_DESCRIPTION). Without one Catch2 skips them all and exits 4, which is a
# ctest skip, not a failure.
set_tests_properties(XacroUr_test XacroFranka_test SogLoader_test PROPERTIES SKIP_RETURN_CODE 4)

add_subdirectory(svg)

# What a scene document costs to write and read back, per storage mode — not a
# ctest, run manually (see the header comment in SceneSerialize_bench.cpp).
add_executable(SceneSerialize_bench SceneSerialize_bench.cpp)
target_link_libraries(SceneSerialize_bench PRIVATE threepp)
