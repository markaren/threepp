
add_test_executable(ColladaLoader_test)
add_test_executable(ObjectAssetReference_test)
add_test_executable(DDSLoader_test)
add_test_executable(EXRLoader_test)
add_test_executable(Fontloader_test)
add_test_executable(GLTFLoader_test)
add_test_executable(AsyncGroup_test)
add_test_executable(CubeTextureLoader_test)
add_test_executable(ImageLoader_test)
add_test_executable(ModelLoader_test)
add_test_executable(OBJLoader_test)
add_test_executable(ObjectLoader_test)
add_test_executable(ObjectExporterSplat_test)
add_test_executable(SceneArchive_test)
add_test_executable(SogLoader_test)
add_test_executable(SplatLoader_test)
add_test_executable(STLLoader_test)
add_test_executable(URDFLoader_test)
add_test_executable(Xacro_test)
add_test_executable(XacroUr_test)
if (THREEPP_WITH_USD)
    add_test_executable(USDLoader_test)
endif ()
# Every XacroUr case needs a Universal_Robots_ROS2_Description clone to expand
# (THREEPP_UR_DESCRIPTION). Without one each case skips with this message, which is a
# ctest skip, not a failure. Not SKIP_RETURN_CODE 4: Catch2 3.4 exits 4 when every case
# skipped, but also when exactly four assertions failed.
set_tests_properties(XacroUr_test PROPERTIES
        SKIP_REGULAR_EXPRESSION "set THREEPP_UR_DESCRIPTION to a Universal_Robots_ROS2_Description clone")

add_subdirectory(svg)

# What a scene document costs to write and read back, per storage mode — not a
# ctest, run manually (see the header comment in SceneSerialize_bench.cpp).
add_executable(SceneSerialize_bench SceneSerialize_bench.cpp)
target_link_libraries(SceneSerialize_bench PRIVATE threepp)
