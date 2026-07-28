
add_test_executable(ColladaLoader_test)
add_test_executable(ObjectAssetReference_test)
add_test_executable(DDSLoader_test)
add_test_executable(Fontloader_test)
add_test_executable(GLTFLoader_test)
add_test_executable(ImageLoader_test)
add_test_executable(OBJLoader_test)
add_test_executable(ObjectLoader_test)
add_test_executable(STLLoader_test)
add_test_executable(URDFLoader_test)

add_subdirectory(svg)

# What a scene document costs to write and read back, per storage mode — not a
# ctest, run manually (see the header comment in SceneSerialize_bench.cpp).
add_executable(SceneSerialize_bench SceneSerialize_bench.cpp)
target_link_libraries(SceneSerialize_bench PRIVATE threepp)
