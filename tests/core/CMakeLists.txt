
add_test_executable(BufferAttribute_test)
add_test_executable(EventDispatcher_test)
add_test_executable(InstanceIds_test)
add_test_executable(Layers_test)
add_test_executable(MatrixWorldForceEquivalence_test)
add_test_executable(Object3D_test)
add_test_executable(Raycaster_test)
add_test_executable(Uniform_test)

# CPU microbenchmark for the transform hierarchy — not a ctest, run manually
# (see the header comment in Object3D_bench.cpp).
add_executable(Object3D_bench Object3D_bench.cpp)
target_link_libraries(Object3D_bench PRIVATE threepp)
