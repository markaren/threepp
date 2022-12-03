link_libraries(threepp Catch2::Catch2)

add_executable(constants_test constants_test.cpp)
add_test(NAME constants_test COMMAND constants_test)

add_executable(Vector3_test Vector3_test.cpp)
add_test(NAME Vector3_test COMMAND Vector3_test)

add_executable(Matrix4_test Matrix4_test.cpp)
add_test(NAME Matrix4_test COMMAND Matrix4_test)

add_executable(Object3D_test Object3D_test.cpp)
add_test(NAME Object3D_test COMMAND Object3D_test)

add_executable(StringUtils_test StringUtils_test.cpp)
target_include_directories(StringUtils_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME StringUtils_test COMMAND StringUtils_test)
