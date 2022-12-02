link_libraries(threepp Catch2::Catch2)

add_executable(Vector3_test Vector3_test.cpp)
add_test(NAME Vector3_test COMMAND Vector3_test)

add_executable(Matrix4_test Matrix4_test.cpp)
add_test(NAME Matrix4_test COMMAND Matrix4_test)
