link_libraries(threepp Catch2::Catch2)

add_executable(Vector3_test Vector3_test.cpp)
add_test(NAME Vector3_test COMMAND Vector3_test)

add_executable(Matrix4_test Matrix4_test.cpp)
add_test(NAME Matrix4_test COMMAND Matrix4_test)

add_executable(StringUtils_test StringUtils_test.cpp)
target_include_directories(StringUtils_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
add_test(NAME StringUtils_test COMMAND StringUtils_test)
