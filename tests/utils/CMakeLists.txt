
add_test_executable(StringUtils_test)
target_include_directories(StringUtils_test PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(StringUtils_test PRIVATE re2::re2)
