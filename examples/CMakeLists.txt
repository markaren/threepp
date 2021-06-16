include_directories(test ${CMAKE_SOURCE_DIR}/include)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC threepp)

add_executable(gl_test gl_test.cpp)
target_link_libraries(gl_test
        PRIVATE
        CONAN_PKG::glfw
        CONAN_PKG::glad)


