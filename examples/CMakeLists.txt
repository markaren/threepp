include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC threepp)

add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp)

add_executable(test_capabilites test_capabilites.cpp)
target_link_libraries(test_capabilites PUBLIC threepp)
target_link_libraries(test_capabilites
        PRIVATE
        CONAN_PKG::glfw
        CONAN_PKG::glad)

add_executable(gl_test gl_test.cpp)
target_link_libraries(gl_test
        PRIVATE
        CONAN_PKG::glfw
        CONAN_PKG::glad)


