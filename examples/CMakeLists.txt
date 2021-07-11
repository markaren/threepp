include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(test_event test_event.cpp)
target_link_libraries(test_event PUBLIC threepp)

add_executable(test_uniform test_uniform.cpp)
target_link_libraries(test_uniform PUBLIC threepp)

add_executable(demo demo.cpp)
target_link_libraries(demo PUBLIC threepp)

add_executable(texture2d texture2d.cpp)
target_link_libraries(texture2d PUBLIC threepp)

add_executable(meshphong meshphong.cpp)
target_link_libraries(meshphong PUBLIC threepp)

add_executable(key_listener key_listener.cpp)
target_link_libraries(key_listener PUBLIC threepp)

add_executable(points_demo points_demo.cpp)
target_link_libraries(points_demo PUBLIC threepp)

add_executable(raw_shader raw_shader.cpp)
target_link_libraries(raw_shader PUBLIC threepp)

add_executable(seascape_demo seascape_demo.cpp)
target_link_libraries(seascape_demo PUBLIC threepp)


add_executable(test_capabilites test_capabilites.cpp)
target_link_libraries(test_capabilites
        PUBLIC
        threepp
        PRIVATE
        CONAN_PKG::glfw
        CONAN_PKG::glad)

add_executable(test_image_loader test_image_loader.cpp)
target_link_libraries(test_image_loader
        PUBLIC
        threepp
        PRIVATE
        CONAN_PKG::glad)


