include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(test_event test_event.cpp)
target_link_libraries(test_event PUBLIC threepp)

add_executable(test_uniform test_uniform.cpp)
target_link_libraries(test_uniform PUBLIC threepp)

add_executable(key_listener key_listener.cpp)
target_link_libraries(key_listener PUBLIC threepp)

add_executable(mouse_listener mouse_listener.cpp)
target_link_libraries(mouse_listener PUBLIC threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PUBLIC threepp)

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




