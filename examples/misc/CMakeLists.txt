
add_executable(mouse_key_listener mouse_key_listener.cpp)
target_link_libraries(mouse_key_listener PRIVATE threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PRIVATE threepp)

if (OpenCV_FOUND)
    add_executable(opencv_demo opencv_demo.cpp)
    target_link_libraries(opencv_demo PRIVATE threepp ${OpenCV_LIBRARIES})
endif ()

add_executable(test_capabilites test_capabilites.cpp)
target_include_directories(test_capabilites PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_capabilites
        PRIVATE
        threepp
        glfw::glfw
        glad::glad)
