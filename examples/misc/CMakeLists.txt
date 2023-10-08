
add_executable(mouse_key_listener mouse_key_listener.cpp)
target_link_libraries(mouse_key_listener PRIVATE threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PRIVATE threepp)

if (OpenCV_FOUND)
    add_executable(opencv_demo opencv_demo.cpp)
    target_link_libraries(opencv_demo PRIVATE threepp ${OpenCV_LIBRARIES})
endif ()

if (imgui_FOUND)
    add_executable(clipping clipping.cpp)
    target_link_libraries(clipping PRIVATE threepp imgui::imgui)
endif ()
