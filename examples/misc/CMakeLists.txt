
add_executable(mouse_key_listener mouse_key_listener.cpp)
target_link_libraries(mouse_key_listener PRIVATE threepp)

add_executable(raycast raycast.cpp)
target_link_libraries(raycast PRIVATE threepp)

if (OpenCV_FOUND)
    add_executable(opencv_demo opencv_demo.cpp)
    target_link_libraries(opencv_demo PRIVATE threepp ${OpenCV_LIBRARIES})
    target_include_directories(opencv_demo PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()

if (imgui_FOUND)
    add_executable(clipping clipping.cpp)
    target_link_libraries(clipping PRIVATE threepp imgui::imgui)

    add_executable(morphtargets morphtargets.cpp)
    target_link_libraries(morphtargets PRIVATE threepp imgui::imgui)
endif ()

if (assimp_FOUND)
    add_executable(morphtargets_sphere morphtargets_sphere.cpp)
    target_link_libraries(morphtargets_sphere PRIVATE threepp assimp::assimp)
endif ()
