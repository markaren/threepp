
add_executable(catmull_room_curve3 curves/catmull_room_curve3.cpp)
target_link_libraries(catmull_room_curve3 PUBLIC threepp glfw::glfw)

add_executable(cubic_bezier_curve curves/cubic_bezier_curve.cpp)
target_link_libraries(cubic_bezier_curve PUBLIC threepp glfw::glfw)

if (nlohmann_json_FOUND)
    add_executable(fonts core/fonts.cpp)
    target_link_libraries(fonts PUBLIC threepp nlohmann_json::nlohmann_json glfw::glfw)
endif()
