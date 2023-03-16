
add_executable(catmull_room_curve3 curves/catmull_room_curve3.cpp)
target_link_libraries(catmull_room_curve3 PUBLIC threepp)

add_executable(cubic_bezier_curve curves/cubic_bezier_curve.cpp)
target_link_libraries(cubic_bezier_curve PUBLIC threepp)

if (nlohmann_json_FOUND)
    add_executable(fonts core/fonts.cpp)
    target_link_libraries(fonts PUBLIC threepp)
endif()
