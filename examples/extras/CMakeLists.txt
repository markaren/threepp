
add_executable(catmull_room_curve3 curves/catmull_room_curve3.cpp)
target_link_libraries(catmull_room_curve3 PRIVATE threepp)

add_executable(cubic_bezier_curve curves/cubic_bezier_curve.cpp)
target_link_libraries(cubic_bezier_curve PRIVATE threepp)

add_executable(fonts core/fonts.cpp)
target_link_libraries(fonts PRIVATE threepp)
if (nlohmann_json_FOUND)
    target_link_libraries(fonts PRIVATE nlohmann_json::nlohmann_json)
endif ()
if (imgui_FOUND)
    target_link_libraries(fonts PRIVATE imgui::imgui)
endif ()
