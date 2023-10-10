
add_executable(Crane3R main.cpp Crane3R.cpp)
target_link_libraries(Crane3R PRIVATE threepp)
target_include_directories(Crane3R PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")

if (imgui_FOUND)
    target_link_libraries(Crane3R PRIVATE imgui::imgui)
endif ()
