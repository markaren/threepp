
add_executable(Crane3R main.cpp Crane3R.cpp)
target_link_libraries(Crane3R PRIVATE threepp kine)
target_include_directories(Crane3R PRIVATE "${CMAKE_SOURCE_DIR}/examples/utility")
if (UNIX)
    target_link_libraries(Crane3R PRIVATE pthread)
endif()

if (imgui_FOUND)
    target_link_libraries(Crane3R PRIVATE imgui::imgui)
endif()
