
add_executable(Crane3R main.cpp Crane3R.cpp)
target_link_libraries(Crane3R PRIVATE threepp kine)
target_include_directories(Crane3R PRIVATE "${CMAKE_SOURCE_DIR}/examples/utility")
if (imgui_FOUND)
    target_link_libraries(Crane3R PRIVATE imgui::imgui)
    target_compile_definitions(Crane3R PRIVATE "HAS_IMGUI")
endif()
