
add_executable(Crane3R main.cpp)
target_link_libraries(Crane3R PRIVATE threepp)
if (imgui_FOUND)
    target_link_libraries(Crane3R PRIVATE imgui::imgui)
    target_compile_definitions(Crane3R PRIVATE "HAS_IMGUI")
endif()
