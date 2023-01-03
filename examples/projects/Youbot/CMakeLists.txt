
add_executable(Youbot youbot.cpp)
target_link_libraries(Youbot PRIVATE threepp)

if (imgui_FOUND)
    add_executable(Youbot_kine youbot_kine.cpp)
    target_link_libraries(Youbot_kine PRIVATE threepp kine imgui::imgui)
endif()
