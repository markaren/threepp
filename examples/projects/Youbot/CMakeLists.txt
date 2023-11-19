
add_executable(Youbot youbot.cpp)
target_link_libraries(Youbot PRIVATE threepp assimp::assimp)

if (imgui_FOUND)

    add_executable(Youbot_kine youbot_kine.cpp)
    target_link_libraries(Youbot_kine PRIVATE threepp imgui::imgui assimp::assimp)
    target_include_directories(Youbot_kine PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")

endif ()
