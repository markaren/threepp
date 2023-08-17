
add_executable(Youbot youbot.cpp)
target_link_libraries(Youbot PRIVATE threepp assimp::assimp glfw::glfw)

if (imgui_FOUND)

    add_executable(Youbot_kine youbot_kine.cpp)
    target_link_libraries(Youbot_kine PRIVATE threepp imgui::imgui assimp::assimp glfw::glfw)
    target_include_directories(Youbot_kine PRIVATE "${CMAKE_SOURCE_DIR}/examples/libs")

endif()
