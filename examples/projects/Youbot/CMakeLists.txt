
add_executable(Youbot youbot.cpp)
target_link_libraries(Youbot PRIVATE threepp)

if (THREEPP_BUILD_EXAMPLES_WITH_KINE)
    add_executable(Youbot_kine youbot_kine.cpp)
    target_link_libraries(Youbot_kine PRIVATE threepp kine imgui::imgui)

    if (Eigen_FOUND)
        target_compile_definitions(Youbot_kine PRIVATE "HAS_EIGEN")
    endif()
endif()

