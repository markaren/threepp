
if (BULLET_FOUND)

    add_executable(bullet_demo bullet_demo.cpp)
    target_link_libraries(bullet_demo PRIVATE threepp Bullet::Bullet)

    add_executable(instanced_physics instanced_physics.cpp)
    target_link_libraries(instanced_physics PRIVATE threepp Bullet::Bullet)

    if (imgui_FOUND)
        add_executable(pid_control pid_control.cpp)
        target_link_libraries(pid_control PRIVATE threepp Bullet::Bullet imgui::imgui)
        target_include_directories(pid_control PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
    endif ()

endif ()
