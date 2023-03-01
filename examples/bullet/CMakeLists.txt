
add_executable(hello_bullet hello_bullet.cpp)
target_link_libraries(hello_bullet PRIVATE threepp Bullet::Bullet)

add_executable(hello_bullet2 hello_bullet2.cpp)
target_link_libraries(hello_bullet2 PRIVATE threepp Bullet::Bullet)

if (imgui_FOUND)
    add_executable(pid_control pid_control.cpp)
    target_link_libraries(pid_control PRIVATE threepp Bullet::Bullet imgui::imgui)
    if (matplotlib_FOUND)
        target_link_libraries(pid_control PRIVATE matplotlib::matplotlib)
    endif()
    target_include_directories(pid_control PRIVATE "${CMAKE_SOURCE_DIR}/examples/libs")
endif ()
