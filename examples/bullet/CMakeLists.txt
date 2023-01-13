
add_library(bullet_integration BulletEngine.cpp)
target_link_libraries(bullet_integration PUBLIC threepp Bullet::Bullet)

add_executable(hello_bullet hello_bullet.cpp)
target_link_libraries(hello_bullet PRIVATE bullet_integration)

if (imgui_FOUND)
    add_executable(pid_control pid_control.cpp)
    target_link_libraries(pid_control PRIVATE bullet_integration imgui::imgui)
    target_include_directories(pid_control PRIVATE "${CMAKE_SOURCE_DIR}/examples/utility")
endif ()
