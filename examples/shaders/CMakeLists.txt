
add_executable(raw_shader raw_shader.cpp)
target_link_libraries(raw_shader PRIVATE threepp)

add_executable(seascape_demo seascape_demo.cpp)
target_link_libraries(seascape_demo PRIVATE threepp)
