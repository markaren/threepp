
add_executable(cubemap_shader cubemap_shader.cpp)
target_link_libraries(cubemap_shader PUBLIC threepp)

add_executable(raw_shader raw_shader.cpp)
target_link_libraries(raw_shader PUBLIC threepp)

add_executable(seascape_demo seascape_demo.cpp)
target_link_libraries(seascape_demo PUBLIC threepp)



