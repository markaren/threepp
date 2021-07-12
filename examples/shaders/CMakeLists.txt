include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(raw_shader raw_shader.cpp)
target_link_libraries(raw_shader PUBLIC threepp)

add_executable(seascape_demo seascape_demo.cpp)
target_link_libraries(seascape_demo PUBLIC threepp)



