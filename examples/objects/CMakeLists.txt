
add_executable(reflector reflector.cpp)
target_link_libraries(reflector PUBLIC threepp)

add_executable(water water.cpp)
target_link_libraries(water PUBLIC threepp)


