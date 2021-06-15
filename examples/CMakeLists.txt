#include_directories(test ${CMAKE_SOURCE_DIR}/include)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC math)


