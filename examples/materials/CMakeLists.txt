include_directories(test ${CMAKE_SOURCE_DIR}/include)
include_directories(test ${CMAKE_SOURCE_DIR}/src)

add_executable(meshphong meshphong.cpp)
target_link_libraries(meshphong PUBLIC threepp)


