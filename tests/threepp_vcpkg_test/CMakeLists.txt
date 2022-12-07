cmake_minimum_required(VERSION 3.15)
project(threepp_vcpkg_test)

set(CMAKE_CXX_STANDARD 17)

find_package(threepp CONFIG REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE threepp::threepp)
