cmake_minimum_required(VERSION 3.19)
project(threepp_fetchcontent_demo)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(THREEPP_BUILD_TESTS OFF)
set(THREEPP_BUILD_EXAMPLES OFF)
FetchContent_Declare(
        threepp
        GIT_REPOSITORY https://github.com/markaren/threepp.git
        GIT_TAG 12d6db13fe3d03d0f173621f1954016e61c9d99b
)
FetchContent_MakeAvailable(threepp)

add_executable(threepp_demo main.cpp)
target_link_libraries(threepp_demo PRIVATE threepp::threepp)
