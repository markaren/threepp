cmake_minimum_required(VERSION 3.19)
project(threepp_fetchcontent_demo)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(THREEPP_BUILD_TESTS OFF)
set(THREEPP_BUILD_EXAMPLES OFF)
FetchContent_Declare(
        threepp
        GIT_REPOSITORY https://github.com/markaren/threepp.git
        GIT_TAG c1343ad5045042c7430f1a714bd494f23082d107
)
FetchContent_MakeAvailable(threepp)

add_executable(threepp_demo main.cpp)
target_link_libraries(threepp_demo PRIVATE threepp::threepp)
