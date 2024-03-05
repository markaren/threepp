cmake_minimum_required(VERSION 3.19)
project(threepp_fetchcontent_demo)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(THREEPP_BUILD_TESTS OFF)
set(THREEPP_BUILD_EXAMPLES OFF)
FetchContent_Declare(
        threepp
        GIT_REPOSITORY https://github.com/markaren/threepp.git
        GIT_TAG 78032e414e5ef058f7f286491c47d7c4fa9ec313
)
FetchContent_MakeAvailable(threepp)

add_executable(threepp_demo main.cpp)
target_link_libraries(threepp_demo PRIVATE threepp::threepp)
