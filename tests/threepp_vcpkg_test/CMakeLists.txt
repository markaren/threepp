cmake_minimum_required(VERSION 3.17)
project(threepp_vcpkg_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(threepp CONFIG REQUIRED)
#find_package(imgui CONFIG REQUIRED)

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE threepp::threepp)

if (DEFINED EMSCRIPTEN)
    set_target_properties(demo
            PROPERTIES SUFFIX ".html"
            LINK_FLAGS " --bind -s USE_GLFW=3 -s WASM=1 -g4")
endif (DEFINED EMSCRIPTEN)
