cmake_minimum_required(VERSION 3.19)
project(threepp_vcpkg_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(threepp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable("demo" main.cpp)
target_link_libraries("demo" PRIVATE threepp::threepp imgui::imgui)

if (DEFINED EMSCRIPTEN)

    set(LINK_FLAGS " --bind -sUSE_GLFW=3 -sGL_DEBUG=1 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 \
    -sFULL_ES3 -sASSERTIONS -sALLOW_MEMORY_GROWTH -sWASM=1")

    set_target_properties("demo"
            PROPERTIES SUFFIX ".html"
            LINK_FLAGS "${LINK_FLAGS}"
    )

endif ()
