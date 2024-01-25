cmake_minimum_required(VERSION 3.17)
project(threepp_vcpkg_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(threepp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable("demo" main.cpp)
target_link_libraries("demo" PRIVATE threepp::threepp imgui::imgui)

if (NOT DEFINED EMSCRIPTEN)

    file(COPY "${CMAKE_CURRENT_LIST_DIR}/data/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/data")

else ()
    set(LINK_FLAGS " --bind -sUSE_GLFW=3 -sGL_DEBUG=1 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 \
    -sFULL_ES3 -sASSERTIONS -sALLOW_MEMORY_GROWTH -sWASM=1 \
    --embed-file data/fonts@data/fonts")

    set_target_properties("demo"
            PROPERTIES SUFFIX ".html"
            LINK_FLAGS "${LINK_FLAGS}"
    )

endif ()
