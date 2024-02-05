cmake_minimum_required(VERSION 3.19)
project(wxthree)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(wxWidgets_USE_STATIC 1)
set(wxBUILD_SHARED OFF)
set(wxUSE_OPENGL ON)
FetchContent_Declare(
        wxWidgets
        GIT_REPOSITORY https://github.com/wxWidgets/wxWidgets.git
        GIT_TAG v3.2.4
)
FetchContent_GetProperties(wxwidgets)
if(NOT wxwidgets_POPULATED)
    FetchContent_Populate(wxwidgets)
    add_subdirectory(${wxwidgets_SOURCE_DIR} ${wxwidgets_BUILD_DIR})
endif()

set(THREEPP_BUILD_TESTS OFF)
set(THREEPP_BUILD_EXAMPLES OFF)
set(THREEPP_WITH_GLFW OFF)
FetchContent_Declare(
        threepp
        GIT_REPOSITORY https://github.com/markaren/threepp.git
        GIT_TAG 276002d3dad0b71217a2cd7246a6f450122f2b55
)
FetchContent_MakeAvailable(threepp)

add_executable(wxthree WIN32 main.cpp) # I guess this is Windows specific ?
target_include_directories(wxthree PRIVATE ${wxWidgets_SOURCE_DIR}/include)
target_link_libraries(wxthree PRIVATE threepp::threepp wx::base wx::core wx::gl)
