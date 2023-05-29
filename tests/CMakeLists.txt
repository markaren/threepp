find_package(nlohmann_json CONFIG QUIET)
add_compile_definitions(DATA_FOLDER="${PROJECT_SOURCE_DIR}/data")

function(add_test_executable name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE threepp Catch2::Catch2WithMain)
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_test_executable(constants_test)

add_subdirectory(cameras)
add_subdirectory(core)
add_subdirectory(math)
add_subdirectory(utils)
add_subdirectory(renderers)
add_subdirectory(loaders)
