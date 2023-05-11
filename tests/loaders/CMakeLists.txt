
if (nlohmann_json_FOUND)
    add_test_executable(Fontloader_test)
    target_link_libraries(Fontloader_test PRIVATE nlohmann_json::nlohmann_json)
endif ()

if (THREEPP_WITH_SVG_FEATURE)
    add_subdirectory(svg)
endif ()
