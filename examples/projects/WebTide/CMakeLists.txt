
if (THREEPP_WITH_DAWN)
    add_example(NAME "webtide" SOURCES main.cpp)
    target_include_directories(webtide PRIVATE "${WGPU_INCLUDE_DIR}")
    target_link_libraries(webtide PRIVATE ${WGPU_LIBRARY})
endif ()
