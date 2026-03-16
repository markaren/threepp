
add_subdirectory(gl)

if (THREEPP_WITH_DAWN)
    # Build the test executable (but don't use add_test_executable which
    # registers the test without xvfb-run)
    add_executable(CrossRenderer_test CrossRenderer_test.cpp)
    target_link_libraries(CrossRenderer_test PRIVATE threepp Catch2::Catch2WithMain)
    target_include_directories(CrossRenderer_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${WGPU_INCLUDE_DIR}"
    )
    target_link_libraries(CrossRenderer_test PRIVATE
            ${WGPU_LIBRARY}
    )
    target_compile_definitions(CrossRenderer_test PRIVATE DATA_FOLDER="${PROJECT_SOURCE_DIR}/data")

    # CrossRenderer_test needs a display server (uses GLFW for Canvas).
    # Use xvfb-run when available so it works in headless CI.
    find_program(XVFB_RUN xvfb-run)
    if (XVFB_RUN)
        add_test(NAME CrossRenderer_test COMMAND ${XVFB_RUN} -a $<TARGET_FILE:CrossRenderer_test>)
    else ()
        add_test(NAME CrossRenderer_test COMMAND CrossRenderer_test)
    endif ()
endif ()
