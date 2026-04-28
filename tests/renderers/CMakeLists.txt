
add_subdirectory(gl)

if (THREEPP_WITH_WGPU AND THREEPP_WITH_GLFW)
    # Helper function to create a cross-renderer test target with Wgpu/WebGPU support.
    function(add_cross_renderer_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE threepp Catch2::Catch2WithMain)
        target_include_directories(${name} PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
                "${WGPU_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        target_link_libraries(${name} PRIVATE ${WGPU_LIBRARY})
        target_compile_definitions(${name} PRIVATE DATA_FOLDER="${PROJECT_SOURCE_DIR}/data")

        # These tests need a display server (GLFW for Canvas).
        # Use xvfb-run when available so they work in headless CI.
        find_program(XVFB_RUN xvfb-run)
        if (XVFB_RUN)
            add_test(NAME ${name} COMMAND ${XVFB_RUN} -a $<TARGET_FILE:${name}>)
        else ()
            add_test(NAME ${name} COMMAND ${name})
        endif ()

        # CI runners without a GPU/GL driver cause all tests to skip (or Canvas
        # construction to throw on Windows WGL). Mark as skipped rather than failed.
        set_tests_properties(${name} PROPERTIES
            SKIP_REGULAR_EXPRESSION "No GPU backend available for Wgpu;WGL: The driver does not appear to support OpenGL"
        )
    endfunction()

    add_cross_renderer_test(CrossRenderer_gl_test CrossRenderer_gl_test.cpp)
    add_cross_renderer_test(CrossRenderer_wgpu_test CrossRenderer_wgpu_test.cpp)
    add_cross_renderer_test(CrossRenderer_parity_test CrossRenderer_parity_test.cpp)
    add_cross_renderer_test(CrossRenderer_furnace_test CrossRenderer_furnace_test.cpp)
    add_cross_renderer_test(WgpuRenderer_perf_test WgpuRenderer_perf_test.cpp)

    if (THREEPP_WITH_INFERENCE)
        add_subdirectory(inference)
    endif ()
endif ()
