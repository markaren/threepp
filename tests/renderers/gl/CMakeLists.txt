
add_test_executable(GLRenderLists_test)
add_test_executable(GLUniforms_test)

# End-to-end GL pixel tests. These construct a Canvas, so they need GLFW and a
# display server; xvfb-run is used when available so they work in headless CI.
# On a runner with no GL driver at all, Canvas construction throws — reported as
# skipped rather than failed.
if (THREEPP_WITH_GLFW)
    function(add_gl_render_test name)
        add_executable(${name} ${name}.cpp)
        target_link_libraries(${name} PRIVATE threepp Catch2::Catch2WithMain)
        target_include_directories(${name} PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
                "${CMAKE_CURRENT_SOURCE_DIR}")
        target_compile_definitions(${name} PRIVATE DATA_FOLDER="${THREEPP_DATA_DIR}")

        find_program(XVFB_RUN xvfb-run)
        if (XVFB_RUN)
            add_test(NAME ${name} COMMAND ${XVFB_RUN} -a $<TARGET_FILE:${name}>)
        else ()
            add_test(NAME ${name} COMMAND ${name})
        endif ()

        set_tests_properties(${name} PROPERTIES
                SKIP_REGULAR_EXPRESSION "WGL: The driver does not appear to support OpenGL")
    endfunction()

    add_gl_render_test(GLRenderer_test)
    add_gl_render_test(GLFurnace_test)
    add_gl_render_test(GLShadowMap_test)
    add_gl_render_test(GLBackgroundOrtho_test)
    add_gl_render_test(GLSensorScan_test)
    add_gl_render_test(GLToneMapping_test)
endif ()
