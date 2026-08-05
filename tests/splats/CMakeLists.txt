
add_test_executable(SplatData_test)

# Neither of these needs a window: SplatCloud is fully built and sorted on the
# CPU, and SplatSH_test only reads the shader source out of it. They still need
# the GLFW frontend to LINK, because SplatCloud.cpp dynamic_casts the renderer
# to GLRenderer to ask a render target for its viewport, and that reference
# drags GLRenderer.cpp — and with it Canvas::initWindow, which does not exist
# in a THREEPP_WITH_GLFW=OFF build — into every binary that touches SplatCloud.
#
# Gated rather than fixed here: the fix is to get GLRenderer's viewport out
# from behind a concrete-type cast, which is a renderer change, not a splat
# one. Same pattern as tests/renderers/gl/CMakeLists.txt.
if (THREEPP_WITH_GLFW)
    add_test_executable(SplatCloudSort_test)
    add_test_executable(SplatSH_test)
else ()
    message(STATUS "no GLFW frontend, skipping the SplatCloud-linking splat tests..")
endif ()
