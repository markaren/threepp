
add_subdirectory(gl)

# First-party BC7 mode-6 encoder, round-tripped against the renderer's own
# BCnDecode — pins the palette weights / P-bit / anchor conventions the two
# sides must agree on.
add_test_executable(BC7Encode_test)

# Vulkan deferred golden-image regression net. Plain exit-code program (not
# Catch2): renders fixed deterministic scenes and compares to committed PPM
# references via examples/vulkan/capture_util.hpp. Needs an RT-capable GPU at
# runtime → exits 42 (CTest "Skipped") where there isn't one. References are
# GPU-specific, so this is a local pre-push check, not a hard cross-hardware gate.
if (THREEPP_WITH_VULKAN AND THREEPP_WITH_GLFW)
    add_executable(VulkanGolden_test VulkanGolden_test.cpp)
    target_link_libraries(VulkanGolden_test PRIVATE threepp)
    target_include_directories(VulkanGolden_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/examples/vulkan")# capture_util.hpp
    target_compile_definitions(VulkanGolden_test PRIVATE
            DATA_FOLDER="${THREEPP_DATA_DIR}"
            PROJECT_FOLDER="${PROJECT_SOURCE_DIR}")
    add_test(NAME VulkanGolden_test COMMAND VulkanGolden_test)
    set_tests_properties(VulkanGolden_test PROPERTIES SKIP_RETURN_CODE 42)

    # Camera-model geometry: film gauge / focal length round-trip and the
    # reported cameraIntrinsics() checked against the renderer's own
    # projection. Numeric (not pixel) claims, so unlike the goldens this is
    # hardware-independent — but it still needs a device to construct the
    # renderer, hence the same skip code.
    add_executable(VulkanCameraModel_test VulkanCameraModel_test.cpp)
    target_link_libraries(VulkanCameraModel_test PRIVATE threepp)
    target_include_directories(VulkanCameraModel_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME VulkanCameraModel_test COMMAND VulkanCameraModel_test)
    set_tests_properties(VulkanCameraModel_test PROPERTIES SKIP_RETURN_CODE 42)

    # Both backends in ONE process: GLFW window hints are sticky, so a Vulkan
    # canvas used to leave GLFW_CLIENT_API=GLFW_NO_API set and any GL canvas
    # created afterwards got a context-less window (hard exit, no traceback).
    # Needs its own executable — every other test uses a single backend, and the
    # hazard is invisible unless the Vulkan canvas comes first.
    add_executable(CanvasApiOrder_test CanvasApiOrder_test.cpp)
    target_link_libraries(CanvasApiOrder_test PRIVATE threepp)
    target_include_directories(CanvasApiOrder_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME CanvasApiOrder_test COMMAND CanvasApiOrder_test)
    set_tests_properties(CanvasApiOrder_test PROPERTIES
            SKIP_RETURN_CODE 42
            SKIP_REGULAR_EXPRESSION "WGL: The driver does not appear to support OpenGL")
endif ()
