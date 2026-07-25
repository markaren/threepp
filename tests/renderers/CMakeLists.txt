
add_subdirectory(gl)

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
endif ()
