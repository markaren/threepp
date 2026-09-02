
add_subdirectory(gl)

# First-party BC7 mode-6 encoder, round-tripped against the renderer's own
# BCnDecode — pins the palette weights / P-bit / anchor conventions the two
# sides must agree on.
add_test_executable(BC7Encode_test)

# Canvas manual placement, pre-window paths only: a requested position()
# surfaces through the getter, and setPosition() before the window exists
# records the request. The canvas is headless and never animated, so no window
# is created and no GPU or display is needed.
if (THREEPP_WITH_GLFW)
    add_test_executable(CanvasPosition_test)
    # Whether the GLFW being linked has the Null platform (3.4+). Without it a
    # display-less Linux box cannot construct a canvas at all — glfwInit fails
    # on X11 — and the test skips itself rather than reporting a defect in code
    # it never reached. Ubuntu 22.04's libglfw3-dev is 3.3.x, which is what the
    # ASan job (THREEPP_USE_EXTERNAL_GLFW=ON, no Xvfb) builds against.
    set(_null_platform 1)
    if (THREEPP_USE_EXTERNAL_GLFW AND glfw3_VERSION AND glfw3_VERSION VERSION_LESS 3.4)
        set(_null_platform 0)
    endif ()
    target_compile_definitions(CanvasPosition_test PRIVATE
            THREEPP_GLFW_HAS_NULL_PLATFORM=${_null_platform})
    unset(_null_platform)
    # Catch2 exits 4 when every one of its test cases skipped itself, which is
    # what the guard above does on such a machine. The file holds three cases
    # carrying one assertion each, so four failed assertions is not a number it
    # can otherwise return.
    set_tests_properties(CanvasPosition_test PROPERTIES SKIP_RETURN_CODE 4)
endif ()

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

    # A/B probe for the analyticDirectSplit light pick. NOT a ctest: it needs
    # two builds (old shader / new shader) to say anything, and it exists
    # because the golden scenes above have zero analytic lights and so never
    # execute that code path at all.
    add_executable(VulkanShadowPick_probe VulkanShadowPick_probe.cpp)
    target_link_libraries(VulkanShadowPick_probe PRIVATE threepp)
    target_include_directories(VulkanShadowPick_probe PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/examples/libs")# capture_util.hpp
    target_compile_definitions(VulkanShadowPick_probe PRIVATE
            DATA_FOLDER="${THREEPP_DATA_DIR}")

    # Multi-view deferred rendering: N cameras, one scene, one submission.
    # Parity against the primary, temporal-history independence, add/remove
    # lifecycle, the primary's overlay culling being immune to a secondary, and
    # per-view depth/instance-id AOVs. Same plain exit-code + skip-42 shape as
    # the goldens; the opt-in --bench / --sweep modes report timings and are not
    # part of the assertions.
    add_executable(VulkanMultiView_test VulkanMultiView_test.cpp)
    target_link_libraries(VulkanMultiView_test PRIVATE threepp)
    target_include_directories(VulkanMultiView_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/examples/vulkan")# capture_util.hpp
    add_test(NAME VulkanMultiView_test COMMAND VulkanMultiView_test)
    set_tests_properties(VulkanMultiView_test PROPERTIES SKIP_RETURN_CODE 42)

    # Measurement probe for overlay-vs-splat temporal stability. Not a CTest —
    # a harness for quantifying the baked-surface preview's shimmer.
    add_executable(SplatOverlayFlicker_probe SplatOverlayFlicker_probe.cpp)
    target_link_libraries(SplatOverlayFlicker_probe PRIVATE threepp)
    target_include_directories(SplatOverlayFlicker_probe PRIVATE "${PROJECT_SOURCE_DIR}/src")

    # Does a cloud OCCLUDE the post-resolve overlay? The flicker probe above
    # measures the exempt path's stability; this one measures the plain path's
    # correctness on the kSnapUiBlend fill route (an untextured transparent
    # MeshBasicMaterial), which is where the auto-latch has to fire for the
    # stamp to run at all. Exit-code probe, not a CTest golden.
    add_executable(SplatOverlayOcclusion_probe SplatOverlayOcclusion_probe.cpp)
    target_link_libraries(SplatOverlayOcclusion_probe PRIVATE threepp)
    target_include_directories(SplatOverlayOcclusion_probe PRIVATE "${PROJECT_SOURCE_DIR}/src")

    # Gaussian-splat compute tile rasterizer: bit-exact repeat-frame
    # determinism (the gate on the prefix-sum tile expansion — an atomic-append
    # expansion passes every visual check and fails this one), splat-behind-mesh
    # occlusion, and a golden image for a GENERATED cloud so the repo needs no
    # asset. Same plain exit-code + skip-42 shape as the goldens.
    add_executable(VulkanSplat_test VulkanSplat_test.cpp)
    target_link_libraries(VulkanSplat_test PRIVATE threepp)
    target_include_directories(VulkanSplat_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(VulkanSplat_test PRIVATE
            PROJECT_FOLDER="${PROJECT_SOURCE_DIR}")
    add_test(NAME VulkanSplat_test COMMAND VulkanSplat_test)
    set_tests_properties(VulkanSplat_test PROPERTIES SKIP_RETURN_CODE 42)

    # Splats in the mirror (plans/splat-volume-reflections.md, test-ladder items
    # 3 and 4): the two claims only a rendered frame can settle — a cloud shows
    # up in the water below it (grazing pose; face-on Fresnel is ~0.02 and hides
    # the whole feature), and a cloud with NO reflector in the scene changes
    # nothing, because the primary leg never marches the volume. Region
    # statistics against a same-scene control, not a stored reference: an Ocean
    # advances on the wall clock and is never byte-reproducible.
    #
    # REGISTERED TWICE, and the pair IS the A/B: the .off run asserts the
    # opposite of what the default run asserts, so the effect is pinned to
    # svLeg rather than to "the image changed".
    add_executable(VulkanSplatVolume_test VulkanSplatVolume_test.cpp)
    target_link_libraries(VulkanSplatVolume_test PRIVATE threepp)
    target_include_directories(VulkanSplatVolume_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/examples/vulkan")# capture_util.hpp
    target_compile_definitions(VulkanSplatVolume_test PRIVATE
            DATA_FOLDER="${THREEPP_DATA_DIR}"
            PROJECT_FOLDER="${PROJECT_SOURCE_DIR}")
    add_test(NAME VulkanSplatVolume_test COMMAND VulkanSplatVolume_test)
    set_tests_properties(VulkanSplatVolume_test PROPERTIES SKIP_RETURN_CODE 42)
    add_test(NAME VulkanSplatVolume_test.off COMMAND VulkanSplatVolume_test)
    set_tests_properties(VulkanSplatVolume_test.off PROPERTIES
            SKIP_RETURN_CODE 42
            ENVIRONMENT "THREEPP_VK_SPLATVOL_OFF=1")

    # Depth fusion of a splat scan into a triangle surface
    # (plans/splat-surface-bake.md P1): determinism of the bake, the metric
    # accuracy of a fused plane and sphere shell, the mesh's usability as a
    # cooked PhysX collider, and the floater defenses. Plain exit-code + skip-42
    # like the goldens; --sweep prints the weight-floor and guard tables that
    # placed the defaults and asserts nothing.
    #
    # The collider case needs the PhysX SDK, an optional vcpkg feature. Look for
    # it here rather than relying on tests/extras having done so — that subdir is
    # added after this one, and tests must configure standalone.
    add_executable(VulkanSplatSurface_test VulkanSplatSurface_test.cpp)
    target_link_libraries(VulkanSplatSurface_test PRIVATE threepp)
    target_include_directories(VulkanSplatSurface_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
        find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
    endif ()
    threepp_fix_physx_static_link_order()
    if (TARGET unofficial::omniverse-physx-sdk::sdk)
        target_link_libraries(VulkanSplatSurface_test PRIVATE unofficial::omniverse-physx-sdk::sdk)
        target_compile_definitions(VulkanSplatSurface_test PRIVATE THREEPP_TEST_WITH_PHYSX=1)
    endif ()
    add_test(NAME VulkanSplatSurface_test COMMAND VulkanSplatSurface_test)
    set_tests_properties(VulkanSplatSurface_test PROPERTIES SKIP_RETURN_CODE 42)

    # The cross-hardware gate. Asserts zero VALIDATION errors over a sequence of
    # reconfigurations (AOV resolves, view add/remove, render-scale and MSAA
    # realloc, swapchain recreate, entry-list churn) — a claim every conformant
    # driver must agree on, unlike the pixel goldens above, so it is the one
    # Vulkan test CI can run on a software rasteriser. Skips 42 when there is no
    # device OR no validation layer, since without the layer it proves nothing.
    add_executable(VulkanValidation_test VulkanValidation_test.cpp)
    target_link_libraries(VulkanValidation_test PRIVATE threepp)
    target_include_directories(VulkanValidation_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME VulkanValidation_test COMMAND VulkanValidation_test)
    # Generous: on lavapipe every frame is ray-traced deferred shading on the CPU,
    # and CTest's default 1500 s is not obviously enough on a loaded runner.
    set_tests_properties(VulkanValidation_test PROPERTIES
            SKIP_RETURN_CODE 42
            TIMEOUT 2400)

    # Camera-model geometry: film gauge / focal length round-trip and the
    # reported cameraIntrinsics() checked against the renderer's own
    # projection. Numeric (not pixel) claims, so unlike the goldens this is
    # hardware-independent — but it still needs a device to construct the
    # renderer, hence the same skip code.
    # GPU per-instance world matrices vs the CPU's, bit for bit. The gate on
    # stage 1 of plans/gpu-driven-instances.md: instance_expand.comp produces
    # what MeshEntry::worldMatrix holds, no consumer reads it yet, and the only
    # thing that makes the later stages safe is that the two are provably the
    # same numbers. Numeric (not pixel), so hardware-independent — but it needs
    # a device, hence the same skip code as the goldens.
    add_executable(VulkanInstanceExpand_test VulkanInstanceExpand_test.cpp)
    target_link_libraries(VulkanInstanceExpand_test PRIVATE threepp)
    target_include_directories(VulkanInstanceExpand_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME VulkanInstanceExpand_test COMMAND VulkanInstanceExpand_test)
    set_tests_properties(VulkanInstanceExpand_test PROPERTIES SKIP_RETURN_CODE 42)

    # ParticleField phase 0: the entity must be ONE entry whatever its capacity.
    # Asserts the three O(entries) CPU phases stay flat from a 10k- to a
    # 300k-particle field (read straight out of the CPU phase profiler), that a
    # field adds exactly one entry, and that its buffer plumbing provokes no
    # validation errors. Needs a device, hence the same skip code as the goldens.
    add_executable(VulkanParticleField_test VulkanParticleField_test.cpp)
    target_link_libraries(VulkanParticleField_test PRIVATE threepp)
    target_include_directories(VulkanParticleField_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME VulkanParticleField_test COMMAND VulkanParticleField_test)
    set_tests_properties(VulkanParticleField_test PROPERTIES SKIP_RETURN_CODE 42)

    # The screenshot path's sizing. writeFramebuffer used to take w/h from the
    # canvas's requested window size while the pixels came from the swapchain;
    # a window the platform clamped (Windows work-area clamp) then over-read
    # the readback buffer by the difference and crashed. Gates the odd-height
    # case and the requested-vs-actual case, both against framebufferSize().
    add_executable(VulkanFramebufferWrite_test VulkanFramebufferWrite_test.cpp)
    target_link_libraries(VulkanFramebufferWrite_test PRIVATE threepp)
    target_include_directories(VulkanFramebufferWrite_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME VulkanFramebufferWrite_test COMMAND VulkanFramebufferWrite_test)
    set_tests_properties(VulkanFramebufferWrite_test PROPERTIES SKIP_RETURN_CODE 42)

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
    # A vision sensor's near/far must bound the same quantity on both backends.
    # Same both-backends-in-one-process shape as CanvasApiOrder_test below (and
    # the same Vulkan-canvas-first ordering), because that is what it takes to
    # compare a GL scan against a Vulkan one without serialising through a file.
    add_executable(SensorBackendParity_test SensorBackendParity_test.cpp)
    target_link_libraries(SensorBackendParity_test PRIVATE threepp)
    target_include_directories(SensorBackendParity_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME SensorBackendParity_test COMMAND SensorBackendParity_test)
    set_tests_properties(SensorBackendParity_test PROPERTIES
            SKIP_RETURN_CODE 42
            SKIP_REGULAR_EXPRESSION "WGL: The driver does not appear to support OpenGL")

    add_executable(CanvasApiOrder_test CanvasApiOrder_test.cpp)
    target_link_libraries(CanvasApiOrder_test PRIVATE threepp)
    target_include_directories(CanvasApiOrder_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    add_test(NAME CanvasApiOrder_test COMMAND CanvasApiOrder_test)
    set_tests_properties(CanvasApiOrder_test PROPERTIES
            SKIP_RETURN_CODE 42
            SKIP_REGULAR_EXPRESSION "WGL: The driver does not appear to support OpenGL")
endif ()
