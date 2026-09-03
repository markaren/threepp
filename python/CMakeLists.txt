# threepp — pybind11 module exposing the high-level, three.js-style scene API
# (scene graph, geometries, materials, cameras, lights, GL renderer + headless
# render-to-numpy). Enabled via -DTHREEPP_WITH_PYTHON=ON. pybind11 is fetched
# automatically; the host Python is auto-detected (override -DPython_EXECUTABLE).

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.4   # 3.0.x is the first line that supports CPython 3.14's C-API
        GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(pybind11)

# One module split across several translation units (one PYBIND11_MODULE in
# bindings.cpp; each init_* helper lives in its own TU for parallel compile and
# readability).
pybind11_add_module(threepp_py
        src/bindings.cpp
        src/bind_math.cpp
        src/bind_textures.cpp
        src/bind_core.cpp
        src/bind_geometries.cpp
        src/bind_curves.cpp
        src/bind_bvh.cpp
        src/bind_editor.cpp
        src/bind_materials.cpp
        src/bind_objects.cpp
        src/bind_ocean.cpp
        src/bind_animation.cpp
        src/bind_cameras.cpp
        src/bind_lights.cpp
        src/bind_helpers.cpp
        src/bind_audio.cpp
        src/bind_pointcloud.cpp
        src/bind_terrain.cpp
        src/bind_vegetation.cpp
        src/bind_fauna.cpp
        src/bind_particles.cpp
        src/bind_splats.cpp
        src/bind_render.cpp
        src/bind_loaders.cpp
        src/bind_robot.cpp
        src/bind_text.cpp
        src/bind_vulkan.cpp
        src/bind_imgui.cpp
        src/bind_physx.cpp
        src/bind_sensors.cpp)

# threepp_editor_core: bind_editor.cpp binds the document model (SceneDocument,
# the *Config components), which lives outside libthreepp precisely because it
# is not installed engine API — this module is one of its four in-repo consumers.
target_link_libraries(threepp_py PRIVATE threepp threepp_editor_core)
target_compile_features(threepp_py PRIVATE cxx_std_20)

# threepp is a static lib; linking it into this shared module requires every
# object to be position-independent on ELF platforms. Without it the link fails
# with e.g. "relocation R_X86_64_TPOFF32 against `stbi__g_failure_reason` can
# not be used when making a shared object; recompile with -fPIC" (stb_image's
# thread-local in ImageLoader.cpp). Set PIC on the dependency — only reached when
# THREEPP_WITH_PYTHON is on, so the normal lib/examples/tests build is untouched.
# No-op on Windows (DLLs have no PIC concept).
set_target_properties(threepp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(threepp_editor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Expose the Vulkan deferred renderer + G-buffer AOVs only when threepp has the
# Vulkan backend; otherwise the module is GL-only and VulkanRenderer is absent.
# THREEPP_WITH_VULKAN must match how the threepp lib was built so headers like
# ImguiContext.hpp compile their Vulkan branch in this TU too (without it, the
# Vulkan ImGui path is #ifdef'd out and falls through to GL on a Vulkan window).
if (THREEPP_WITH_VULKAN)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_VULKAN THREEPP_WITH_VULKAN)
    # DELAY-load the Vulkan loader on Windows. vulkan-1.dll is otherwise a
    # load-time import of the .pyd, which would make `import threepp` fail on
    # every machine without a Vulkan runtime — GL-only users included. Delayed,
    # the DLL is first touched when a Vulkan entry point is called; the binding
    # guards that moment (bind_vulkan.cpp probes the loader in the
    # VulkanRenderer constructor and raises a clean RuntimeError instead of
    # letting the delay-load helper raise a structured exception Python cannot
    # catch). Linux needs none of this: auditwheel vendors libvulkan.so.1 into
    # the wheel, and a GPU-less machine fails at VulkanContext creation with a
    # catchable std::runtime_error.
    if (WIN32)
        target_link_options(threepp_py PRIVATE "/DELAYLOAD:vulkan-1.dll")
        target_link_libraries(threepp_py PRIVATE delayimp)
    endif ()
endif ()

# AMD FidelityFX FSR runtime DLL. threepp's FsrUpscaler loads amd_fidelityfx_vk.dll
# by resolving the path of the module it is compiled into (here, threepp_py.pyd) —
# a .pyd can't rely on the default DLL search finding a sibling — so stage the DLL
# next to the module. FFX_DLL is set by cmake/FetchFidelityFX.cmake when the threepp
# lib was configured with -DTHREEPP_WITH_FSR=ON. The setFsr/fsr_available bindings
# are unconditional (they're no-ops when the lib was built without FSR), so no extra
# define is needed here. (A Vulkan+FSR WHEEL would also need an install() rule to
# pack the DLL + its MIT LICENSE; the distributed wheel is GL-only today.)
if (THREEPP_WITH_FSR AND WIN32 AND FFX_DLL)
    add_custom_command(TARGET threepp_py POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FFX_DLL}" $<TARGET_FILE_DIR:threepp_py>)
    message(STATUS "threepp_py: FSR runtime DLL staged next to the module")
endif ()

if (THREEPP_WITH_AUDIO)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_AUDIO)
endif ()

# FBXLoader (OpenFBX) is compiled into the threepp static lib only when threepp
# was configured with -DTHREEPP_WITH_FBX=ON. threepp adds THREEPP_WITH_FBX as a
# PRIVATE define on its own target, so it doesn't reach this module — mirror it
# here as THREEPP_PY_HAS_FBX to gate the binding (#ifdef in bind_loaders.cpp).
if (THREEPP_WITH_FBX)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_FBX)
endif ()

# Dear ImGui (GL backend) for in-window UI from Python. Compile the imgui core +
# GLFW/OpenGL3 backends vendored under examples/external/imgui straight into the
# module; pairs with GLRenderer / a GL Canvas.
set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/examples/external/imgui)
if (EXISTS ${IMGUI_DIR}/imgui.cpp)
    target_sources(threepp_py PRIVATE
            ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_draw.cpp
            ${IMGUI_DIR}/imgui_tables.cpp
            ${IMGUI_DIR}/imgui_widgets.cpp
            ${IMGUI_DIR}/imgui_demo.cpp
            ${IMGUI_DIR}/imgui_impl_glfw.cpp
            ${IMGUI_DIR}/imgui_impl_opengl3.cpp)
    target_include_directories(threepp_py PRIVATE ${IMGUI_DIR})
    target_link_libraries(threepp_py PRIVATE glfw)
    if (WIN32)
        target_link_libraries(threepp_py PRIVATE opengl32)
    endif ()
    # threepp's ImguiContext pulls in the Vulkan ImGui backend when Vulkan is on,
    # so compile imgui_impl_vulkan too (needs the Vulkan headers threepp links).
    if (THREEPP_WITH_VULKAN)
        target_sources(threepp_py PRIVATE ${IMGUI_DIR}/imgui_impl_vulkan.cpp)
        find_package(Vulkan QUIET)
        if (Vulkan_FOUND)
            target_link_libraries(threepp_py PRIVATE Vulkan::Vulkan)
        endif ()
    endif ()
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_IMGUI)
    message(STATUS "threepp_py: Dear ImGui UI enabled")
endif ()

# Optional PhysX rigid-body world (header-only extras/physx/PhysxWorld.hpp), via
# the omniverse-physx-sdk (vcpkg). examples/LibConfig.cmake only find_package()s
# it for the examples, so do it here too. Configure with the vcpkg toolchain, or
# point it at an existing install:
#   -Dunofficial-omniverse-physx-sdk_DIR=<vcpkg_installed>/x64-windows/share/unofficial-omniverse-physx-sdk
find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
threepp_fix_physx_static_link_order()
if (TARGET unofficial::omniverse-physx-sdk::sdk)
    target_link_libraries(threepp_py PRIVATE unofficial::omniverse-physx-sdk::sdk)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_PHYSX)
    # Stage the SDK runtime DLLs (PhysX_64 / Common / Cooking / Foundation, plus the
    # GPU pair) next to the module so `import threepp` resolves them without the
    # vcpkg toolchain's applocal step (we use a _DIR hint, not the toolchain).
    # TARGET_RUNTIME_DLLS is empty here — the SDK is linked via static import libs —
    # so derive the bin dir from the package config location and copy the DLLs.
    if (WIN32 AND unofficial-omniverse-physx-sdk_DIR)
        get_filename_component(_physx_bin "${unofficial-omniverse-physx-sdk_DIR}/../../bin" ABSOLUTE)
        file(GLOB _physx_dlls "${_physx_bin}/PhysX*.dll")
        if (_physx_dlls)
            add_custom_command(TARGET threepp_py POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_physx_dlls} $<TARGET_FILE_DIR:threepp_py>
                    COMMAND_EXPAND_LISTS)
        endif ()

        # WHEEL: ship the CPU runtime, and only the CPU runtime. These four are
        # the module's LOAD-TIME imports (dumpbin-verified) — without them a
        # PhysX wheel dies at `import threepp`, because the POST_BUILD copy
        # above only reaches the build tree, never the wheel. Named explicitly,
        # not globbed: the glob would sweep in PhysXGpu_64.dll (238 MB raw,
        # 76 MB zipped — alone nearly the whole PyPI 100 MiB file cap) and
        # PhysXDevice64.dll. Those two are dlopen'd by bare name at runtime,
        # never linked, so the wheel imports and simulates CPU-only without
        # them; PhysxWorld's GPU dynamics (enableGpuDynamics/enableDirectGpu,
        # soft bodies, GpuSim) then need a source build or the DLLs dropped
        # next to the module by hand. A missing file here fails the install
        # loudly — if the vcpkg port renames its DLLs, better a broken build
        # than a wheel that cannot import.
        foreach (_dll PhysX_64.dll PhysXCommon_64.dll PhysXCooking_64.dll PhysXFoundation_64.dll)
            install(FILES "${_physx_bin}/${_dll}"
                    DESTINATION threepp COMPONENT python_module)
        endforeach ()
    endif ()
    # PhysX 5 is BSD-3-Clause; redistributing its binaries requires carrying the
    # notice. vcpkg stages it as share/physx/copyright — ship it beside the DLLs.
    get_filename_component(_physx_share "${unofficial-omniverse-physx-sdk_DIR}/.." ABSOLUTE)
    if (EXISTS "${_physx_share}/physx/copyright")
        install(FILES "${_physx_share}/physx/copyright"
                DESTINATION threepp COMPONENT python_module RENAME "PHYSX_LICENSE.txt")
    endif ()
    message(STATUS "threepp_py: PhysX rigid-body world enabled")
else ()
    message(STATUS "threepp_py: PhysX not found — physics disabled (threepp.HAS_PHYSX=False)")
endif ()

# The native module is the `threepp.threepp` submodule of the `threepp` package; the file name
# must match PYBIND11_MODULE(threepp,...) (it's the final import-path component either way).
set_target_properties(threepp_py PROPERTIES OUTPUT_NAME threepp)

if (MSVC)
    # /bigobj: bind_math.cpp and bind_objects.cpp exceed the default section limit.
    # /MP: this module is ~26 translation units, several of them slow, and MSBuild
    # (scikit-build-core's generator on Windows) builds a single project serially.
    # Same reasoning and the same measured win as the threepp library itself —
    # see the /MP block in src/CMakeLists.txt.
    target_compile_options(threepp_py PRIVATE /bigobj /MP)
endif ()

# Build the native module INTO the `threepp/` package dir (next to __init__.py), so
# `import threepp` loads the package and re-exports the native API. The PhysX DLLs are
# copied alongside it (POST_BUILD uses $<TARGET_FILE_DIR:threepp_py>), and the module adds
# its own dir to the DLL search path at import.
#
# Skip this under scikit-build-core (SKBUILD): wheel builds stage the module via the
# install() rules below instead of writing into the source tree.
if (NOT DEFINED SKBUILD)
    set(_threepp_py_pkg "${CMAKE_CURRENT_SOURCE_DIR}/threepp")
    set_target_properties(threepp_py PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${_threepp_py_pkg}"
            RUNTIME_OUTPUT_DIRECTORY "${_threepp_py_pkg}")
    # Multi-config generators (Visual Studio — the Windows default — plus Ninja
    # Multi-Config / Xcode) otherwise append a per-config subdir, so the module would
    # land in threepp/Debug/threepp.*.pyd where `import threepp` can't find it. Pin
    # every config to the package dir so the documented build command works under any
    # generator. No-op for single-config generators (CMAKE_CONFIGURATION_TYPES empty).
    foreach (_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_cfg}" _cfg_up)
        set_target_properties(threepp_py PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${_cfg_up} "${_threepp_py_pkg}"
                RUNTIME_OUTPUT_DIRECTORY_${_cfg_up} "${_threepp_py_pkg}")
    endforeach ()
endif ()

# Wheel packaging (scikit-build-core, pyproject.toml at the repo root): install the
# whole `threepp` package — native module + the Python files — under one component, so
# the backend packs a self-contained wheel and never copies stale dev-built binaries
# from the source tree. The C++ lib/headers (default component) are NOT installed into
# the wheel; threepp is statically linked into the module.
# A pybind11 module is a CMake MODULE library, whose artifact is the LIBRARY kind
# on every platform (incl. the Windows .pyd) — so COMPONENT must be set on the
# LIBRARY block, not just RUNTIME, or the binary lands in the default component
# and scikit-build-core's component filter drops it from the wheel.
install(TARGETS threepp_py
        LIBRARY DESTINATION threepp COMPONENT python_module
        RUNTIME DESTINATION threepp COMPONENT python_module)
# Every top-level file in python/threepp/ must be named here: one left out is
# silently ABSENT from the wheel, with no build error. That is how the CUDA
# interop shipped half-broken -- the module had enable_vertex_interop, but
# `from threepp.cuda_interop import VkInteropArray` raised ImportError for every
# wheel user, so the zero-copy path was unreachable outside a source checkout.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/threepp/__init__.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/threepp/urdf.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/threepp/utils.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/threepp/cuda_interop.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/threepp/py.typed"
        DESTINATION threepp COMPONENT python_module)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/threepp/rl"
        DESTINATION threepp COMPONENT python_module
        FILES_MATCHING PATTERN "*.py")

# Type information (PEP 561). `py.typed` is the marker that tells type checkers
# to honour this package's types once it is installed from a wheel — without it
# the stubs below are ignored outside the source tree.
#
# The stubs are a stub *package* (threepp/threepp/{__init__,imgui}.pyi) rather
# than a single flat threepp.pyi, because the native module has a submodule
# (threepp.threepp.imgui) that a flat file cannot express. The dir does not
# shadow the extension module at import: it has no __init__.py, so it is only a
# namespace-package candidate and the .pyd/.so loader wins. Regenerate with the
# threepp_stubs target below. OPTIONAL so a wheel can still be built in a tree
# where the stubs have not been generated.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/threepp/py.typed"
        DESTINATION threepp COMPONENT python_module)
# The binary-notice clause that PHYSX_LICENSE.txt satisfies for PhysX binds the
# REST of what is statically linked into this module too: libwebp and pybind11
# are BSD-3, imgui/pugixml/nlohmann/meshoptimizer are MIT, and the embedded
# GLSL derives from three.js (MIT). THIRD_PARTY.md is the repo's map of all of
# it, with the notices that must ride along reproduced in full — ship it. Not
# OPTIONAL: a license file silently missing is worse than a loud build break.
install(FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY.md"
        DESTINATION threepp COMPONENT python_module)
# editor.pyi is EXCLUDED here and replaced below. The source-tree copy is the
# UNION of the wheel's threepp.editor and the editor app's (which additionally
# compiles bind_editor_physics/_sensors/_camera/_authoring) — the editor needs that
# union for Pylance, but the wheel provides only SplinePath/spline_from_object,
# and py.typed makes checkers trust whatever ships. Shipping the union would
# certify ~40 names that raise AttributeError on a wheel install. See
# wheel-stubs/editor.pyi for the sync contract between the two files.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/threepp/threepp"
        DESTINATION threepp COMPONENT python_module OPTIONAL
        FILES_MATCHING PATTERN "*.pyi"
        PATTERN "editor.pyi" EXCLUDE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/wheel-stubs/editor.pyi"
        DESTINATION threepp/threepp COMPONENT python_module OPTIONAL)

# Optional: regenerate the IDE type stubs in the source tree
# (threepp/threepp/*.pyi). Needs the pinned generator:
#   pip install -r python/requirements-stubs.txt
#   cmake --build <dir> --target threepp_stubs
# scripts/gen_stubs.py wraps pybind11-stubgen so the result is reproducible: it
# normalises the flags, drops the one member the tool cannot express, and fails
# the build if the emitted stub does not parse (the bare CLI exits 0 either way).
if (DEFINED Python_EXECUTABLE)
    add_custom_target(threepp_stubs
            COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_stubs.py"
            --package-dir "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS threepp_py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating threepp type stubs (requires pybind11-stubgen)")
endif ()
