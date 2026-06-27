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
        src/bind_render.cpp
        src/bind_loaders.cpp
        src/bind_robot.cpp
        src/bind_text.cpp
        src/bind_vulkan.cpp
        src/bind_imgui.cpp
        src/bind_physx.cpp)

target_link_libraries(threepp_py PRIVATE threepp)
target_compile_features(threepp_py PRIVATE cxx_std_20)

# threepp is a static lib; linking it into this shared module requires every
# object to be position-independent on ELF platforms. Without it the link fails
# with e.g. "relocation R_X86_64_TPOFF32 against `stbi__g_failure_reason` can
# not be used when making a shared object; recompile with -fPIC" (stb_image's
# thread-local in ImageLoader.cpp). Set PIC on the dependency — only reached when
# THREEPP_WITH_PYTHON is on, so the normal lib/examples/tests build is untouched.
# No-op on Windows (DLLs have no PIC concept).
set_target_properties(threepp PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Expose the Vulkan deferred renderer + G-buffer AOVs only when threepp has the
# Vulkan backend; otherwise the module is GL-only and VulkanRenderer is absent.
# THREEPP_WITH_VULKAN must match how the threepp lib was built so headers like
# ImguiContext.hpp compile their Vulkan branch in this TU too (without it, the
# Vulkan ImGui path is #ifdef'd out and falls through to GL on a Vulkan window).
if (THREEPP_WITH_VULKAN)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_VULKAN THREEPP_WITH_VULKAN)
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
    endif ()
    message(STATUS "threepp_py: PhysX rigid-body world enabled")
else ()
    message(STATUS "threepp_py: PhysX not found — physics disabled (threepp.HAS_PHYSX=False)")
endif ()

# The native module is the `threepp.threepp` submodule of the `threepp` package; the file name
# must match PYBIND11_MODULE(threepp,...) (it's the final import-path component either way).
set_target_properties(threepp_py PROPERTIES OUTPUT_NAME threepp)

if (MSVC)
    target_compile_options(threepp_py PRIVATE /bigobj)
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
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/threepp/__init__.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/threepp/urdf.py"
        DESTINATION threepp COMPONENT python_module)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/threepp/rl"
        DESTINATION threepp COMPONENT python_module
        FILES_MATCHING PATTERN "*.py")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/threepp/threepp.pyi"
        DESTINATION threepp COMPONENT python_module OPTIONAL)

# Optional: generate IDE type stubs into the package (threepp/threepp.pyi). Needs `pip
# install pybind11-stubgen`. Build with: cmake --build <dir> --target threepp_stubs
if (DEFINED Python_EXECUTABLE)
    add_custom_target(threepp_stubs
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            ${Python_EXECUTABLE} -m pybind11_stubgen threepp.threepp
            --output-dir ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS threepp_py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating threepp type stubs (requires pybind11-stubgen)")
endif ()
