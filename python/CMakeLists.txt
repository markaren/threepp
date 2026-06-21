# threepp — pybind11 module exposing the high-level, three.js-style scene API
# (scene graph, geometries, materials, cameras, lights, GL renderer + headless
# render-to-numpy). Enabled via -DTHREEPP_WITH_PYTHON=ON. pybind11 is fetched
# automatically; the host Python is auto-detected (override -DPython_EXECUTABLE).

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
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
        src/bind_cameras.cpp
        src/bind_lights.cpp
        src/bind_render.cpp
        src/bind_loaders.cpp
        src/bind_vulkan.cpp
        src/bind_imgui.cpp)

target_link_libraries(threepp_py PRIVATE threepp)
target_compile_features(threepp_py PRIVATE cxx_std_20)

# Expose the Vulkan deferred renderer + G-buffer AOVs only when threepp has the
# Vulkan backend; otherwise the module is GL-only and VulkanRenderer is absent.
# THREEPP_WITH_VULKAN must match how the threepp lib was built so headers like
# ImguiContext.hpp compile their Vulkan branch in this TU too (without it, the
# Vulkan ImGui path is #ifdef'd out and falls through to GL on a Vulkan window).
if (THREEPP_WITH_VULKAN)
    target_compile_definitions(threepp_py PRIVATE THREEPP_PY_HAS_VULKAN THREEPP_WITH_VULKAN)
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

# Importable as `import threepp` — the file name must match PYBIND11_MODULE(threepp,...).
set_target_properties(threepp_py PROPERTIES OUTPUT_NAME threepp)

if (MSVC)
    target_compile_options(threepp_py PRIVATE /bigobj)
endif ()

# Drop the built module next to the Python scripts so `import threepp` works
# when running from this directory. (Ninja is single-config.)
set_target_properties(threepp_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Optional: generate IDE type stubs (threepp.pyi). Needs `pip install
# pybind11-stubgen`. Build with: cmake --build <dir> --target threepp_stubs
if (DEFINED Python_EXECUTABLE)
    add_custom_target(threepp_stubs
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            ${Python_EXECUTABLE} -m pybind11_stubgen threepp
            --output-dir ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS threepp_py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating threepp type stubs (requires pybind11-stubgen)")
endif ()
