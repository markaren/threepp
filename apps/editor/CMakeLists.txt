
# The imgui static library is declared under examples/external. The editor is
# not an example, so pull that directory in on its own when the examples are not
# being built — nothing else about the examples tree is needed.
if (NOT TARGET imgui)
    add_subdirectory("${PROJECT_SOURCE_DIR}/examples/external" "${CMAKE_BINARY_DIR}/imgui")
endif ()

add_executable(threepp_editor
        main.cpp
        EditorApp.cpp
        EditorSelfTest.cpp
        EditorTheme.cpp
        ExternalScriptEdit.cpp
        FileBrowser.cpp
        PhysicsDebugOverlay.cpp
        SensorOverlay.cpp
        SplineOverlay.cpp
        ViewportMarkers.cpp
        panels/AssetPanel.cpp
        panels/HierarchyPanel.cpp
        panels/InspectorPanel.cpp
        panels/MenuBar.cpp
        panels/ScriptEditorPanel.cpp
        panels/SensorsPanel.cpp
        panels/StatusBar.cpp
        panels/Toolbar.cpp
)

target_link_libraries(threepp_editor PRIVATE threepp imgui::imgui)
target_include_directories(threepp_editor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Where "Edit in VS Code" points Pylance: the PEP 561 stub package that ships
# with the Python bindings (python/threepp/threepp/*.pyi). Baked in at build
# time because the editor has no other way to know where it came from; a binary
# that has moved away from its source tree is put right with the environment
# variable THREEPP_PYTHON_STUBS, which wins over this.
target_compile_definitions(threepp_editor
        PRIVATE THREEPP_EDITOR_PYTHON_STUBS="${PROJECT_SOURCE_DIR}/python/threepp")

# PhysX is an optional vcpkg dependency (feature "physx"). Found here rather
# than relying on the examples having looked for it, so the editor configures
# standalone.
if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
endif ()

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    target_link_libraries(threepp_editor PRIVATE unofficial::omniverse-physx-sdk::sdk)
    target_compile_definitions(threepp_editor PRIVATE THREEPP_EDITOR_WITH_PHYSX=1)
    message(STATUS "threepp_editor: PhysX play session enabled")

    if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-library)
        add_custom_command(TARGET threepp_editor POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
                $<TARGET_FILE_DIR:threepp_editor>)
    endif ()
    if (WIN32 AND TARGET unofficial::omniverse-physx-sdk::gpu-device-library)
        add_custom_command(TARGET threepp_editor POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
                $<TARGET_FILE_DIR:threepp_editor>)
    endif ()
else ()
    message(STATUS "threepp_editor: physx not found, Play runs without a physics session")
endif ()

# --- Python scripting -------------------------------------------------------
# MonoBehaviour-style .py scripts attached to scene objects. Gated like PhysX
# above: everything is found here so the editor configures standalone, and the
# editor still builds and runs without it (the inspector then authors
# ScriptConfig but cannot discover a class's fields).
#
# ON by default and quietly self-disabling, so nobody has to opt in to get it —
# but explicit, because embedding CPython is a deployment decision: the binary
# then needs that exact Python installation on every machine it runs on.
option(THREEPP_EDITOR_WITH_PYTHON "Build the editor's Python scripting support" ON)

if (THREEPP_EDITOR_WITH_PYTHON)
    # Development.Embed, not Development.Module: the editor EMBEDS CPython
    # rather than being imported by it, so it needs the interpreter library
    # itself, not just the headers an extension module gets.
    find_package(Python3 QUIET COMPONENTS Interpreter Development.Embed)
endif ()

if (THREEPP_EDITOR_WITH_PYTHON AND Python3_FOUND)
    # pybind11 is already fetched by python/CMakeLists.txt when
    # -DTHREEPP_WITH_PYTHON=ON; reuse it rather than fetching a second copy.
    if (NOT TARGET pybind11::headers)
        include(FetchContent)
        # Keep pybind11's own Python search on the interpreter we just found,
        # so the headers and the library we link cannot come from two installs.
        set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
        FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG v3.0.4 # 3.0.x is the first line that supports CPython 3.14's C-API
                GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif ()
endif ()

if (THREEPP_EDITOR_WITH_PYTHON AND Python3_FOUND AND TARGET pybind11::headers)

    # Its own library so the headless test can link the same code the editor
    # runs. The bind_*.cpp files are the SAME translation units the Python wheel
    # is built from — that is the whole point: one type registry, shared by the
    # module the scripts import and the objects the editor hands them.
    add_library(threepp_editor_scripting STATIC
            scripting/ScriptHost.cpp
            scripting/ScriptModule.cpp
            scripting/ScriptPlaySession.cpp
            "${PROJECT_SOURCE_DIR}/python/src/bind_math.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_textures.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_core.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_geometries.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_curves.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_editor.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_materials.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_objects.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_animation.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_cameras.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_lights.cpp"
            "${PROJECT_SOURCE_DIR}/python/src/bind_robot.cpp")

    # The physics half of threepp.editor: handles onto the bodies the play
    # session is simulating. Gated on the SDK the same way the play session
    # itself is, so a build without PhysX simply does not have the names.
    if (TARGET unofficial::omniverse-physx-sdk::sdk)
        target_sources(threepp_editor_scripting
                PRIVATE "${PROJECT_SOURCE_DIR}/python/src/bind_editor_physics.cpp")
        target_link_libraries(threepp_editor_scripting PRIVATE unofficial::omniverse-physx-sdk::sdk)
        target_compile_definitions(threepp_editor_scripting PRIVATE THREEPP_EDITOR_WITH_PHYSX=1)
    endif ()

    target_link_libraries(threepp_editor_scripting PUBLIC threepp pybind11::headers Python3::Python)
    target_include_directories(threepp_editor_scripting
            PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/scripting"
            PRIVATE "${PROJECT_SOURCE_DIR}/python/src")
    target_compile_features(threepp_editor_scripting PUBLIC cxx_std_20)
    target_compile_definitions(threepp_editor_scripting PUBLIC THREEPP_EDITOR_WITH_PYTHON=1)
    if (MSVC)
        # bind_math.cpp and bind_objects.cpp exceed the default section limit.
        target_compile_options(threepp_editor_scripting PRIVATE /bigobj)
    endif ()

    target_link_libraries(threepp_editor PRIVATE threepp_editor_scripting)
    message(STATUS "threepp_editor: Python scripting enabled (${Python3_VERSION})")
elseif (THREEPP_EDITOR_WITH_PYTHON)
    message(STATUS "threepp_editor: Python3 Development.Embed not found, scripting disabled")
else ()
    message(STATUS "threepp_editor: Python scripting disabled by option")
endif ()
