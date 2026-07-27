
# The imgui static library is declared under examples/external. The editor is
# not an example, so pull that directory in on its own when the examples are not
# being built — nothing else about the examples tree is needed.
if (NOT TARGET imgui)
    add_subdirectory("${PROJECT_SOURCE_DIR}/examples/external" "${CMAKE_BINARY_DIR}/imgui")
endif ()

add_executable(threepp_editor
        main.cpp
        EditorApp.cpp
        EditorTheme.cpp
        FileBrowser.cpp
        ViewportMarkers.cpp
        panels/AssetPanel.cpp
        panels/HierarchyPanel.cpp
        panels/InspectorPanel.cpp
        panels/MenuBar.cpp
        panels/StatusBar.cpp
        panels/Toolbar.cpp
)

target_link_libraries(threepp_editor PRIVATE threepp imgui::imgui)
target_include_directories(threepp_editor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

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
