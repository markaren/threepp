find_package(imgui CONFIG QUIET)
find_package(assimp CONFIG QUIET)

if (imgui_FOUND)
    set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS HAS_IMGUI)
endif ()
