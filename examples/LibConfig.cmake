
find_package(CURL CONFIG QUIET)
find_package(imgui CONFIG QUIET)
find_package(assimp CONFIG QUIET)
find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)

if (imgui_FOUND)
    set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS HAS_IMGUI)
endif ()
