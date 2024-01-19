
include(LibConfig.cmake)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE threepp)
if (NOT DEFINED EMSCRIPTEN AND imgui_FOUND)
    target_link_libraries(demo PRIVATE imgui::imgui)
endif ()

if (DEFINED EMSCRIPTEN)
    set_target_properties(demo
            PROPERTIES SUFFIX ".html"
            LINK_FLAGS " --bind -s USE_GLFW=3 -s MIN_WEBGL_VERSION=2 -s MAX_WEBGL_VERSION=2 -s FULL_ES3 -s GL_DEBUG=1 -s WASM=1")
endif (DEFINED EMSCRIPTEN)

add_subdirectory(animation)
add_subdirectory(audio)
add_subdirectory(controls)
add_subdirectory(extras)
add_subdirectory(geometries)
add_subdirectory(helpers)
add_subdirectory(lights)
add_subdirectory(loaders)
add_subdirectory(misc)
add_subdirectory(objects)
add_subdirectory(shaders)
add_subdirectory(textures)

add_subdirectory(projects)
