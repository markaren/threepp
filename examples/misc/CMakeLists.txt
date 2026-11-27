
add_example(NAME "lut" WEB LINK_IMGUI)
add_example(NAME "raycast" WEB)
add_example(NAME "clipping" WEB LINK_IMGUI)
add_example(NAME "morphtargets" LINK_IMGUI)
add_example(NAME "morphtargets_sphere")
add_example(NAME "multiple_scenes" WEB)
add_example(NAME "transmission" LINK_IMGUI WEB WEB_EMBED
        "${THREEPP_DATA_DIR}/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr@data/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr"
)
add_example(NAME "gltf_dump")
add_example(NAME "showpiece" WEB)
add_example(NAME "glacier_walker")

# viture_stereo renders side by side for AR/XR glasses. Head tracking comes from
# VITURE's XR Glasses SDK, which is downloaded from viture.com/developer and is
# not part of this repository: point VITURE_SDK_DIR at its root (the folder with
# include/ and x86_64/). Without it the example builds with mouse and keyboard
# head movement.
add_example(NAME "viture_stereo")
set(VITURE_SDK_DIR "" CACHE PATH "Root of VITURE's XR Glasses SDK; gives the viture_stereo example head tracking")
if (VITURE_SDK_DIR AND TARGET viture_stereo)
    find_path(VITURE_INCLUDE_DIR viture_glasses_provider.h
            PATHS "${VITURE_SDK_DIR}/include" NO_DEFAULT_PATH)
    find_library(VITURE_LIBRARY NAMES glasses
            PATHS "${VITURE_SDK_DIR}"
            PATH_SUFFIXES x86_64 aarch64 lib
            NO_DEFAULT_PATH)
    if (VITURE_INCLUDE_DIR AND VITURE_LIBRARY)
        target_include_directories(viture_stereo PRIVATE "${VITURE_INCLUDE_DIR}")
        target_link_libraries(viture_stereo PRIVATE "${VITURE_LIBRARY}")
        target_compile_definitions(viture_stereo PRIVATE THREEPP_EXAMPLE_WITH_VITURE)
        if (WIN32)
            # The example lists the connected USB devices to find the glasses.
            target_link_libraries(viture_stereo PRIVATE setupapi)
            # The SDK wants all of its DLLs in one folder beside the executable
            # (glasses.dll loads the tracking engine and libusb from there).
            get_filename_component(VITURE_LIBRARY_DIR "${VITURE_LIBRARY}" DIRECTORY)
            file(GLOB VITURE_DLLS "${VITURE_LIBRARY_DIR}/*.dll")
            foreach (dll ${VITURE_DLLS})
                add_custom_command(TARGET viture_stereo POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" $<TARGET_FILE_DIR:viture_stereo>)
            endforeach ()
        endif ()
    else ()
        message(WARNING "VITURE_SDK_DIR='${VITURE_SDK_DIR}' has no include/viture_glasses_provider.h or glasses library; viture_stereo is built without head tracking")
    endif ()
endif ()
