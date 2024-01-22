
if (MINIAUDIO_INCLUDE_DIRS)
    add_example(NAME "audio" LINK_IMGUI)
    target_include_directories("audio" PRIVATE "${MINIAUDIO_INCLUDE_DIRS}")
endif ()
