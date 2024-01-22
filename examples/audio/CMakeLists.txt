
if (MINIAUDIO_INCLUDE_DIRS)
    add_example(NAME "audio" LINK_IMGUI)
    if (TARGET "audio")
        target_include_directories("audio" PRIVATE "${MINIAUDIO_INCLUDE_DIRS}")
    endif ()
endif ()
