
add_example(NAME "crane3R" SOURCES main.cpp Crane3R.cpp TRY_LINK_IMGUI)
if (TARGET "crane3R")
    target_include_directories("crane3R" PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()
