
add_example(NAME "inverted_pendulum" SOURCES main.cpp TRY_LINK_IMGUI)
if (TARGET "inverted_pendulum")
    target_include_directories("inverted_pendulum" PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()
