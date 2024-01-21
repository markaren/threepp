
add_example(NAME "Crane3R" SOURCES main.cpp Crane3R.cpp TRY_LINK_IMGUI)
target_include_directories(Crane3R PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
