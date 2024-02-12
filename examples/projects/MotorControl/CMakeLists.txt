
add_example(NAME "MotorControl" SOURCES main.cpp LINK_IMGUI)
if (TARGET "MotorControl")
    target_include_directories("MotorControl" PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()
