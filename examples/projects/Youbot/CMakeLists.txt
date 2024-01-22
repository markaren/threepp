
add_example(NAME "youbot" LINK_ASSIMP)

add_example(NAME "youbot_kine" LINK_IMGUI LINK_ASSIMP)
if (TARGET "youbot_kine")
    target_include_directories("youbot_kine" PRIVATE "${PROJECT_SOURCE_DIR}/examples/libs")
endif ()

