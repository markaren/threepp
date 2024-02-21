
add_example(NAME "youbot" LINK_ASSIMP)

add_example(NAME "youbot_kine" LINK_IMGUI LINK_ASSIMP)
if (TARGET "youbot_kine")
    target_link_libraries("youbot_kine" PRIVATE kine)
endif ()

