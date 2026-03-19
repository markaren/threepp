
add_example(NAME "youbot_kine" LINK_IMGUI)
if (TARGET "youbot_kine")
    target_link_libraries("youbot_kine" PRIVATE kine)
endif ()

