
add_example(NAME "crane3R" SOURCES main.cpp Crane3R.cpp LINK_IMGUI WEB WEB_EMBED
        "${PROJECT_SOURCE_DIR}/data/fonts@data/fonts"
        "${PROJECT_SOURCE_DIR}/data/models/obj/Crane3R@data/models/obj/Crane3R"
)
if (TARGET "crane3R")
    target_link_libraries("crane3R" PRIVATE kine)
endif ()
