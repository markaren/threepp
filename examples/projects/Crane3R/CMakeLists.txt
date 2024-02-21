
add_example(NAME "crane3R" SOURCES main.cpp Crane3R.cpp TRY_LINK_IMGUI WEB WEB_EMBED
        "../data/fonts@data/fonts"
        "../data/models/obj/Crane3R@data/models/obj/Crane3R"
)
if (TARGET "crane3R")
    target_link_libraries("crane3R" PRIVATE kine)
endif ()
