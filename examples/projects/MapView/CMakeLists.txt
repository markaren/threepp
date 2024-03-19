
if (TARGET geothreepp)
    
    add_example(NAME "MapView" SOURCES "main.cpp")
    if (TARGET "MapView")
        target_link_libraries("MapView" PRIVATE geothreepp)
    endif ()

endif ()
