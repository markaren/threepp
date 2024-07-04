
add_example(NAME "Optimization" SOURCES optimization.cpp LINK_IMGUI WEB)
if (TARGET "Optimization")
    target_link_libraries("Optimization" PRIVATE optimization)
endif ()
