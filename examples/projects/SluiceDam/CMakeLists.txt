
add_example(NAME "SluiceDam" SOURCES main.cpp LINK_IMGUI WEB)
if (TARGET "SluiceDam")
    # the water's loops run on every core when the compiler has OpenMP (the same numbers without it)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries("SluiceDam" PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endif ()
