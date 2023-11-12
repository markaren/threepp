
if (THREEPP_BUILD_EXAMPLE_PROJECTS)

    add_subdirectory(snake)
    add_subdirectory(Crane3R)

    if (assimp_FOUND)
        add_subdirectory(Youbot)
    endif ()

else ()
    message(AUTHOR_WARNING "Building examples, but NOT example projects. Re-run with -DTHREEPP_BUILD_EXAMPLE_PROJECTS=ON to change this behaviour.")
endif ()
