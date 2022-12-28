
add_subdirectory(snake)

if (THREEPP_BUILD_EXAMPLES_WITH_KINE)
    add_subdirectory(kine)
    add_subdirectory(Crane3R)
endif()

if (assimp_FOUND)
    add_subdirectory(Youbot)
endif()
