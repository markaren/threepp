
find_package(assimp CONFIG QUIET)
find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)


if (THREEPP_FETCH_ASSIMP AND (NOT TARGET assimp::assimp))
    message("Fetching assimp for examples..")

    include(FetchContent)
    set(ASSIMP_BUILD_TESTS OFF)
    set(ASSIMP_INSTALL OFF)
    FetchContent_Declare(
            assimp
            GIT_REPOSITORY https://github.com/assimp/assimp.git
            GIT_TAG        v6.0.2
            GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(assimp)
endif ()

add_subdirectory(external)
