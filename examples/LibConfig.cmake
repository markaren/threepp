
find_package(assimp CONFIG QUIET)
find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
threepp_fix_physx_static_link_order()

add_subdirectory(external)
