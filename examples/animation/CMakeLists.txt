
if (assimp_FOUND)
    add_executable(simple_skinning simple_skinning.cpp)
    target_link_libraries(simple_skinning PRIVATE threepp assimp::assimp)

    add_executable(assimp_bones assimp_bones.cpp)
    target_link_libraries(assimp_bones PRIVATE threepp assimp::assimp)
endif ()
