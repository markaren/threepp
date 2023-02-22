
add_executable(lod lod.cpp)
target_link_libraries(lod PUBLIC threepp)

add_executable(reflector reflector.cpp)
target_link_libraries(reflector PUBLIC threepp)

add_executable(water water.cpp)
target_link_libraries(water PUBLIC threepp)

add_executable(sprite sprite.cpp)
target_link_libraries(sprite PUBLIC threepp)

if (assimp_FOUND)
    add_executable(decal decal.cpp)
    target_link_libraries(decal PUBLIC threepp assimp::assimp)
endif()
