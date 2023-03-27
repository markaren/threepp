
add_executable(stl_loader stl_loader.cpp)
target_link_libraries(stl_loader PUBLIC threepp)

add_executable(obj_loader obj_loader.cpp)
target_link_libraries(obj_loader PUBLIC threepp)

if (assimp_FOUND)
    add_executable(assimp_loader_obj assimp_loader_obj.cpp)
    target_link_libraries(assimp_loader_obj PUBLIC threepp assimp::assimp)

    add_executable(assimp_loader_glb assimp_loader_glb.cpp)
    target_link_libraries(assimp_loader_glb PUBLIC threepp assimp::assimp)
endif ()
