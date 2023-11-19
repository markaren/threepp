
add_executable(hemi_light hemi_light.cpp)
target_link_libraries(hemi_light PRIVATE threepp)

add_executable(point_light point_light.cpp)
target_link_libraries(point_light PRIVATE threepp)

add_executable(spot_light spot_light.cpp)
target_link_libraries(spot_light PRIVATE threepp)

add_executable(directional directional.cpp)
target_link_libraries(directional PRIVATE threepp)
