
add_library(optimization INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/DifferentialEvolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/ParticleSwarmOptimization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Problem.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Optimizer.hpp"
)
target_include_directories(optimization INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(bvh OBJECT
        bvh/BVH.hpp
        bvh/BVH.cpp
)
target_link_libraries(bvh PUBLIC threepp)
target_include_directories(bvh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(bvh PROPERTIES POSITION_INDEPENDENT_CODE ON)
