
add_library(threepp_optimization INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/DifferentialEvolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/ParticleSwarmOptimization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Problem.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Optimizer.hpp"
)
add_library(optimization ALIAS threepp_optimization)
target_include_directories(threepp_optimization INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
