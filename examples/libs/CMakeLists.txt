
add_library(optimization INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/DifferentialEvolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/ParticleSwarmOptimization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Problem.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/optimization/Optimizer.hpp"
)
target_include_directories(optimization INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
