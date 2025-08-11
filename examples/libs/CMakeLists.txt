
add_library(kine INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/Kine.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineComponent.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineLimit.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/KineLink.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/kine/ik/CCDSolver.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/ik/IKSolver.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/KineJoint.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/PrismaticJoint.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kine/joints/RevoluteJoint.hpp"
)
target_include_directories(kine INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

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
