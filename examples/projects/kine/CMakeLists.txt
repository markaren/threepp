
add_library(kine INTERFACE
        "Kine.hpp"
        "KineComponent.hpp"
        "KineLink.hpp"
        "KineLimit.hpp"
        "joints/KineJoint.hpp"
        "joints/RevoluteJoint.hpp"
        "joints/PrismaticJoint.hpp"
        "ik/IKSolver.hpp"
        "ik/DLSSolver.hpp"
        "ik/CCDSolver.hpp"
        )
target_include_directories(kine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kine INTERFACE threepp Eigen3::Eigen)
