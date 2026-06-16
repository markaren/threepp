
add_subdirectory(Crane3R)
add_subdirectory(MotorControl)
add_subdirectory(SpheroControl)
add_subdirectory(Optimization)
add_subdirectory(Youbot)

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    add_subdirectory(Physics)
    add_subdirectory(RobotCell)
    add_subdirectory(Vehicle)
    add_subdirectory(Drive)
    add_subdirectory(Shooter)
endif ()

if (THREEPP_WITH_WGPU)
    add_subdirectory(WebTide)
endif ()

if (TARGET rltools)
    add_subdirectory(RLtools)
endif ()
