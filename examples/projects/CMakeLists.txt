
add_subdirectory(Crane3R)
add_subdirectory(MotorControl)
add_subdirectory(Snake)
add_subdirectory(SpheroControl)
add_subdirectory(Optimization)
add_subdirectory(Youbot)

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    add_subdirectory(Physics)
endif ()

if (THREEPP_WITH_WGPU)
    add_subdirectory(WebTide)
endif ()
