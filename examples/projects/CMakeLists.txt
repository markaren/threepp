
add_subdirectory(Crane3R)
add_subdirectory(InvertedPendulum)
add_subdirectory(MotorControl)
add_subdirectory(Snake)
add_subdirectory(SpheroControl)
add_subdirectory(Optimization)
add_subdirectory(Youbot)

if (THREEPP_WITH_DAWN)
    add_subdirectory(WebTide)
endif ()
