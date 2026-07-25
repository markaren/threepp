add_test_executable(PointCloud_test)
add_test_executable(Sensor_test)

# Imu_test needs the PhysX SDK, which is an optional vcpkg dependency (feature
# "physx"). Look for it here rather than relying on the examples having already
# done so — tests must configure standalone.
if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
endif ()

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    add_test_executable(Imu_test)
    target_link_libraries(Imu_test PRIVATE unofficial::omniverse-physx-sdk::sdk)
else ()
    message(STATUS "physx not found, skipping 'Imu_test'..")
endif ()
