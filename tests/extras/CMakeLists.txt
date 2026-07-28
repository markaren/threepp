add_test_executable(EditorAnimation_test)
add_test_executable(EditorCommands_test)
add_test_executable(EditorDocument_test)
add_test_executable(EditorLights_test)
add_test_executable(EditorPlaySession_test)
add_test_executable(EditorScript_test)
add_test_executable(EditorScriptWorkspace_test)
add_test_executable(EditorSensorConfig_test)
add_test_executable(EditorSpline_test)
add_test_executable(EditorTextureSlots_test)
add_test_executable(EditorUndoRebind_test)
add_test_executable(PointCloud_test)
add_test_executable(Sensor_test)
add_test_executable(VisionSensor_test)

# Imu_test needs the PhysX SDK, which is an optional vcpkg dependency (feature
# "physx"). Look for it here rather than relying on the examples having already
# done so — tests must configure standalone.
if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
endif ()

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    foreach (name Imu_test JointEncoder_test ContactSensor_test ForceTorqueSensor_test EditorPhysicsPlay_test EditorSensorPlay_test SoftBody_test)
        add_test_executable(${name})
        target_link_libraries(${name} PRIVATE unofficial::omniverse-physx-sdk::sdk)
    endforeach ()
else ()
    message(STATUS "physx not found, skipping the PhysX-backed sensor tests..")
endif ()

# The scripting session lives in apps/editor (it needs Python, which libthreepp
# must stay free of) and is built as a library precisely so this test can drive
# the same code the editor runs. Present only when the editor is being built and
# Python's embed component was found.
if (TARGET threepp_editor_scripting)
    add_test_executable(EditorScriptPlay_test)
    target_link_libraries(EditorScriptPlay_test PRIVATE threepp_editor_scripting)

    # threepp.editor's RigidBody/SoftBody handles need both halves: the script
    # host above, and the PhysX SDK the play session is built on.
    if (TARGET unofficial::omniverse-physx-sdk::sdk)
        add_test_executable(EditorScriptPhysics_test)
        target_link_libraries(EditorScriptPhysics_test
                PRIVATE threepp_editor_scripting unofficial::omniverse-physx-sdk::sdk)
    endif ()
else ()
    message(STATUS "editor scripting not built, skipping EditorScriptPlay_test..")
endif ()
