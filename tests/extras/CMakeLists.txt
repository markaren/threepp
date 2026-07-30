add_test_executable(EditorAnimation_test)
add_test_executable(EditorArticulationConfig_test)
add_test_executable(EditorCommands_test)
add_test_executable(EditorDocument_test)
add_test_executable(EditorLights_test)
add_test_executable(EditorPlaySession_test)
add_test_executable(EditorRobotTransplant_test)
add_test_executable(EditorScript_test)
add_test_executable(EditorScriptWorkspace_test)
add_test_executable(EditorSensorConfig_test)
add_test_executable(EditorSpline_test)
add_test_executable(EditorTextureSlots_test)
add_test_executable(EditorUndoRebind_test)
add_test_executable(EditorVisionPlay_test)
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
    foreach (name Imu_test JointEncoder_test ContactSensor_test ForceTorqueSensor_test EditorPhysicsPlay_test EditorArticulationPlay_test EditorSensorPlay_test SoftBody_test)
        add_test_executable(${name})
        target_link_libraries(${name} PRIVATE unofficial::omniverse-physx-sdk::sdk)
    endforeach ()

    # The convex-collider tests need V-HACD. Linking threepp_convex_decomp
    # carries its INTERFACE THREEPP_EDITOR_WITH_VHACD, so the header's inline
    # methods compile with the decomposition paths on — no per-target macro to
    # keep in sync (and get wrong, an ODR trap since the header is header-only).
    #
    # EditorArticulationPlay_test and EditorSensorPlay_test also include the
    # header but do NOT link this, so they compile it consistently WITHOUT the
    # macro (a valid, single-definition-per-binary state); they exercise no
    # decomposition path.
    if (TARGET threepp_convex_decomp)
        add_test_executable(EditorConvexDecomp_test)
        target_link_libraries(EditorConvexDecomp_test
                PRIVATE unofficial::omniverse-physx-sdk::sdk threepp_convex_decomp)

        # EditorPhysicsPlay_test's own cases are Box/Auto-on-mesh (no decompose),
        # but it must link V-HACD to compile the header the same way as any code
        # it shares a binary with, and to keep the door open for a decomposition
        # case here later.
        target_link_libraries(EditorPhysicsPlay_test PRIVATE threepp_convex_decomp)
    endif ()
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

    # script_from_object is deliberately NOT PhysX-gated — scripts exist in every
    # Python build, and so does the one verb that lets them find each other — so
    # its test sits here beside EditorScriptPlay_test rather than in the SDK
    # block below.
    add_test_executable(EditorScriptLookup_test)
    target_link_libraries(EditorScriptLookup_test PRIVATE threepp_editor_scripting)

    # threepp.editor's RigidBody/SoftBody handles need both halves: the script
    # host above, and the PhysX SDK the play session is built on. The sensor
    # handles are the same story one session over, and fixed_update is a script
    # session driven BY the physics world's substep loop — no SDK, no clock. The
    # collision callbacks are driven by that world's contact reports, likewise,
    # and raycast is a query put to that same world.
    if (TARGET unofficial::omniverse-physx-sdk::sdk)
        foreach (name EditorScriptPhysics_test EditorScriptSensors_test EditorScriptFixedUpdate_test
                EditorScriptCollision_test EditorScriptRaycast_test)
            add_test_executable(${name})
            target_link_libraries(${name}
                    PRIVATE threepp_editor_scripting unofficial::omniverse-physx-sdk::sdk)
        endforeach ()
    endif ()
else ()
    message(STATUS "editor scripting not built, skipping EditorScriptPlay_test..")
endif ()
