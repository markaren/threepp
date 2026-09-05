add_test_executable(CanopyForest_test)
add_test_executable(CellStreamer_test)
add_test_executable(DataUtils_test)
add_test_executable(DownwashEffect_test)
add_test_executable(EditorAnimation_test)
add_test_executable(EditorArticulationConfig_test)
add_test_executable(EditorCharacterConfig_test)
add_test_executable(EditorCommands_test)
add_test_executable(EditorConveyorConfig_test)
add_test_executable(EditorFlockConfig_test)
add_test_executable(EditorDocument_test)
add_test_executable(EditorJointConfig_test)
add_test_executable(EditorLights_test)
add_test_executable(EditorParticleConfig_test)
add_test_executable(EditorPlaySession_test)
add_test_executable(EditorRenderConfig_test)
add_test_executable(EditorRobotTransplant_test)
add_test_executable(EditorScript_test)
add_test_executable(EditorScriptWorkspace_test)
add_test_executable(EditorSensorConfig_test)
add_test_executable(EditorShadowFit_test)
add_test_executable(EditorSpline_test)
add_test_executable(EditorTextureSlots_test)
add_test_executable(EditorUndoRebind_test)
add_test_executable(EditorVehicleConfig_test)
add_test_executable(EditorVisionPlay_test)
add_test_executable(EditorXacroArgs_test)
add_test_executable(Flock_test)
add_test_executable(GeoTerrainPack_test)
add_test_executable(InverseKinematics_test)
add_test_executable(PointCloud_test)
add_test_executable(Sensor_test)
add_test_executable(VisionSensor_test)
add_test_executable(SonarModel_test)

# extras/uav: SITL wire codec, loopback socket path, and the NED<->threepp
# frame mapping — PhysX-free. The test drives its own raw UDP sender against
# the bridge, hence ws2_32 (the bridge's own socket links inside libthreepp).
add_test_executable(SitlBridge_test)
if (WIN32)
    target_link_libraries(SitlBridge_test PRIVATE ws2_32)
endif ()

# The MAVLink v2 framer and the sector geometry that feeds it — both pinned by
# golden bytes / a synthetic caster, so no socket and no PhysX here.
add_test_executable(MavlinkOut_test)

# Imu_test needs the PhysX SDK, which is an optional vcpkg dependency (feature
# "physx"). Look for it here rather than relying on the examples having already
# done so — tests must configure standalone.
if (NOT TARGET unofficial::omniverse-physx-sdk::sdk)
    find_package(unofficial-omniverse-physx-sdk CONFIG QUIET)
endif ()
threepp_fix_physx_static_link_order()

if (TARGET unofficial::omniverse-physx-sdk::sdk)
    foreach (name Imu_test JointEncoder_test ContactSensor_test ForceTorqueSensor_test EditorPhysicsPlay_test EditorArticulationPlay_test EditorJointPlay_test EditorVehiclePlay_test EditorSensorPlay_test SoftBody_test EditorConveyor_test EditorGranular_test PhysxWorld_test PbdParticles_test)
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
    # and raycast is a query put to that same world. The trigger callbacks ride
    # that world's TRIGGER reports, and their volumes are cooked by the same
    # physics session. threepp.editor.time is the same dependency once more: its
    # simulated half is read straight off the playing world, and the divergence
    # between its two clocks only happens where there is a substep accumulator to
    # fall behind. start_coroutine's wait() rides that same simulated clock, and
    # the case that proves it is not wall-paced needs a world to starve.
    if (TARGET unofficial::omniverse-physx-sdk::sdk)
        foreach (name EditorScriptPhysics_test EditorScriptSensors_test EditorScriptFixedUpdate_test
                EditorScriptCollision_test EditorScriptRaycast_test EditorScriptTrigger_test
                EditorScriptClock_test EditorScriptCoroutine_test)
            add_test_executable(${name})
            target_link_libraries(${name}
                    PRIVATE threepp_editor_scripting unofficial::omniverse-physx-sdk::sdk)
        endforeach ()

        # The shipped example scene, loaded from the string compiled into the
        # editor and played through the real sessions. It needs all three: the
        # document (threepp_editor_examples), the script host, and the SDK whose
        # world the drone's controller pushes against.
        if (TARGET threepp_editor_examples)
            add_test_executable(EditorExampleScene_test)
            target_link_libraries(EditorExampleScene_test
                    PRIVATE threepp_editor_examples threepp_editor_scripting
                    unofficial::omniverse-physx-sdk::sdk)
        endif ()
    endif ()
else ()
    message(STATUS "editor scripting not built, skipping EditorScriptPlay_test..")
endif ()

# threepp_player's core — the play runtime with no window on it. Its own library
# for exactly this reason (apps/player/CMakeLists.txt), so the test drives the
# code the binary runs. Linking it carries the PhysX and Python gates it was
# built with, so this one test covers whichever halves the build actually has.
if (TARGET threepp_player_core)
    add_test_executable(PlayerCore_test)
    target_link_libraries(PlayerCore_test PRIVATE threepp_player_core)
else ()
    message(STATUS "threepp_player not built, skipping PlayerCore_test..")
endif ()

# Every test above that includes threepp/extras/editor/* gets it from
# threepp_editor_core — the document model and play runtime moved out of
# libthreepp (editor-core/CMakeLists.txt has the why). Guarded per target
# because half of these only exist when PhysX / scripting / the apps are on;
# for the scripting- and player-linked tests the link arrives PUBLIC already,
# and CMake dedups, but a direct include keeps a direct link here.
foreach (name
        EditorAnimation_test EditorArticulationConfig_test EditorArticulationPlay_test
        EditorCharacterConfig_test EditorCommands_test EditorConvexDecomp_test
        EditorConveyorConfig_test EditorConveyor_test EditorDocument_test
        EditorExampleScene_test EditorFlockConfig_test EditorGranular_test
        EditorJointConfig_test EditorJointPlay_test EditorParticleConfig_test
        EditorPhysicsPlay_test EditorPlaySession_test EditorRenderConfig_test
        EditorRobotTransplant_test EditorScriptClock_test EditorScriptCollision_test
        EditorScriptCoroutine_test EditorScriptFixedUpdate_test EditorScriptLookup_test
        EditorScriptPhysics_test EditorScriptPlay_test EditorScriptRaycast_test
        EditorScriptSensors_test EditorScriptTrigger_test EditorScriptWorkspace_test
        EditorScript_test EditorSensorConfig_test EditorSensorPlay_test
        EditorShadowFit_test EditorSpline_test EditorTextureSlots_test
        EditorUndoRebind_test EditorVehicleConfig_test EditorVehiclePlay_test
        EditorVisionPlay_test EditorXacroArgs_test PlayerCore_test)
    if (TARGET ${name})
        target_link_libraries(${name} PRIVATE threepp_editor_core)
    endif ()
endforeach ()
