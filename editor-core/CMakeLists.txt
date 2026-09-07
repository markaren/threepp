# threepp_editor_core — the editor/player document model and play runtime,
# deliberately OUTSIDE libthreepp and NOT installed.
#
# What lives here is the shared foundation of the two apps: the scene document
# and its *Config component schema (compiled TUs), the undo stack and selection,
# and the header-only play sessions the editor and the player both drive. None
# of it is engine API — the core library has zero edges into this directory, no
# example uses it, and its consumers are exactly the editor, the player, the
# bind_editor* Python TUs and the editor tests, all in this repository.
#
# Why not in include/: `install(DIRECTORY include/ ...)` ships every header
# there, and several of these change meaning under per-app macros the library
# cannot own — the PhysX-backed play sessions compile against the SDK only when
# the CONSUMER links it, and PhysicsPlaySession.hpp branches on the V-HACD
# macro carried by threepp_convex_decomp. Installing headers whose content
# depends on defines the installed target does not export is the same disease
# THREEPP_WITH_VULKAN had before it went PUBLIC — except here it is incurable,
# because PhysX is optional per consumer. So the headers live with this target,
# the include prefix is unchanged (threepp/extras/editor/...), and linking
# threepp_editor_core is what grants access.
#
# The macro posture is exactly what it was when these TUs sat inside libthreepp:
# this target defines nothing itself; THREEPP_WITH_VULKAN / THREEPP_WITH_AUDIO
# arrive PUBLIC from threepp, and THREEPP_EDITOR_WITH_PHYSX / _VHACD remain the
# consumers' declaration of which halves they built (apps/editor, apps/player).

add_library(threepp_editor_core STATIC
        src/AcousticSurfaceConfig.cpp
        src/AnimationConfig.cpp
        src/AnimationPlaySession.cpp
        src/ArticulationConfig.cpp
        src/CharacterConfig.cpp
        src/CommandStack.cpp
        src/ConveyorConfig.cpp
        src/EditorCommands.cpp
        src/EditorSettings.cpp
        src/FlockConfig.cpp
        src/GeneratorConfig.cpp
        src/GranularConfig.cpp
        src/JointConfig.cpp
        src/MaterialTextureSlots.cpp
        src/ObjectFactory.cpp
        src/ParticleFieldConfig.cpp
        src/PhysicsConfig.cpp
        src/PlaySession.cpp
        src/RenderConfig.cpp
        src/RobotConfig.cpp
        src/SceneDocument.cpp
        src/SceneSnapshot.cpp
        src/ScriptConfig.cpp
        src/ScriptWorkspace.cpp
        src/Selection.cpp
        src/SensorConfig.cpp
        src/ShadowFit.cpp
        src/SoundConfig.cpp
        src/SplatImportConfig.cpp
        src/SplatSurfaceConfig.cpp
        src/SplineConfig.cpp
        src/TerrainConfig.cpp
        src/TerrainSculpt.cpp
        src/TextConfig.cpp
        src/TreeConfig.cpp
        src/VehicleConfig.cpp
)

# The audio play session is gated exactly as it was in src/CMakeLists.txt: no
# audio build, no session TU. The THREEPP_WITH_AUDIO define itself arrives
# PUBLIC from threepp.
if (THREEPP_WITH_AUDIO)
    target_sources(threepp_editor_core PRIVATE src/AudioPlaySession.cpp)
endif ()

# PUBLIC threepp: every header here builds on the scene graph, and threepp's
# PUBLIC cxx_std_20 / THREEPP_WITH_VULKAN / THREEPP_WITH_AUDIO ride along, so
# consumers compile these headers the same way this target does.
target_link_libraries(threepp_editor_core PUBLIC threepp)
target_include_directories(threepp_editor_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# EditorSettings.cpp serializes with the same vendored nlohmann JSON the rest
# of threepp uses — the header keeps it out of the public surface, so the dir
# is PRIVATE here exactly as it is on the threepp target.
target_include_directories(threepp_editor_core PRIVATE "${PROJECT_SOURCE_DIR}/src/external/nlohmann")
