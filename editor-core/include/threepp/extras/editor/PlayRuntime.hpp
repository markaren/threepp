// Which play sessions run, and in what order: the one definition the editor and
// threepp_player share.
//
// The player is a second front end over the editor's play runtime. If the two
// disagree about which sessions run, a scene that works under Play stops
// working in CI. Both build their sessions here, so they cannot disagree.
//
// Header-only, and included only from .cpp files. Each session is gated on the
// macro for the half it needs: THREEPP_EDITOR_WITH_PHYSX, THREEPP_WITH_VULKAN,
// THREEPP_WITH_AUDIO and THREEPP_EDITOR_WITH_PYTHON. threepp_editor_core
// defines none of them (editor-core/CMakeLists.txt): they are each consumer's
// declaration of which halves it built. Compiled in the consumer, this sees the
// same macros as the sessions it builds, which a function compiled into the
// library would not.
//
// Registration order is update order, and PlayController stops the sessions in
// reverse:
//
//   physics         first: the sessions below that need a world borrow its world.
//   conveyor        belts in that world. Its stop comes before physics', while
//                   the world is still alive to unregister from.
//   particles       needs a renderer, not a world, so it plays in every build.
//   granular        PBD grains in the physics world, emitted between its steps.
//   splat surfaces  scan colliders cooked into the physics world (Vulkan builds
//                   only: the fusion bake and the sensor master are Vulkan).
//   character       capsule controllers in the physics world. It updates after
//                   the step, so a kinematic sweep sees the settled frame.
//   animation       defers authored characters to the session above when the
//                   build has one; without PhysX it animates them itself.
//   flock           no dependencies.
//   audio           authored sound.
//   sensors         after physics, whose world the pushed sensors register with,
//                   and after animation, so a scan sees the pose the frame ended on.
//   scripts         last, so a script's transform edits are the frame's final
//                   word. A sensor a script moves is therefore read a frame later.

#ifndef THREEPP_EDITOR_PLAYRUNTIME_HPP
#define THREEPP_EDITOR_PLAYRUNTIME_HPP

#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/FlockPlaySession.hpp"
#include "threepp/extras/editor/ParticleFieldPlaySession.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/PlaySessions.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"
#include "threepp/extras/editor/SplatSurfaceCache.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/CharacterPlaySession.hpp"
#include "threepp/extras/editor/ConveyorPlaySession.hpp"
#include "threepp/extras/editor/GranularPlaySession.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/extras/editor/SplatSurfacePlaySession.hpp"
#endif
#endif

#ifdef THREEPP_WITH_AUDIO
#include "threepp/extras/editor/AudioPlaySession.hpp"
#endif

// The script session lives beside the embedded interpreter in
// threepp_editor_scripting (apps/editor/scripting). That target declares
// THREEPP_EDITOR_WITH_PYTHON and its include directory PUBLIC, so the macro
// never arrives without the header.
#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <functional>
#include <memory>
#include <string>

namespace threepp {

    class Object3D;
    class Renderer;

}// namespace threepp

namespace threepp::editor {

    // What a front end lends the sessions. Every field may stay empty: a
    // headless player has no renderer, no camera and no listener, and the
    // sessions handle that.
    struct PlayWiring {

        // Particle fields ask it which backend this is, the granular session
        // picks its visual by it, splat surfaces bake with it, and the vision
        // sensors scan with it. setPlayRenderer() changes it later.
        Renderer* renderer = nullptr;

        // Where the viewer stands, for particle fields that follow the viewer.
        // Asked once per frame.
        std::function<Vector3()> viewpoint;

        // The splat-surface bake memo. Borrowed: the editor's inspector warms
        // the same one with "Bake now".
        SplatSurfaceCache* splatSurfaceCache = nullptr;

        // Where the vision sensors parent their nodes, and the node hidden for
        // the duration of every scan (an overlay a lidar must not range
        // against). Both editor-only.
        Object3D* sensorRig = nullptr;
        Object3D* hiddenDuringScan = nullptr;

        // The node the audio listener rides. Null leaves the listener at the
        // origin; AudioPlaySession::setListenerHost() changes it later.
        Object3D* listenerHost = nullptr;

        // Receives each session message after the session has logged it in its
        // editor.* category (EditorLog.hpp).
        std::function<void(const std::string&)> logger;
    };

    // Points every session that reads the renderer at `renderer`. Null is a
    // supported value (see PlayWiring::renderer).
    inline void setPlayRenderer(PlaySessions& sessions, Renderer* renderer) {

        if (sessions.particles) sessions.particles->setRenderer(renderer);
#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (sessions.granular) sessions.granular->setRenderer(renderer);
#ifdef THREEPP_WITH_VULKAN
        if (sessions.splatSurfaces) sessions.splatSurfaces->setRenderer(renderer);
#endif
#endif
        if (sessions.sensors) sessions.sensors->setRenderer(renderer);
    }

    // Builds every session this build has, wires it, keeps its handle in
    // `sessions` and registers it with `play`, in the order above. `document`
    // is the one the sessions will play: the flock's perch bake skips its
    // editor-only nodes. Both must outlive the sessions.
    inline void buildPlaySessions(PlayController& play, const SceneDocument& document,
                                  const PlayWiring& wiring, PlaySessions& sessions) {

        const auto& logger = wiring.logger;

#ifdef THREEPP_EDITOR_WITH_PHYSX
        sessions.physics = std::make_shared<PhysicsPlaySession>();
        if (logger) sessions.physics->setLogger(logger);
        play.addSession(sessions.physics);

        sessions.conveyor = std::make_shared<ConveyorPlaySession>();
        sessions.conveyor->setPhysics(sessions.physics.get());
        play.addSession(sessions.conveyor);
#endif

        sessions.particles = std::make_shared<ParticleFieldPlaySession>();
        if (wiring.viewpoint) sessions.particles->setViewpoint(wiring.viewpoint);
        if (logger) sessions.particles->setLogger(logger);
        play.addSession(sessions.particles);

#ifdef THREEPP_EDITOR_WITH_PHYSX
        sessions.granular = std::make_shared<GranularPlaySession>();
        sessions.granular->setPhysics(sessions.physics.get());
        if (logger) sessions.granular->setLogger(logger);
        play.addSession(sessions.granular);

#ifdef THREEPP_WITH_VULKAN
        sessions.splatSurfaces = std::make_shared<SplatSurfacePlaySession>();
        sessions.splatSurfaces->setPhysics(sessions.physics.get());
        sessions.splatSurfaces->setCache(wiring.splatSurfaceCache);
        if (logger) sessions.splatSurfaces->setLogger(logger);
        play.addSession(sessions.splatSurfaces);
#endif

        sessions.character = std::make_shared<CharacterPlaySession>();
        sessions.character->setPhysics(sessions.physics.get());
        if (logger) sessions.character->setLogger(logger);
        play.addSession(sessions.character);
#endif

        sessions.animation = std::make_shared<AnimationPlaySession>();
        sessions.animation->setSkipCharacters(sessions.character != nullptr);
        play.addSession(sessions.animation);

        // The perch bake walks the whole play scene, and a front end parents
        // overlay meshes into it (gizmo handles, light markers, waypoint
        // pucks). A marker hovering at altitude reads as the highest surface
        // in its column, and the flock climbs to it.
        sessions.flock = std::make_shared<FlockPlaySession>();
        sessions.flock->setMeshFilter([&document](const Mesh& mesh) { return !document.isEditorOnly(mesh); });
        play.addSession(sessions.flock);

#ifdef THREEPP_WITH_AUDIO
        sessions.audio = std::make_shared<AudioPlaySession>();
        sessions.audio->setListenerHost(wiring.listenerHost);
        if (logger) sessions.audio->setLogger(logger);
        play.addSession(sessions.audio);
#endif

#ifdef THREEPP_EDITOR_WITH_PHYSX
        {
            auto sensors = std::make_shared<PhysxSensorPlaySession>();
            sensors->setPhysics(sessions.physics.get());
            sessions.sensors = std::move(sensors);
        }
#else
        // The base session runs the vision sensors, which need a renderer
        // rather than a world. Body and joint sensors are authored, and say at
        // Play which build they are waiting for.
        sessions.sensors = std::make_shared<SensorPlaySession>();
#endif
        sessions.sensors->setRig(wiring.sensorRig);
        sessions.sensors->setHiddenDuringScan(wiring.hiddenDuringScan);
        if (logger) sessions.sensors->setLogger(logger);
        play.addSession(sessions.sensors);

#ifdef THREEPP_EDITOR_WITH_PYTHON
        sessions.scripts = std::make_shared<ScriptPlaySession>();
        if (logger) sessions.scripts->setLogger(logger);
        play.addSession(sessions.scripts);
#endif

        setPlayRenderer(sessions, wiring.renderer);
    }

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PLAYRUNTIME_HPP
