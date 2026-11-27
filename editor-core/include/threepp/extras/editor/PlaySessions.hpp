// The play sessions of one front end, by type.
//
// The editor and threepp_player build theirs with buildPlaySessions()
// (PlayRuntime.hpp), which decides which sessions run and in what order. This
// struct only holds the typed handles a front end keeps to read them back: the
// status readouts, the sensor overlay, the episode counts. The PlayController
// holds the same sessions and runs them.
//
// The session types are forward-declared, so a header every panel includes can
// hold one of these. A handle is null when the build has no such session (no
// PhysX, no Vulkan, no audio, no Python).

#ifndef THREEPP_EDITOR_PLAYSESSIONS_HPP
#define THREEPP_EDITOR_PLAYSESSIONS_HPP

#include "threepp/extras/editor/PlaySession.hpp"

#include <memory>

namespace threepp::editor {

    class AnimationPlaySession;
    class AudioPlaySession;
    class CharacterPlaySession;
    class ConveyorPlaySession;
    class FlockPlaySession;
    class GranularPlaySession;
    class ParticleFieldPlaySession;
    class PhysicsPlaySession;
    class ScriptPlaySession;
    class SensorPlaySession;
    class SplatSurfacePlaySession;

    struct PlaySessions {

        // In registration order. See PlayRuntime.hpp for why each one sits
        // where it does.
        std::shared_ptr<PhysicsPlaySession> physics;
        std::shared_ptr<ConveyorPlaySession> conveyor;
        std::shared_ptr<ParticleFieldPlaySession> particles;
        std::shared_ptr<GranularPlaySession> granular;
        std::shared_ptr<SplatSurfacePlaySession> splatSurfaces;
        std::shared_ptr<CharacterPlaySession> character;
        std::shared_ptr<AnimationPlaySession> animation;
        std::shared_ptr<FlockPlaySession> flock;
        std::shared_ptr<AudioPlaySession> audio;
        // A PhysxSensorPlaySession in a PhysX build.
        std::shared_ptr<SensorPlaySession> sensors;
        std::shared_ptr<ScriptPlaySession> scripts;

        // Unregisters every session from `play` and drops the handles, the last
        // registered first. A front end calls this from its destructor, before
        // the members the sessions point into (the sensor rig, the overlay, the
        // document's scene) are destroyed, after stopping any play in progress.
        void release(PlayController& play) {

            play.clearSessions();
            scripts.reset();
            sensors.reset();
            audio.reset();
            flock.reset();
            animation.reset();
            character.reset();
            splatSurfaces.reset();
            granular.reset();
            particles.reset();
            conveyor.reset();
            physics.reset();
        }
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PLAYSESSIONS_HPP
