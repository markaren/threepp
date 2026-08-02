// Plays each object's authored sound during a play session.
//
// Every object carrying a SoundConfig and a file gets one miniaudio sound: a
// PositionalAudio parented to the authored node (so its world matrix, and
// therefore the spatialization, follows whatever moves it) or a plain Audio
// when the config asked for ambience. Autoplay ones start immediately.
//
// FAILURE IS NORMAL HERE, and never fatal. A machine with no audio device
// cannot construct an AudioListener at all — ma_engine_init throws — and a
// missing or undecodable file throws per sound. Both are caught, logged, and
// skipped: a scene whose sound file moved must still play, exactly as a soft
// body that cannot cook still leaves the rigid bodies simulating.
//
// Compiled only into builds configured with -DTHREEPP_WITH_AUDIO=ON, which is
// also the macro (PUBLIC on the threepp target, so every consumer compiles the
// same one) the editor gates its own audio code on.

#ifndef THREEPP_EDITOR_AUDIOPLAYSESSION_HPP
#define THREEPP_EDITOR_AUDIOPLAYSESSION_HPP

#include "threepp/extras/editor/PlaySession.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Audio;
    class AudioListener;
    class Object3D;
    class PositionalAudio;

}// namespace threepp

namespace threepp::editor {

    class AudioPlaySession: public PlaySession {

    public:
        // Out of line: the audio types are forward-declared here, and the
        // implicit members would need them complete in every including TU.
        AudioPlaySession();
        ~AudioPlaySession() override;

        [[nodiscard]] std::string name() const override { return "Audio"; }

        // Anything worth telling the user about a sound that could not be
        // loaded, or about the device that could not be opened — the same hook
        // the physics and script sessions use.
        void setLogger(std::function<void(const std::string&)> logger);

        // Where a RELATIVE userData["soundFile"] resolves from; the editor
        // passes the open document's directory. Absolute paths ignore it.
        void setResourcePath(std::filesystem::path directory);

        // The node whose world pose the listener rides — the play camera.
        // Borrowed, and only read during update(); null simply leaves the
        // listener at the origin, which is what a headless run wants.
        void setListenerHost(Object3D* host);

        void start(Scene& scene) override;
        void update(float dt) override;
        void stop() override;

        // --- readouts ---------------------------------------------------
        // Whether the audio device came up. False after a start() on a machine
        // with no output device, in which case the session is a no-op.
        [[nodiscard]] bool listenerReady() const;
        // Sounds this session actually built (files that failed are not here).
        [[nodiscard]] std::size_t soundCount() const;
        // Whether the sound authored on `uuid` is playing right now.
        [[nodiscard]] bool isPlaying(const std::string& uuid) const;

        // The audibility bound update() enforces on every positional sound:
        // 1 inside max distance minus a short ease-out band, 0 at and past
        // max. miniaudio itself only CLAMPS its falloff at max — without this
        // an Inverse sound stays audible everywhere at its max-distance level.
        // Static and pure so the selftest can pin the curve.
        [[nodiscard]] static float distanceGate(float distance, float maxDistance);

    private:
        struct Entry {
            std::string uuid;
            // Borrowed; the editor does not touch the graph while playing, and
            // stop() runs before the scene is restored.
            Object3D* node = nullptr;
            std::unique_ptr<Audio> sound;
            // The same object as `sound` when the sound is spatialized — kept
            // as the Object3D half, which is what was addRef'd to `node`.
            PositionalAudio* spatial = nullptr;
            // Authored volume and range, re-applied by update()'s distance
            // gate — the gate multiplies the volume, so the authored value has
            // to live outside the ma_sound it keeps overwriting.
            float volume = 1;
            float maxDistance = 10000;
        };

        void log(const std::string& message) const;

        // DECLARED FIRST so it is DESTROYED LAST. An ma_sound holds a pointer
        // into the ma_engine it was initialised from, and uninitialising the
        // engine first is a use-after-free — so the listener outlives every
        // sound by construction, not by remembering to clear in order.
        std::unique_ptr<AudioListener> listener_;
        std::vector<Entry> entries_;

        Object3D* listenerHost_ = nullptr;
        std::filesystem::path resourcePath_;
        std::function<void(const std::string&)> logger_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_AUDIOPLAYSESSION_HPP
