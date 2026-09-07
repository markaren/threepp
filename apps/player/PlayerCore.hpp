// The player, minus the window.
//
// Everything threepp_player does to a document that does not involve a screen:
// open it, stand the play sessions up, step them, tear them down, and count what
// happened. The binary adds a canvas, a camera and a debug-draw overlay on top;
// a test links this and drives it headlessly, which is how the exit-code
// contract and episode independence are actually asserted rather than asserted
// about.
//
// The sessions and their ORDER are the editor's, deliberately (see EditorApp's
// constructor): physics, conveyor, animation, audio, sensors, scripts.
// Registration order is update order, and PlayController stops them in the
// reverse of it, so a script's stop() still has a world and the sensors still
// have an SDK. The player is a second front end over that runtime, not a second
// runtime — if the two ever disagree about the order, or about WHICH sessions
// run, a scene that works in the editor stops working in CI, which is the one
// thing this must never do. Every optional session is gated on the same macro
// the editor gates it on, so a build without PhysX, audio or Python drops
// exactly the same sessions in both front ends.
//
// An EPISODE is one full play -> step* -> stop cycle. Stop restores the snapshot
// PlayController took at play(), so the document an episode starts from is
// byte-for-byte the one the last episode started from: episodes are independent
// by construction, not by anybody remembering to reset something.

#ifndef THREEPP_PLAYER_PLAYERCORE_HPP
#define THREEPP_PLAYER_PLAYERCORE_HPP

#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Group;
    class Object3D;
    class Renderer;
    class Scene;

}// namespace threepp

namespace threepp::editor {

    class AudioPlaySession;
    class ConveyorPlaySession;
    class GranularPlaySession;
    class CharacterPlaySession;
    class ParticleFieldPlaySession;
    class PhysicsPlaySession;
    class ScriptPlaySession;
    class SensorPlaySession;

}// namespace threepp::editor

namespace threepp::player {

    // What one episode did. Everything a CI gate could reasonably want to see in
    // a log line, and the `ok()` that decides the process's exit code.
    struct EpisodeResult {

        int index = 0;
        // The sessions came up. False means play() refused — a physics world that
        // would not start, a script that raised at import — and `error` says so.
        bool started = false;
        std::string error;

        int frames = 0;
        // Simulated seconds, read off the physics world's own clock — the sum
        // of the fixed substeps it actually took, which is NOT the sum of the
        // deltas stepped: a step longer than the world's catch-up cap simulates
        // 4/60 s and discards the rest, however long the step claimed to be.
        // Without a world (no SDK in the build) it degrades to the sum of
        // deltas, the only elapsed time there is.
        float seconds = 0.f;

        // Scripts that raised and were disabled for the rest of the episode,
        // read after stop() — see endEpisode() for why that is the only correct
        // moment to read it.
        std::size_t scriptErrors = 0;
        std::size_t scriptInstances = 0;

        std::size_t sensorCount = 0;
        std::size_t sensorRows = 0;
        std::size_t bodyCount = 0;
        // Conveyors the belt sim picked up. Zero in a build without PhysX, and
        // zero for a document with none — but a document that HAS one and
        // reports zero is the regression this exists to make visible in a log.
        std::size_t conveyorCount = 0;
        // Authored particle-field nodes the play session saw. Counted on EVERY
        // backend, including the ones that draw no particles at all (the type is
        // Vulkan-only): what this reports is what the DOCUMENT contains, and a
        // headless run that reports zero for a scene full of snow is the same
        // regression as above.
        std::size_t particleFieldCount = 0;
        // Authored granular chutes the play session saw. Counted whether or not
        // the machine could pour them — grains are CUDA-only, and a run that
        // declined still ran a document that HAS a chute in it.
        std::size_t granularCount = 0;

        [[nodiscard]] bool ok() const {
            return started && error.empty() && scriptErrors == 0;
        }
    };


    class PlayerCore {

    public:
        PlayerCore();
        ~PlayerCore();

        PlayerCore(const PlayerCore&) = delete;
        PlayerCore& operator=(const PlayerCore&) = delete;

        // --- wiring (set before the first episode) ---------------------------

        // Console sink for what the sessions have to say. Called on the calling
        // thread, from open/beginEpisode/step/endEpisode only.
        void setLogger(std::function<void(const std::string&)> logger);

        // The renderer the VISION sensors (depth, lidar) scan with. Null is a
        // supported state, not a degraded one the caller has to guard: the
        // sensor session builds those sensors anyway, says once that it has no
        // renderer, and never scans them. The proprioceptive half (IMU,
        // encoders, contact, force/torque) needs nothing but a physics world and
        // records exactly as it would with a screen attached.
        void setRenderer(Renderer* renderer);

        // Turn CSV recording on and point it somewhere. See beginEpisode() for
        // what a multi-episode run does with the path — the short version is
        // that each episode gets its own subdirectory, because the sensor
        // session names its files after the sensor and truncates them on open.
        void setRecordDirectory(const std::filesystem::path& dir, bool perEpisodeSubdirectories);
        [[nodiscard]] bool recording() const { return recording_; }

        // The node whose world pose the audio listener rides — a windowed
        // player passes its camera. Borrowed. Null is the headless default and
        // leaves the listener at the origin, which is what a run with nobody
        // listening wants. No-op in a build without audio.
        void setAudioListenerHost(Object3D* host);

        // Where the viewpoint is, for particle fields authored to FOLLOW it: a
        // weather field wraps its spawn box toroidally about the camera so a
        // few thousand particles cover a whole scene. Same contract as the
        // listener host above — borrowed, read per frame, and null (the
        // headless default) leaves the box where the authored node put it,
        // since a run with nobody looking has no viewpoint to wrap about.
        void setViewpointHost(Object3D* host);

        // Who empties scripting::debugDraw() at the end of each step.
        //
        // The list is filled by every threepp.editor.draw_* call a script makes
        // and is capped; ScriptPlaySession switches it ON and nothing switches
        // it off, so a front end that neither draws nor drains it runs until the
        // cap and then logs a drop every frame. A windowed player passes the
        // overlay's sync here (it drains by drawing); with no drain set, step()
        // clears the list itself. Either way the list does not grow.
        void setDebugDrawDrain(std::function<void()> drain);

        // --- the document ----------------------------------------------------

        bool open(const std::filesystem::path& path, std::string* error = nullptr);
        // The same parse from text the caller already has — a document over a
        // socket, a scene compiled into a test.
        bool openJson(const std::string& json, std::string* error = nullptr);

        [[nodiscard]] editor::SceneDocument& document() { return document_; }
        [[nodiscard]] Scene& scene() const { return document_.scene(); }

        // Where a front end parents what it draws but must never save: it is
        // registered with the document as editor-only (so it survives the
        // snapshot/restore of every episode without ever entering one) and is
        // handed to the sensor session as hidden-during-scan, so a lidar cannot
        // range against somebody's debug arrow.
        [[nodiscard]] Group* overlay() const;

        // The sensor session, for a front end that VISUALIZES what it measured —
        // the point-cloud overlay reads its entries. Never null after
        // construction; its entries are empty between episodes.
        [[nodiscard]] editor::SensorPlaySession* sensors() const { return sensors_.get(); }

        // --- episodes --------------------------------------------------------

        [[nodiscard]] bool playing() const;

        bool beginEpisode(int index, std::string* error = nullptr);
        void step(float dt);
        // Simulated seconds of the episode in flight (see EpisodeResult::seconds)
        // — what a --seconds budget must be checked against, precisely because it
        // is not the sum of the deltas the caller has been stepping.
        [[nodiscard]] float episodeSeconds() const { return current_.seconds; }
        // Stops the sessions, restores the snapshot and returns the finished
        // result, which is also appended to results().
        EpisodeResult endEpisode();

        // begin + `frames` steps of `dt` + end, for a caller with no window to
        // pump. A refused start still produces (and records) a result.
        EpisodeResult runEpisode(int index, int frames, float dt);

        [[nodiscard]] const std::vector<EpisodeResult>& results() const { return results_; }
        [[nodiscard]] std::size_t failedEpisodes() const;
        [[nodiscard]] std::size_t totalScriptErrors() const;

        // The whole point of the player: 0 when every episode ran clean, 1 when
        // any of them failed to start, failed to stop or ended with a script
        // error — and 1 when nothing ran at all, which is a failure to do the
        // job rather than a vacuous success.
        [[nodiscard]] int exitCode() const;

        // One frame of a 60 Hz simulation. The default PhysxWorld timestep too,
        // so a step of exactly this is one substep.
        static constexpr float defaultDt = 1.f / 60.f;

    private:
        void log(const std::string& message);

        editor::SceneDocument document_;
        editor::PlayController play_;

        std::shared_ptr<editor::PhysicsPlaySession> physics_;
        std::shared_ptr<editor::ConveyorPlaySession> conveyor_;
        std::shared_ptr<editor::ParticleFieldPlaySession> particles_;
        std::shared_ptr<editor::GranularPlaySession> granular_;
        // Authored characters, simulated but not driven: the player has no
        // teleop, so they stand and idle unless a script moves them.
        std::shared_ptr<editor::CharacterPlaySession> character_;
        std::shared_ptr<editor::AudioPlaySession> audio_;
        std::shared_ptr<editor::SensorPlaySession> sensors_;
        std::shared_ptr<editor::ScriptPlaySession> scripts_;

        std::shared_ptr<Group> overlay_;
        std::shared_ptr<Group> sensorRig_;

        std::function<void(const std::string&)> logger_;
        std::function<void()> drain_;
        // Borrowed, and only read from the particle session's per-frame
        // callback — never dereferenced between episodes.
        Object3D* viewpointHost_ = nullptr;

        std::filesystem::path recordRoot_;
        bool recording_ = false;
        bool recordPerEpisode_ = false;

        EpisodeResult current_;
        std::vector<EpisodeResult> results_;
    };

}// namespace threepp::player

#endif//THREEPP_PLAYER_PLAYERCORE_HPP
