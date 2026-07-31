// threepp_player — the editor's play runtime with the editor taken off.
//
// The editor AUTHORS a document: physics bodies, articulations, sensors and
// scripts, all declarative userData that serializes with the scene (see
// doc/editor.md). This plays one. No ImGui, no panels, no undo, no gizmos, no
// selection — a window, a camera and the four play sessions, or not even the
// window.
//
// It exists to be a gate. Point it at a scene whose script drives a trained
// policy, give it a seed budget and an episode count, let the sensors write
// their CSVs, and read the exit code: zero if every episode ran clean, nonzero
// if any script raised or the document would not play. That is a thing CI can
// hold, which the editor — an interactive application whose success condition is
// "somebody looked at it" — is not.
//
// It is an EVALUATION vehicle. It runs the document as written, one instance at
// a time, at the fidelity the editor would. Batched rollouts across hundreds of
// parallel environments are GpuSim's job and are deliberately not this.

#ifndef THREEPP_PLAYER_PLAYERAPP_HPP
#define THREEPP_PLAYER_PLAYERAPP_HPP

#include "PlayerCore.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace threepp {

    class Canvas;
    class OrbitControls;
    class Renderer;

}// namespace threepp

namespace threepp::player {

    class DebugDrawOverlay;
    class SensorCloudOverlay;

    struct PlayerOptions {

        std::filesystem::path scene;

        // How many times to play the document, back to back. Each one is a full
        // play -> run -> stop cycle, and stop restores the snapshot, so they are
        // independent by construction.
        int episodes = 1;

        // Per-episode budget. Zero means unbounded; a windowed run then plays
        // until the window is closed. Both may be set, and whichever is reached
        // first ends the episode.
        int frames = 0;
        float seconds = 0.f;

        // No visible window. The canvas is still created (GLFW_VISIBLE false),
        // so there is still a GL context and the vision sensors still scan.
        bool headless = false;

        // The Vulkan backend, same flag as the editor's. A build without it
        // warns once and plays on OpenGL rather than refusing the document.
        bool vulkan = false;

        // Fixed simulation step. Zero means "use the wall clock", which is what
        // a windowed run wants; a headless run defaults this to 1/60 because a
        // bounded, reproducible run is the whole point of being headless.
        float dt = 0.f;

        // Root for the sensor CSVs. Empty means do not record.
        std::filesystem::path record;

        int width = 1280;
        int height = 720;

        // What a headless run does when given neither --frames nor --seconds.
        // Ten seconds of simulation: long enough for a controller to have done
        // something, short enough that a mistake in a CI file is not a hang.
        static constexpr float headlessDefaultSeconds = 10.f;
    };


    class PlayerApp {

    public:
        explicit PlayerApp(PlayerOptions options);
        ~PlayerApp();

        // Opens the document, plays every episode and returns the process's
        // exit code.
        int run();

    private:
        // False when the window was closed mid-episode, which ends the run.
        bool playEpisode(int index);
        void frame(float dt);

        void buildView();
        void applyDocumentRender();
        void applyDocumentView();
        void frameSceneBounds();
        void updateFollow();

        void log(const std::string& message);
        void report(const EpisodeResult& result);

        PlayerOptions options_;
        PlayerCore core_;

        std::unique_ptr<Canvas> canvas_;
        std::unique_ptr<Renderer> renderer_;
        std::unique_ptr<OrbitControls> orbit_;
        std::unique_ptr<DebugDrawOverlay> debugDraw_;
        std::unique_ptr<SensorCloudOverlay> sensorCloud_;

        PerspectiveCamera camera_{55.f, 16.f / 9.f, 0.05f, 5000.f};

        // scene.userData["editorFollow"] — the object the camera chases, by
        // name, re-resolved every episode because stop() rebuilds the scene.
        std::string followName_;
        Object3D* follow_ = nullptr;
        Vector3 followLast_;
        bool followSeeded_ = false;
    };

}// namespace threepp::player

#endif//THREEPP_PLAYER_PLAYERAPP_HPP
