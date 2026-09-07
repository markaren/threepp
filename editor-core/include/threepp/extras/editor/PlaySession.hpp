// Play mode: the seam between the editor and whatever runtime is driving the
// scene.
//
// A PlaySession is three calls. It gets the live Scene on start(), a delta on
// every frame, and stop() when the user is done. In between it may do ANYTHING
// to the scene — move objects, swap materials, add and delete nodes — because
// the editor took a full snapshot before starting and rebuilds the scene from it
// afterwards (see SceneSnapshot). No session has to be reversible, and none has
// to cooperate with any other.
//
// Adding a runtime is therefore: implement three methods, hand an instance to
// PlayController::addSession(). PhysicsPlaySession (PhysX) is the one that
// ships; an animation player, a behaviour-tree runner or a robot controller
// would slot in the same way.

#ifndef THREEPP_EDITOR_PLAYSESSION_HPP
#define THREEPP_EDITOR_PLAYSESSION_HPP

#include "threepp/extras/editor/SceneSnapshot.hpp"

#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Scene;

}

namespace threepp::editor {

    class SceneDocument;

    class PlaySession {

    public:
        virtual ~PlaySession() = default;

        // Build whatever the runtime needs from the scene's current state. The
        // reference stays valid until stop() — the editor does not touch the
        // graph while playing.
        virtual void start(Scene& scene) = 0;

        // `dt` is real seconds since the last update, already zero while paused
        // (update is simply not called then).
        virtual void update(float dt) = 0;

        // Release everything acquired in start(). Called before the scene is
        // restored, so the session must not hold pointers into it afterwards.
        virtual void stop() = 0;

        [[nodiscard]] virtual std::string name() const { return "PlaySession"; }
    };


    // Owns the play state machine and the snapshot that makes Stop lossless.
    class PlayController {

    public:
        enum class State {
            Stopped,
            Playing,
            Paused
        };

        void addSession(std::shared_ptr<PlaySession> session);
        void clearSessions();
        [[nodiscard]] const std::vector<std::shared_ptr<PlaySession>>& sessions() const { return sessions_; }

        [[nodiscard]] State state() const { return state_; }
        [[nodiscard]] bool stopped() const { return state_ == State::Stopped; }
        [[nodiscard]] bool paused() const { return state_ == State::Paused; }

        // Snapshot the document, then start every session. A session that throws
        // is reported through `error` and the whole play attempt is rolled back —
        // half-started physics is worse than no physics.
        bool play(SceneDocument& document, std::string* error = nullptr);

        void pause();
        void resume();
        void togglePause();

        // Stop the sessions and restore the snapshot. `document` gets a fresh
        // Scene; the caller re-resolves its selection by uuid afterwards.
        bool stop(SceneDocument& document, std::string* error = nullptr);

        // No-op unless playing. Sessions are updated in registration order.
        void update(float dt);

        // Wall-clock seconds of un-paused play since the last play().
        [[nodiscard]] float elapsed() const { return elapsed_; }

    private:
        void stopSessions();

        std::vector<std::shared_ptr<PlaySession>> sessions_;
        SceneSnapshot snapshot_;
        State state_ = State::Stopped;
        float elapsed_ = 0.f;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PLAYSESSION_HPP
