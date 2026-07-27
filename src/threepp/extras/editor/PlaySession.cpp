
#include "threepp/extras/editor/PlaySession.hpp"

#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/scenes/Scene.hpp"

#include <exception>

using namespace threepp;
using namespace threepp::editor;


void PlayController::addSession(std::shared_ptr<PlaySession> session) {

    if (session) sessions_.push_back(std::move(session));
}

void PlayController::clearSessions() {

    sessions_.clear();
}

bool PlayController::play(SceneDocument& document, std::string* error) {

    if (state_ != State::Stopped) {
        // Already running: Play on a paused session means resume.
        if (state_ == State::Paused) resume();
        return true;
    }

    if (!document.capture(snapshot_, error)) return false;

    elapsed_ = 0.f;
    state_ = State::Playing;

    for (std::size_t i = 0; i < sessions_.size(); ++i) {
        try {
            sessions_[i]->start(document.scene());
        } catch (const std::exception& e) {
            if (error) *error = sessions_[i]->name() + ": " + e.what();
            // Unwind the ones that did start, then leave the editor exactly as
            // it was — the scene has not been touched yet at this point.
            for (std::size_t j = i; j-- > 0;) {
                try {
                    sessions_[j]->stop();
                } catch (...) {
                }
            }
            state_ = State::Stopped;
            snapshot_.clear();
            return false;
        }
    }

    return true;
}

void PlayController::pause() {

    if (state_ == State::Playing) state_ = State::Paused;
}

void PlayController::resume() {

    if (state_ == State::Paused) state_ = State::Playing;
}

void PlayController::togglePause() {

    if (state_ == State::Playing) {
        state_ = State::Paused;
    } else if (state_ == State::Paused) {
        state_ = State::Playing;
    }
}

bool PlayController::stop(SceneDocument& document, std::string* error) {

    if (state_ == State::Stopped) return true;

    // Sessions first: they may hold pointers into the scene that is about to be
    // replaced.
    stopSessions();
    state_ = State::Stopped;
    elapsed_ = 0.f;

    if (!snapshot_.valid()) return true;

    const bool ok = document.restore(snapshot_, error);
    snapshot_.clear();
    return ok;
}

void PlayController::update(float dt) {

    if (state_ != State::Playing) return;

    elapsed_ += dt;
    for (const auto& session : sessions_) {
        session->update(dt);
    }
}

void PlayController::stopSessions() {

    for (const auto& session : sessions_) {
        try {
            session->stop();
        } catch (...) {
            // A runtime that fails to shut down cleanly must not strand the
            // editor in Play mode with no way back to the snapshot.
        }
    }
}
