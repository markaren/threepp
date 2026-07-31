
#include "PlayerCore.hpp"

#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#endif

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

using namespace threepp;
using namespace threepp::player;

namespace {

    // "007" — an episode's directory suffix, zero-padded so a shell listing
    // sorts the way the run happened.
    std::string episodeTag(int index) {

        std::ostringstream text;
        text << std::setw(3) << std::setfill('0') << index;
        return text.str();
    }

}// namespace


PlayerCore::PlayerCore() {

    // Two nodes the document carries but never saves. Registered as
    // editor-only, which means SceneDocument detaches them for every export AND
    // for every play snapshot, and re-attaches them to the scene each episode's
    // stop() restores — so they survive an arbitrary number of episodes without
    // ever becoming part of one.
    overlay_ = Group::create();
    overlay_->name = "__player_overlay";
    document_.addEditorOnly(*overlay_);

    // A SIBLING of the overlay, not a child of it: the overlay is hidden for the
    // duration of every sensor scan (a lidar must not range against a script's
    // debug arrow), and a sensor must not be hidden from itself. Same split, and
    // same reason, as the editor's.
    sensorRig_ = Group::create();
    sensorRig_->name = "__player_sensor_rig";
    document_.addEditorOnly(*sensorRig_);

    // --- the editor's registration order, and nothing else -------------------
    // Update order is registration order; stop order is its reverse. See the
    // header, and EditorApp's constructor, which this mirrors line for line.

#ifdef THREEPP_EDITOR_WITH_PHYSX
    physics_ = std::make_shared<editor::PhysicsPlaySession>();
    physics_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(physics_);
#endif

    play_.addSession(std::make_shared<editor::AnimationPlaySession>());

    // After physics (whose world the pushed sensors register with) and after the
    // animation player, so a scan sees the pose the frame ended on.
#ifdef THREEPP_EDITOR_WITH_PHYSX
    {
        auto sensors = std::make_shared<editor::PhysxSensorPlaySession>();
        sensors->setPhysics(physics_.get());
        sensors_ = std::move(sensors);
    }
#else
    // The base session runs the VISION sensors, which need a renderer rather
    // than a physics world. The body and joint sensors author and say which
    // build they are waiting for.
    sensors_ = std::make_shared<editor::SensorPlaySession>();
#endif
    sensors_->setRig(sensorRig_.get());
    sensors_->setHiddenDuringScan(overlay_.get());
    sensors_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(sensors_);

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Last, so a script's transform edits are the final word for the frame.
    scripts_ = std::make_shared<editor::ScriptPlaySession>();
    scripts_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(scripts_);
#endif
}

PlayerCore::~PlayerCore() {

    // An episode still running at destruction is a caller that threw or bailed.
    // Stop it properly rather than letting the sessions be destroyed mid-play.
    if (!play_.stopped()) {
        std::string error;
        play_.stop(document_, &error);
    }

    // The sensor session parents nodes into sensorRig_, which is a member of
    // this same object: member destruction order would take the rig down first
    // and leave the session's destructor unlinking from freed memory. Drop the
    // sessions here, while everything they point into is still alive. (The
    // editor's destructor does exactly this, for exactly this reason.)
    play_.clearSessions();
    scripts_.reset();
    sensors_.reset();
    physics_.reset();

    if (sensorRig_) {
        sensorRig_->clear();
        document_.removeEditorOnly(*sensorRig_);
    }
    if (overlay_) {
        overlay_->clear();
        document_.removeEditorOnly(*overlay_);
    }
}

void PlayerCore::setLogger(std::function<void(const std::string&)> logger) {

    logger_ = std::move(logger);
}

void PlayerCore::setRenderer(Renderer* renderer) {

    if (sensors_) sensors_->setRenderer(renderer);
}

void PlayerCore::setRecordDirectory(const std::filesystem::path& dir, bool perEpisodeSubdirectories) {

    recordRoot_ = dir;
    recordPerEpisode_ = perEpisodeSubdirectories;
    recording_ = !dir.empty();
}

void PlayerCore::setDebugDrawDrain(std::function<void()> drain) {

    drain_ = std::move(drain);
}

Group* PlayerCore::overlay() const {

    return overlay_.get();
}

bool PlayerCore::open(const std::filesystem::path& path, std::string* error) {

    if (!document_.open(path, error)) return false;
    for (const auto& warning : document_.warnings()) log(warning);
    return true;
}

bool PlayerCore::openJson(const std::string& json, std::string* error) {

    if (!document_.openJson(json, error)) return false;
    for (const auto& warning : document_.warnings()) log(warning);
    return true;
}

bool PlayerCore::playing() const {

    return !play_.stopped();
}

bool PlayerCore::beginEpisode(int index, std::string* error) {

    if (!play_.stopped()) {
        if (error) *error = "an episode is already playing";
        return false;
    }

    current_ = EpisodeResult{};
    current_.index = index;

    // Where this episode's CSVs go, decided BEFORE play() because the files are
    // opened lazily on the first measurement — which happens inside the first
    // step().
    //
    // The per-episode subdirectory is not decoration. SensorPlaySession names a
    // file for the sensor's label and the first 8 characters of its uuid, and
    // opens it with std::ios::trunc. Those are stable across episodes (the
    // snapshot restores the same uuids, which is the point of episodes being
    // independent), so a second episode recording into the same directory would
    // silently overwrite the first one's rows and a --episodes=100 run would
    // leave exactly one episode's data behind.
    if (sensors_ && recording_) {
        auto directory = recordRoot_;
        if (recordPerEpisode_) directory /= ("episode_" + episodeTag(index));
        sensors_->setRecordDirectory(directory);
        sensors_->setRecording(true);
    }

    std::string failure;
    if (!play_.play(document_, &failure)) {
        current_.error = failure.empty() ? "play refused the document" : failure;
        if (error) *error = current_.error;
        results_.push_back(current_);
        return false;
    }

    current_.started = true;

    // Read while the sessions are up: stop() drops the entries these count.
    if (sensors_) current_.sensorCount = sensors_->sensorCount();
#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (physics_) current_.bodyCount = physics_->bodyCount();
#endif
#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (scripts_) current_.scriptInstances = scripts_->instanceCount();
    // ScriptPlaySession::start() switched the list on; anything left in it from
    // the previous episode is not this one's.
    editor::scripting::debugDraw().clear();
#endif

    return true;
}

void PlayerCore::step(float dt) {

    if (play_.stopped()) return;

    play_.update(dt);
    ++current_.frames;
    current_.seconds += dt;

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Somebody has to empty this every frame — see setDebugDrawDrain(). A
    // drainer that renders the lines does it by consuming them; with none set,
    // the list is simply cleared, which is what a headless run wants: the
    // scripts still RUN their draw calls (a draw is not allowed to behave
    // differently just because nobody is looking), the segments just go nowhere.
    if (drain_) {
        drain_();
    } else {
        editor::scripting::debugDraw().clear();
    }
#endif
}

EpisodeResult PlayerCore::endEpisode() {

    if (play_.stopped()) return current_;

    // Before the sessions go: this counts rows on entries stop() is about to
    // destroy.
    if (sensors_) current_.sensorRows = sensors_->recordedRows();

    std::string error;
    if (!play_.stop(document_, &error)) {
        current_.error = error.empty() ? "stop failed" : error;
    }

    // AFTER stop(), and this is the only correct moment. A script's own stop()
    // can raise, so reading before it would miss the last error of the episode;
    // and the next start() clears the map, so reading later would report zero.
#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (scripts_) current_.scriptErrors = scripts_->errorCount();
    // The session switched the list off with its stop(); drop whatever the last
    // frame left in it so the next episode starts from nothing.
    editor::scripting::debugDraw().clear();
#endif

    results_.push_back(current_);
    return current_;
}

EpisodeResult PlayerCore::runEpisode(int index, int frames, float dt) {

    if (!beginEpisode(index)) return results_.back();
    for (int i = 0; i < frames; ++i) step(dt);
    return endEpisode();
}

std::size_t PlayerCore::failedEpisodes() const {

    std::size_t failed = 0;
    for (const auto& result : results_) {
        if (!result.ok()) ++failed;
    }
    return failed;
}

std::size_t PlayerCore::totalScriptErrors() const {

    std::size_t errors = 0;
    for (const auto& result : results_) errors += result.scriptErrors;
    return errors;
}

int PlayerCore::exitCode() const {

    // Nothing ran: the player was asked to validate something and did not. That
    // is a failure of the job, not a vacuous pass — a CI gate that goes green
    // because the scene never loaded is worse than no gate.
    if (results_.empty()) return 1;
    return failedEpisodes() > 0 ? 1 : 0;
}

void PlayerCore::log(const std::string& message) {

    if (logger_) logger_(message);
}
