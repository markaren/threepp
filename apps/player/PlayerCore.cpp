
#include "PlayerCore.hpp"

#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/ParticleFieldPlaySession.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/CharacterPlaySession.hpp"
#include "threepp/extras/editor/ConveyorPlaySession.hpp"
#include "threepp/extras/editor/GranularPlaySession.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#endif

#ifdef THREEPP_WITH_AUDIO
#include "threepp/extras/editor/AudioPlaySession.hpp"
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
    // Right after physics, as in the editor: its start() borrows the world
    // physics just built, and the reverse stop order tears the belts down while
    // that world is still alive. Without this session an authored conveyor is
    // inert scenery here — the visuals sit still and nothing resting on a belt
    // is carried — so a document that works under Play stops working in CI.
    conveyor_ = std::make_shared<editor::ConveyorPlaySession>();
    conveyor_->setPhysics(physics_.get());
    play_.addSession(conveyor_);
#endif

    // Right after the conveyor and outside its guard, exactly as in the editor:
    // an authored particle field needs a renderer rather than a physics world,
    // so it plays in every build. On a backend that cannot draw it — which is
    // every headless run — the session says so once and still COUNTS the nodes,
    // because a document that has a field and reports none is the regression.
    particles_ = std::make_shared<editor::ParticleFieldPlaySession>();
    particles_->setLogger([this](const std::string& message) { log(message); });
    particles_->setViewpoint([this] {
        Vector3 position;
        if (viewpointHost_) viewpointHost_->getWorldPosition(position);
        return position;
    });
    play_.addSession(particles_);

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Right after the fields and back inside the PhysX guard, as in the editor:
    // grains are a PBD simulation in the world physics built, borrowed the same
    // way the belts are, and emitted between its steps. No renderer is set here
    // — a headless run has none — which resolves the authored "auto" visual to
    // the InstancedMesh, the one that does not need Vulkan.
    granular_ = std::make_shared<editor::GranularPlaySession>();
    granular_->setPhysics(physics_.get());
    granular_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(granular_);

    // Characters, borrowing the same world for the same reason. The player has
    // no teleop of its own (nor does it drive vehicles), so an authored
    // character stands its ground and plays its idle — but it is SIMULATED: it
    // has its capsule, it is animated, and a script can drive it. Without this
    // it would be a bind-pose statue, since the animation session below defers
    // to it.
    character_ = std::make_shared<editor::CharacterPlaySession>();
    character_->setPhysics(physics_.get());
    character_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(character_);
#endif

    {
        auto animations = std::make_shared<editor::AnimationPlaySession>();
        animations->setSkipCharacters(character_ != nullptr);
        play_.addSession(animations);
    }

#ifdef THREEPP_WITH_AUDIO
    // Authored sound. A headless run leaves the listener host null, so the
    // listener sits at the origin — but the session still LOADS every authored
    // file, which is the part worth having in CI: a sound whose file moved is
    // logged on the run that broke it rather than the next time somebody
    // listens. A machine with no audio device at all is handled inside the
    // session — it logs once and becomes a no-op, never fatal.
    audio_ = std::make_shared<editor::AudioPlaySession>();
    audio_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(audio_);
#endif

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
    audio_.reset();
    granular_.reset();
    particles_.reset();
    conveyor_.reset();
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
    // The particle session only ever asks the renderer WHICH BACKEND it is:
    // ParticleField draws on Vulkan and nowhere else, so a headless run (null)
    // and a GL run both take the skip-visuals path.
    if (particles_) particles_->setRenderer(renderer);
#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Same question, different answer: the granular session has an InstancedMesh
    // to fall back on, so what the backend picks is which visual, not whether.
    if (granular_) granular_->setRenderer(renderer);
#endif
}

void PlayerCore::setViewpointHost(Object3D* host) {

    viewpointHost_ = host;
}

void PlayerCore::setAudioListenerHost(Object3D* host) {

#ifdef THREEPP_WITH_AUDIO
    if (audio_) audio_->setListenerHost(host);
#else
    (void) host;
#endif
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

#ifdef THREEPP_WITH_AUDIO
    // Where a relative userData["soundFile"] resolves from. Per episode rather
    // than once, for the same reason the editor sets it per play: the anchor is
    // the document's directory, and open() can be pointed somewhere else
    // between runs of the same PlayerCore.
    if (audio_) {
        audio_->setResourcePath(document_.path().empty()
                                        ? std::filesystem::path{}
                                        : document_.path().parent_path());
    }
#endif

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
    if (particles_) current_.particleFieldCount = particles_->fieldNodeCount();
#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (physics_) current_.bodyCount = physics_->bodyCount();
    if (conveyor_) current_.conveyorCount = conveyor_->conveyorCount();
    if (granular_) current_.granularCount = granular_->granularNodeCount();
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
    // Simulated seconds, not stepped wall time. The two disagree whenever a
    // step is longer than the world's catch-up cap (maxSubSteps substeps, then
    // the leftover accumulator is DISCARDED): a --dt=0.5 step advances the sim
    // by 4/60 s, not 0.5 s. --seconds promises simulated seconds, so it is read
    // off the world's own clock — the one that stamps sensor samples — and the
    // sum of deltas is only the fallback for a build or a document with no
    // world to ask. The world is fresh per episode, so no baseline subtraction.
    bool simulated = false;
#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (physics_ && physics_->world()) {
        current_.seconds = static_cast<float>(physics_->world()->simTime());
        simulated = true;
    }
#endif
    if (!simulated) current_.seconds += dt;

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
