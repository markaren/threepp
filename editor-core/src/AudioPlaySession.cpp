
#include "threepp/extras/editor/AudioPlaySession.hpp"

#include "threepp/extras/editor/AcousticSurfaceConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"

#include "threepp/audio/Acoustics.hpp"
#include "threepp/audio/Audio.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <exception>
#include <utility>

using namespace threepp;
using namespace threepp::editor;

namespace {

    PositionalAudio::DistanceModel modelOf(SoundConfig::DistanceModel model) {

        switch (model) {
            case SoundConfig::DistanceModel::None: return PositionalAudio::DistanceModel::None;
            case SoundConfig::DistanceModel::Inverse: return PositionalAudio::DistanceModel::Inverse;
            case SoundConfig::DistanceModel::Linear: return PositionalAudio::DistanceModel::Linear;
            case SoundConfig::DistanceModel::Exponential: return PositionalAudio::DistanceModel::Exponential;
        }
        return PositionalAudio::DistanceModel::Inverse;
    }

    std::string nameOf(const Object3D& object) {

        return object.name.empty() ? "(" + object.type() + ")" : object.name;
    }

}// namespace


AudioPlaySession::AudioPlaySession() = default;

AudioPlaySession::~AudioPlaySession() = default;

void AudioPlaySession::setLogger(std::function<void(const std::string&)> logger) {

    logger_ = std::move(logger);
}

void AudioPlaySession::setResourcePath(std::filesystem::path directory) {

    resourcePath_ = std::move(directory);
}

void AudioPlaySession::setListenerHost(Object3D* host) {

    listenerHost_ = host;
}

void AudioPlaySession::log(const std::string& message) const {

    if (logger_) logger_(message);
}

void AudioPlaySession::start(Scene& scene) {

    // Collect first, build second: addRef parents a node under an object the
    // traversal is walking, and mutating the graph mid-traverse is the kind of
    // thing that works until it does not.
    struct Authored {
        Object3D* node;
        SoundConfig config;
        std::filesystem::path file;
    };
    std::vector<Authored> authored;

    // The acoustic geometry, gathered in the same walk. Only meshes: an
    // AcousticScene entry is a BVH over a BufferGeometry.
    struct Surface {
        const Mesh* mesh;
        AcousticSurface surface;
    };
    std::vector<Surface> surfaces;

    scene.traverse([&](Object3D& object) {
        if (const auto* mesh = object.as<Mesh>()) {
            const auto acoustic = AcousticSurfaceConfig::read(object);
            if (acoustic && acoustic->enabled) {
                surfaces.push_back({mesh, AcousticSurface{acoustic->transmission, acoustic->absorption}});
            }
        }

        const auto config = SoundConfig::read(object);
        if (!config) return;

        const auto stored = SoundConfig::file(object);
        if (stored.empty()) {
            log("sound on \"" + nameOf(object) + "\" has no file - nothing to play");
            return;
        }
        authored.push_back({&object, *config, SoundConfig::resolveFile(stored, resourcePath_)});
    });

    if (authored.empty()) return;

    // One engine for the whole session, opened only once there is something to
    // play — a scene with no sounds must not grab the audio device.
    if (!listener_) {
        try {
            listener_ = std::make_unique<AudioListener>();
        } catch (const std::exception& e) {
            // No device (headless, CI, exclusive-mode driver). Say so once and
            // become a no-op; every other session still runs.
            log("audio device unavailable - " + std::to_string(authored.size()) +
                " authored sound(s) will not play (" + e.what() + ")");
            return;
        }
    }

    for (auto& item : authored) {

        const auto& config = item.config;

        Entry entry;
        entry.uuid = item.node->uuid;
        entry.node = item.node;
        entry.volume = config.volume;
        entry.maxDistance = config.maxDistance;

        try {
            if (config.positional) {
                auto sound = std::make_unique<PositionalAudio>(*listener_, item.file);
                entry.spatial = sound.get();
                entry.sound = std::move(sound);
            } else {
                entry.sound = std::make_unique<Audio>(*listener_, item.file);
            }
        } catch (const std::exception& e) {
            // One bad file never aborts Play — the same bargain the physics
            // session strikes with a mesh it cannot cook.
            log("sound on \"" + nameOf(*item.node) + "\" failed to load " +
                item.file.string() + " (" + e.what() + ")");
            continue;
        }

        entry.sound->setLooping(config.loop);
        entry.sound->setVolume(config.volume);
        entry.sound->setPlaybackRate(config.rate);

        if (entry.spatial) {
            entry.spatial->setMinDistance(config.minDistance);
            entry.spatial->setMaxDistance(config.maxDistance);
            entry.spatial->setRolloffFactor(config.rolloff);
            entry.spatial->setDistanceModel(modelOf(config.model));
            // Parented, not owned: the node's world matrix is what feeds the
            // spatialization, and updateMatrixWorld reaches it every frame.
            item.node->addRef(*entry.spatial);
        }

        if (config.autoplay) {
            // stop() is a pause in miniaudio, so a second Play of the same
            // scene would otherwise resume mid-clip.
            entry.sound->seekToStart();
            entry.sound->play();
        }

        entries_.push_back(std::move(entry));
    }

    // Ray-traced acoustics, built only when both halves were authored: flagged
    // geometry AND something spatialized to hear through it. Anything less and
    // the session stays exactly what it was before acoustics existed — no BVH
    // built, no per-frame casts.
    const auto spatialCount = std::count_if(entries_.begin(), entries_.end(),
                                            [](const Entry& entry) { return entry.spatial != nullptr; });

    if (!surfaces.empty() && spatialCount > 0) {

        acousticScene_ = std::make_unique<AcousticScene>();
        for (const auto& item : surfaces) {
            acousticScene_->add(*item.mesh, item.surface);
        }
        surfaceCount_ = surfaces.size();

        acoustics_ = std::make_unique<AcousticsSystem>(*acousticScene_, *listener_);
        for (auto& entry : entries_) {
            if (entry.spatial) acoustics_->add(*entry.spatial);
        }

        log("acoustics: " + std::to_string(surfaces.size()) + " surface(s) occluding " +
            std::to_string(spatialCount) + " positional sound(s)");
    }
}

void AudioPlaySession::update(float dt) {

    if (!listener_) return;

    // The listener has no parent, so its world matrix is its own — copy the
    // host's world pose into it rather than parenting, which keeps this working
    // whichever camera the viewport is currently using (and with none at all).
    if (listenerHost_) {
        listenerHost_->getWorldPosition(listener_->position);
        listenerHost_->getWorldQuaternion(listener_->quaternion);
    }
    listener_->updateMatrixWorld(true);

    // miniaudio clamps the falloff AT max distance instead of silencing past
    // it (only the Linear model ever reaches zero), so an Inverse sound stayed
    // audible across the whole map at its max-distance level. The rings sell
    // max as the audible RANGE — enforce that here, for every model: authored
    // volume inside, a short ease-out band, silence beyond. Under None that
    // reads "constant level within range", which is what a zone ambience
    // wants. setVolume is the one channel miniaudio leaves for this.
    Vector3 nodePos;
    for (auto& entry : entries_) {
        if (!entry.spatial || !entry.node) continue;
        entry.node->getWorldPosition(nodePos);
        const float gate = distanceGate(nodePos.distanceTo(listener_->position), entry.maxDistance);
        entry.sound->setVolume(entry.volume * gate);
    }

    // After the gate, and it does not fight with it: occlusion is a separate
    // factor inside the sound, so the authored volume this just wrote survives.
    if (acoustics_) acoustics_->update(dt);
}

void AudioPlaySession::stop() {

    // Acoustics first — it borrows the sounds and the meshes below.
    acoustics_.reset();
    acousticScene_.reset();
    surfaceCount_ = 0;

    // Unlink before destroying: addRef leaves a raw pointer in the parent's
    // children list, so a sound freed while still parented strands it.
    for (auto& entry : entries_) {
        if (entry.spatial && entry.node) entry.node->remove(*entry.spatial);
    }
    // Sounds first, engine after — see the member order in the header.
    entries_.clear();
    listener_.reset();
}

bool AudioPlaySession::listenerReady() const {

    return listener_ != nullptr;
}

std::size_t AudioPlaySession::soundCount() const {

    return entries_.size();
}

float AudioPlaySession::distanceGate(float distance, float maxDistance) {

    // A tenth of the range, kept between 0.25 m (a point-blank range should
    // still not click shut) and 5 m (a longer fade audibly detaches from the
    // ring that claims to be the boundary).
    const float band = std::clamp(maxDistance * 0.1f, 0.25f, 5.f);
    if (distance <= maxDistance - band) return 1.f;
    if (distance >= maxDistance) return 0.f;
    const float t = (maxDistance - distance) / band;
    return t * t * (3.f - 2.f * t);
}

bool AudioPlaySession::isPlaying(const std::string& uuid) const {

    for (const auto& entry : entries_) {
        if (entry.uuid == uuid) return entry.sound && entry.sound->isPlaying();
    }
    return false;
}

bool AudioPlaySession::acousticsActive() const {

    return acoustics_ != nullptr;
}

std::size_t AudioPlaySession::acousticSurfaceCount() const {

    return surfaceCount_;
}

float AudioPlaySession::occlusionOf(const std::string& uuid) const {

    if (!acoustics_) return 0.f;

    for (const auto& entry : entries_) {
        if (entry.uuid == uuid && entry.spatial) return acoustics_->occlusionOf(*entry.spatial);
    }
    return 0.f;
}
