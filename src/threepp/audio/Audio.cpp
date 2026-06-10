
#include "threepp/audio/Audio.hpp"


#ifndef MINIAUDIO_IMPLEMENTATION

#if defined(__GNUC__) || defined(__clang__)
// Temporarily disable specific warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"    // Disables all warnings
#pragma GCC diagnostic ignored "-Wextra"  // Disables extra warnings
#pragma GCC diagnostic ignored "-Wpedantic"  // Disables pedantic warnings
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#if defined(__GNUC__) || defined(__clang__)
// Re-enable the warnings
#pragma GCC diagnostic pop
#endif

#endif

#include <stdexcept>


using namespace threepp;

namespace {

    Vector3 _pos;
    Vector3 _scale;
    Quaternion _quat;
    Vector3 _orientation;

}// namespace


struct AudioListener::Impl {

    ma_engine engine{};
    ma_engine_config engineConfig;


    Impl(): engineConfig(ma_engine_config_init()) {

        ma_result result = ma_engine_init(&engineConfig, &engine);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("[Audio] Failed to init engine");
        }
    }

    ~Impl() {

        ma_engine_uninit(&engine);
    }
};


AudioListener::AudioListener()
    : pimpl_(std::make_unique<Impl>()) {}

float AudioListener::getMasterVolume() const {

    float volume{};
    ma_device_get_master_volume(pimpl_->engine.pDevice, &volume);

    return volume;
}
void AudioListener::setMasterVolume(float volume) {

    ma_device_set_master_volume(pimpl_->engine.pDevice, volume);
}

void AudioListener::updateMatrixWorld(bool force) {
    Object3D::updateMatrixWorld(force);

    matrixWorld->decompose(_pos, _quat, _scale);

    _orientation.set(0, 0, -1).applyQuaternion(_quat);

    ma_engine_listener_set_position(&pimpl_->engine, 0, _pos.x, _pos.y, _pos.z);
    ma_engine_listener_set_direction(&pimpl_->engine, 0, _orientation.x, _orientation.y, _orientation.z);
}

AudioListener::~AudioListener() = default;


struct Audio::Impl {

    ma_sound sound_{};

    Impl(AudioListener& ctx, const std::filesystem::path& file) {
        ma_result result = ma_sound_init_from_file(&ctx.pimpl_->engine, file.string().c_str(), MA_SOUND_FLAG_NO_SPATIALIZATION, nullptr, nullptr, &sound_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("[Audio] Failed to load audio file");
        }
    }

    ~Impl() {
        ma_sound_uninit(&sound_);
    }
};

Audio::Audio(AudioListener& ctx, const std::filesystem::path& file)
    : pimpl_(std::make_unique<Impl>(ctx, file)) {}

Audio::~Audio() = default;

void Audio::setLooping(bool flag) {

    ma_sound_set_looping(&pimpl_->sound_, flag);
}

void Audio::stop() {

    ma_sound_stop(&pimpl_->sound_);
}

void Audio::seekToStart() {

    ma_sound_seek_to_pcm_frame(&pimpl_->sound_, 0);
}

void Audio::togglePlay() {

    if (!isPlaying()) {
        play();
    } else {
        stop();
    }
}

void Audio::setVolume(float volume) {

    ma_sound_set_volume(&pimpl_->sound_, volume);
}

void Audio::setPlaybackRate(float rate) {

    ma_sound_set_pitch(&pimpl_->sound_, rate);
}

void Audio::play() {

    ma_sound_start(&pimpl_->sound_);
}

bool Audio::isPlaying() const {

    return ma_sound_is_playing(&pimpl_->sound_);
}

PositionalAudio::PositionalAudio(AudioListener& ctx, const std::filesystem::path& file): Audio(ctx, file) {

    ma_sound_set_spatialization_enabled(&pimpl_->sound_, true);
}

void PositionalAudio::setMinDistance(float distance) {

    ma_sound_set_min_distance(&pimpl_->sound_, distance);
}

void PositionalAudio::setMaxDistance(float distance) {

    ma_sound_set_max_distance(&pimpl_->sound_, distance);
}

void PositionalAudio::setRolloffFactor(float rolloff) {

    ma_sound_set_rolloff(&pimpl_->sound_, rolloff);
}

void PositionalAudio::setDistanceModel(DistanceModel model) {

    ma_attenuation_model m = ma_attenuation_model_inverse;
    switch (model) {
        case DistanceModel::None: m = ma_attenuation_model_none; break;
        case DistanceModel::Inverse: m = ma_attenuation_model_inverse; break;
        case DistanceModel::Linear: m = ma_attenuation_model_linear; break;
        case DistanceModel::Exponential: m = ma_attenuation_model_exponential; break;
    }
    ma_sound_set_attenuation_model(&pimpl_->sound_, m);
}

void PositionalAudio::updateMatrixWorld(bool force) {
    Object3D::updateMatrixWorld(force);

    matrixWorld->decompose(_pos, _quat, scale);

    _orientation.set(0, 0, -1).applyQuaternion(_quat);

    ma_sound_set_position(&pimpl_->sound_, _pos.x, _pos.y, _pos.z);
    ma_sound_set_direction(&pimpl_->sound_, _orientation.x, _orientation.y, _orientation.z);
}
