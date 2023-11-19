
#ifndef THREEPP_AUDIO_HPP
#define THREEPP_AUDIO_HPP

#include "threepp/core/Object3D.hpp"

#include <filesystem>
#include <memory>

#include "AudioListener.hpp"


namespace threepp {

    class Audio {

    public:
        Audio(AudioListener& ctx, const std::filesystem::path& file): ctx_(ctx) {

            ma_result result = ma_sound_init_from_file(&ctx_.engine, file.string().c_str(), MA_SOUND_FLAG_NO_SPATIALIZATION, nullptr, nullptr, &sound_);
            if (result != MA_SUCCESS) {
                throw std::runtime_error("[Audio] Failed to load audio file");
            }
        }

        Audio(Audio&&) = delete;
        Audio& operator=(Audio&&) = delete;
        Audio(const Audio&) = delete;
        Audio& operator=(const Audio&) = delete;

        [[nodiscard]] bool isPlaying() const {

            return ma_sound_is_playing(&sound_);
        }

        void play() {

            ma_sound_start(&sound_);
        }

        void setVolume(float volume) {

            ma_sound_set_volume(&sound_, volume);
        }

        void togglePlay() {

            if (!isPlaying()) {
                play();
            } else {
                stop();
            }
        }

        void stop() {

            ma_sound_stop(&sound_);
        }

        void setLooping(bool flag) {
            ma_sound_set_looping(&sound_, flag);
        }

        ~Audio() {
            ma_sound_uninit(&sound_);
        }

    protected:
        AudioListener& ctx_;
        ma_sound sound_;
    };

    class PositionalAudio: public Audio, public Object3D {

    public:
        PositionalAudio(AudioListener& ctx, const std::filesystem::path& file): Audio(ctx, file) {

            ma_sound_set_spatialization_enabled(&sound_, true);
        }

        void updateMatrixWorld(bool force) override {
            Object3D::updateMatrixWorld(force);

            matrixWorld->decompose(_pos, _quat, scale);

            _orientation.set(0, 0, -1).applyQuaternion(_quat);

            ma_sound_set_position(&sound_, _pos.x, _pos.y, _pos.z);
            ma_sound_set_direction(&sound_, _orientation.x, _orientation.y, _orientation.z);
        }
    };

}// namespace threepp


#endif//THREEPP_AUDIO_HPP
