
#ifndef THREEPP_AUDIO_HPP
#define THREEPP_AUDIO_HPP

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <filesystem>
#include <memory>

namespace detail {

    using CallbackType = void (*)(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

}

namespace threepp {

    class Audio {

    public:
        bool autoPlay{false};

        Audio(const std::filesystem::path& file) {

            engineConfig = ma_engine_config_init();

            ma_result result = ma_engine_init(&engineConfig, &engine);
            if (result != MA_SUCCESS) {
                throw std::runtime_error("[Audio] Failed init engine");
            }

            result = ma_sound_init_from_file(&engine, file.string().c_str(), 0, nullptr, nullptr, &sound);
            if (result != MA_SUCCESS) {
                throw std::runtime_error("[Audio] Failed to load audio file");
            }

            if (autoPlay) {
                play();
            }
        }

        [[nodiscard]] bool isPlaying() const {

            return ma_sound_is_playing(&sound);
        }

        void play() {
            ma_sound_start(&sound);
        }

        void togglePlay() {
            if (!isPlaying()) {
                play();
            } else {
                stop();
            }
        }

        void stop() {
            ma_sound_stop(&sound);
        }

        float getMasterVolume() {

            float volume;
            ma_device_get_master_volume(engine.pDevice, &volume);

            return volume;
        }

        void setLooping(bool flag) {
            ma_sound_set_looping(&sound, flag);
        }

        void setMasterVolume(float volume) {

            ma_device_set_master_volume(engine.pDevice, volume);
        }

        ~Audio() {
            ma_sound_uninit(&sound);
            ma_engine_uninit(&engine);
        }

    private:

        ma_engine engine;
        ma_engine_config engineConfig;
        ma_sound sound;
    };

}// namespace threepp


#endif//THREEPP_AUDIO_HPP
