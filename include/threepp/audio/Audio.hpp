
#ifndef THREEPP_AUDIO_HPP
#define THREEPP_AUDIO_HPP

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "AudioListener.hpp"
#include "threepp/audio/AudioListener.hpp"

#include <filesystem>
#include <memory>

namespace detail {

    using CallbackType = void(*)(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

}

namespace threepp {

    class Audio {

    public:

        bool autoPlay{false};

        Audio(const std::filesystem::path& file) {

            result = ma_decoder_init_file("data/sounds/376737_Skullbeatz___Bad_Cat_Maste.mp3", nullptr, &decoder);
            if (result != MA_SUCCESS) {
                throw std::runtime_error("[Audio] Failed to load audio file");
            }

            detail::CallbackType callback = [](ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount){

                auto* audio = static_cast<Audio*>(pDevice->pUserData);

                auto* pDecoder = static_cast<ma_decoder*>(&audio->decoder);
                if (pDecoder == nullptr) {
                    return;
                }

                ma_decoder_read_pcm_frames(pDecoder, pOutput, frameCount, nullptr);

                (void)pInput;
            };

            deviceConfig = ma_device_config_init(ma_device_type_playback);
            deviceConfig.playback.format   = decoder.outputFormat;
            deviceConfig.playback.channels = decoder.outputChannels;
            deviceConfig.sampleRate        = decoder.outputSampleRate;
            deviceConfig.dataCallback      = *callback;
            deviceConfig.pUserData         = this;

            if (ma_device_init(nullptr, &deviceConfig, &device) != MA_SUCCESS) {
                printf("Failed to open playback device.\n");
                ma_decoder_uninit(&decoder);
                throw std::runtime_error("[Audio] Failed to open playback device.");
            }

            if (autoPlay) {
                play();
            }

        }

        bool isPlaying() {

            return playing_;
        }

        void play() {
            if (ma_device_start(&device) != MA_SUCCESS) {
                throw std::runtime_error("Failed to start playback device.");
            }

            playing_ = true;
        }

        void play(bool flag) {
            if (!isPlaying()) {
                play();
            } else {
                stop();
            }
        }

        void stop() {
            if (ma_device_stop(&device) != MA_SUCCESS) {
                // Handle error
            }
            playing_ = false;
        }

        float getMasterVolume() {

            float volume;
            ma_device_get_master_volume(&device, &volume);

            return volume;
        }

        void setLooping(bool flag) {
            ma_data_source_set_looping(&decoder, flag);
        }

        void setMasterVolume(float volume) {

            ma_device_set_master_volume(&device, volume);
        }

        ~Audio() {
            ma_device_uninit(&device);
            ma_decoder_uninit(&decoder);
        }

    private:
        ma_result result;
        ma_decoder decoder;
        ma_device_config deviceConfig;
        ma_device device;

        bool playing_{false};

    };

}


#endif//THREEPP_AUDIO_HPP
