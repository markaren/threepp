
#ifndef THREEPP_AUDIOLISTENER_HPP
#define THREEPP_AUDIOLISTENER_HPP

#include "threepp/core/Object3D.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <stdexcept>

namespace threepp {

    // The Audio API is highly experimental
    class AudioListener: public Object3D {

    public:
        AudioListener(): engineConfig(ma_engine_config_init()) {

            ma_result result = ma_engine_init(&engineConfig, &engine);
            if (result != MA_SUCCESS) {
                throw std::runtime_error("[Audio] Failed to init engine");
            }
        }

        AudioListener(AudioListener&&) = delete;
        AudioListener& operator=(AudioListener&&) = delete;
        AudioListener(const AudioListener&) = delete;
        AudioListener& operator=(const AudioListener&) = delete;

        [[nodiscard]] float getMasterVolume() const {

            float volume;
            ma_device_get_master_volume(engine.pDevice, &volume);

            return volume;
        }

        void setMasterVolume(float volume) {

            ma_device_set_master_volume(engine.pDevice, volume);
        }

        void updateMatrixWorld(bool force) override {
            Object3D::updateMatrixWorld(force);

            Vector3 pos;
            Vector3 scale;
            Quaternion quat;
            matrixWorld->decompose(pos, quat, scale);

            Vector3 _orientation;
            _orientation.set(0, 0, -1).applyQuaternion(quat);

            ma_engine_listener_set_position(&engine, 0, pos.x, pos.y, pos.z);
            ma_engine_listener_set_direction(&engine, 0, _orientation.x, _orientation.y, _orientation.z);
        }

        ~AudioListener() override {
            ma_engine_uninit(&engine);
        }

    private:
        ma_engine engine;
        ma_engine_config engineConfig;

        friend class Audio;
    };

}// namespace threepp

#endif//THREEPP_AUDIOLISTENER_HPP
