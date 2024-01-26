
#ifndef THREEPP_AUDIO_HPP
#define THREEPP_AUDIO_HPP

#include "threepp/core/Object3D.hpp"

#include <filesystem>
#include <memory>


namespace threepp {

    // The Audio API is highly experimental
    class AudioListener: public Object3D {

    public:
        AudioListener();

        AudioListener(AudioListener&&) = delete;
        AudioListener& operator=(AudioListener&&) = delete;
        AudioListener(const AudioListener&) = delete;
        AudioListener& operator=(const AudioListener&) = delete;

        [[nodiscard]] float getMasterVolume() const;

        void setMasterVolume(float volume);

        void updateMatrixWorld(bool force) override;

        ~AudioListener() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        friend class Audio;
    };

    class Audio {

    public:
        Audio(AudioListener& ctx, const std::filesystem::path& file);

        Audio(Audio&&) = delete;
        Audio& operator=(Audio&&) = delete;
        Audio(const Audio&) = delete;
        Audio& operator=(const Audio&) = delete;

        [[nodiscard]] bool isPlaying() const;

        void play();

        void setVolume(float volume);

        void togglePlay();

        void stop();

        void setLooping(bool flag);

        virtual ~Audio();

    protected:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

    class PositionalAudio: public Audio, public Object3D {

    public:
        PositionalAudio(AudioListener& ctx, const std::filesystem::path& file);

        void updateMatrixWorld(bool force) override;
    };

}// namespace threepp


#endif//THREEPP_AUDIO_HPP
