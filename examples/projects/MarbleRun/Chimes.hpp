// The sound of the sculpture: every hoop rings a note of its own as a marble
// goes through it, and the note is heard from where the hoop is. The notes are
// synthesised at startup, as the struck bars of a glockenspiel, and tuned to a
// pentatonic scale, so that the marbles can ring them in any order and it
// stays in tune.

#ifndef THREEPP_MARBLERUN_CHIMES_HPP
#define THREEPP_MARBLERUN_CHIMES_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"

#include <vector>

#if defined(THREEPP_WITH_AUDIO)

#include "threepp/audio/Audio.hpp"
#include "threepp/audio/WavFile.hpp"
#include "threepp/extras/audio/SynthUtil.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace marblerun {

    class Chimes {

    public:
        bool muted = false;

        // positions: where the hoops are, in the frame of `room`.
        // ears: what carries the listener, which is the camera.
        Chimes(const std::vector<threepp::Vector3>& positions, threepp::Object3D& ears, threepp::Object3D& room) {
            constexpr int scale[]{0, 2, 4, 7, 9};
            constexpr float lowest = 392.f;// G4

            const auto folder = std::filesystem::temp_directory_path() / "threepp_marblerun";
            std::filesystem::create_directories(folder);

            listener_.setMasterVolume(0.7f);
            ears.addRef(listener_);

            for (size_t i = 0; i < positions.size(); ++i) {
                const int semitones = scale[i % 5] + 12 * static_cast<int>(i / 5);
                const auto file = folder / ("chime_" + std::to_string(i) + ".wav");
                threepp::audio::writeWav(file, strike(lowest * std::pow(2.f, static_cast<float>(semitones) / 12)), sampleRate);

                // At full volume within arm's length, and fading with distance
                // from there: the sculpture is under a metre across, so that
                // from a marble's back the hoop ahead is far louder than the rest.
                auto voice = std::make_unique<threepp::PositionalAudio>(listener_, file);
                voice->position.copy(positions[i]);
                voice->setMinDistance(0.6f);
                voice->setRolloffFactor(1.f);
                room.addRef(*voice);
                voices_.push_back(std::move(voice));
            }
        }

        // speed: that of the marble going through [m/s]
        void ring(int index, float speed) {
            if (muted || index < 0 || index >= static_cast<int>(voices_.size())) return;

            auto& voice = *voices_[index];
            voice.stop();
            voice.seekToStart();
            voice.setVolume(std::clamp(0.2f + 0.7f * speed, 0.f, 0.7f));
            voice.play();
        }

    private:
        static constexpr int sampleRate = 44100;

        threepp::AudioListener listener_;
        std::vector<std::unique_ptr<threepp::PositionalAudio>> voices_;

        // A struck bar. Its overtones are not whole multiples of the note,
        // which is what makes it ring like metal and not like a flute.
        static std::vector<float> strike(float frequency) {
            struct Mode {
                float ratio, level, decay;// decay [s]
            };
            constexpr Mode modes[]{{1.f, 1.f, 0.5f}, {2.76f, 0.4f, 0.2f}, {5.4f, 0.18f, 0.08f}};

            std::vector<float> samples(static_cast<size_t>(sampleRate * 1.6f));
            for (size_t i = 0; i < samples.size(); ++i) {
                const float t = static_cast<float>(i) / sampleRate;

                float x = 0;
                for (const auto& mode : modes) {
                    x += mode.level * std::exp(-t / mode.decay) * std::sin(threepp::math::TWO_PI * frequency * mode.ratio * t);
                }
                // no click at either end
                const float fadeOut = std::min(1.f, static_cast<float>(samples.size() - i) / (0.05f * sampleRate));
                samples[i] = x * std::min(1.f, t / 0.002f) * fadeOut;
            }

            return threepp::audio::synth::normalized(std::move(samples), 0.7f);
        }
    };

}// namespace marblerun

#else

namespace marblerun {

    // Built without sound.
    class Chimes {

    public:
        bool muted = false;

        Chimes(const std::vector<threepp::Vector3>&, threepp::Object3D&, threepp::Object3D&) {}

        void ring(int, float) {}
    };

}// namespace marblerun

#endif

#endif//THREEPP_MARBLERUN_CHIMES_HPP
