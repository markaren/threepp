
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>


using namespace threepp;

namespace {

    Vector3 _pos;
    Vector3 _scale;
    Quaternion _quat;
    Vector3 _orientation;

    // 20 kHz (open) down to ~600 Hz (fully occluded), interpolated in log
    // frequency — linear interpolation spends the whole first half of the
    // range somewhere inaudible.
    float occlusionCutoff(float occlusion, ma_uint32 sampleRate) {

        const auto cutoff = static_cast<float>(20000.0 * std::pow(600.0 / 20000.0, occlusion));

        // Below Nyquist, or the biquad turns into a divide-by-nonsense.
        // Parenthesized: miniaudio drags in windows.h, and with it min/max macros.
        return (std::min)(cutoff, static_cast<float>(sampleRate) * 0.45f);
    }


    // miniaudio ships no reverb, so here is Schroeder's, by way of Jezar's
    // Freeverb: 8 damped feedback combs in parallel into 4 series allpasses,
    // per channel, with the channels detuned against each other to widen the
    // image. Delay lengths are the canonical 44.1 kHz ones, rescaled.
    constexpr int _combTuning[8]{1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
    constexpr int _allpassTuning[4]{556, 441, 341, 225};
    constexpr int _stereoSpread = 23;
    constexpr float _reverbInputGain = 0.015f;

    struct Comb {
        std::vector<float> buffer;
        size_t index{};
        float store{};

        float process(float input, float feedback, float damp) {

            const float output = buffer[index];
            store = output * (1.f - damp) + store * damp;
            buffer[index] = input + store * feedback;
            if (++index >= buffer.size()) index = 0;

            return output;
        }
    };

    struct Allpass {
        std::vector<float> buffer;
        size_t index{};

        float process(float input) {

            const float buffered = buffer[index];
            buffer[index] = input + buffered * 0.5f;
            if (++index >= buffer.size()) index = 0;

            return buffered - input;
        }
    };

    struct ReverbNode {
        ma_node_base base{};

        ma_uint32 channels{};
        std::vector<std::array<Comb, 8>> combs;
        std::vector<std::array<Allpass, 4>> allpasses;

        // Written from whatever thread drives the acoustics, read from the
        // audio thread.
        std::atomic<float> roomSize{0.7f};
        std::atomic<float> damp{0.4f};
        std::atomic<float> wet{0.f};
    };

    void reverbProcess(ma_node* pNode, const float** ppFramesIn, ma_uint32* pFrameCountIn,
                       float** ppFramesOut, ma_uint32* pFrameCountOut) {

        auto* node = reinterpret_cast<ReverbNode*>(pNode);

        const float* in = ppFramesIn[0];
        float* out = ppFramesOut[0];

        const ma_uint32 frames = *pFrameCountOut;
        const ma_uint32 channels = node->channels;

        const float feedback = node->roomSize.load(std::memory_order_relaxed);
        const float damp = node->damp.load(std::memory_order_relaxed);
        const float wet = node->wet.load(std::memory_order_relaxed);

        for (ma_uint32 frame = 0; frame < frames; ++frame) {
            for (ma_uint32 c = 0; c < channels; ++c) {

                const float input = in[frame * channels + c] * _reverbInputGain;

                float acc = 0.f;
                for (auto& comb : node->combs[c]) acc += comb.process(input, feedback, damp);
                for (auto& allpass : node->allpasses[c]) acc = allpass.process(acc);

                out[frame * channels + c] = acc * wet;
            }
        }

        *pFrameCountIn = frames;
        *pFrameCountOut = frames;
    }

    // CONTINUOUS_PROCESSING: the tail has to keep ringing after the sends fall silent.
    ma_node_vtable _reverbVtable{reverbProcess, nullptr, 1, 1, MA_NODE_FLAG_CONTINUOUS_PROCESSING};

    std::unique_ptr<ReverbNode> createReverbNode(ma_engine* engine) {

        ma_uint32 channels = ma_engine_get_channels(engine);
        const float scale = static_cast<float>(ma_engine_get_sample_rate(engine)) / 44100.f;

        const auto length = [scale](int tuning, int spread) {
            return static_cast<size_t>((std::max)(1, static_cast<int>(std::lround((tuning + spread) * scale))));
        };

        auto node = std::make_unique<ReverbNode>();
        node->channels = channels;
        node->combs.resize(channels);
        node->allpasses.resize(channels);

        for (ma_uint32 c = 0; c < channels; ++c) {

            const int spread = static_cast<int>(c) * _stereoSpread;

            for (int i = 0; i < 8; ++i) node->combs[c][i].buffer.assign(length(_combTuning[i], spread), 0.f);
            for (int i = 0; i < 4; ++i) node->allpasses[c][i].buffer.assign(length(_allpassTuning[i], spread), 0.f);
        }

        auto config = ma_node_config_init();
        config.vtable = &_reverbVtable;
        config.pInputChannels = &channels;
        config.pOutputChannels = &channels;

        if (ma_node_init(ma_engine_get_node_graph(engine), &config, nullptr, &node->base) != MA_SUCCESS) {
            return nullptr;
        }

        if (ma_node_attach_output_bus(&node->base, 0, ma_engine_get_endpoint(engine), 0) != MA_SUCCESS) {
            ma_node_uninit(&node->base, nullptr);
            return nullptr;
        }

        return node;
    }

}// namespace


struct AudioListener::Impl {

    ma_engine engine{};
    ma_engine_config engineConfig;

    std::unique_ptr<ReverbNode> reverb;


    Impl(): engineConfig(ma_engine_config_init()) {

        ma_result result = ma_engine_init(&engineConfig, &engine);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("[Audio] Failed to init engine");
        }
    }

    // The bus is shared by every source, so whoever needs it first builds it.
    ReverbNode* reverbNode() {

        if (!reverb) reverb = createReverbNode(&engine);

        return reverb.get();
    }

    ~Impl() {

        if (reverb) ma_node_uninit(&reverb->base, nullptr);

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

void AudioListener::setReverb(float rt60, float wet) {

    if (!pimpl_->reverb && (rt60 <= 0.f || wet <= 0.f)) return;

    auto* reverb = pimpl_->reverbNode();
    if (!reverb) return;

    // rt60 0 -> 0.7 (a cupboard), ~1 s -> ~0.89, >= 3 s -> 0.98 (a cathedral).
    // Freeverb's decay is not calibrated in seconds; the curve is tuning.
    const float roomSize = (std::clamp)(0.7f + 0.28f * (rt60 / (rt60 + 1.f)) / 0.75f, 0.7f, 0.98f);

    reverb->roomSize.store(roomSize, std::memory_order_relaxed);
    reverb->wet.store((std::clamp)(wet, 0.f, 1.f), std::memory_order_relaxed);
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
    ma_engine* engine_;
    AudioListener* listener_;

    ma_lpf_node lpf_{};
    bool lpfActive_{false};
    float lpfCutoff_{0};

    ma_splitter_node splitter_{};
    bool splitterActive_{false};

    float userVolume_{1};
    float occlusionGain_{1};

    Impl(AudioListener& ctx, const std::filesystem::path& file)
        : engine_(&ctx.pimpl_->engine), listener_(&ctx) {
        ma_result result = ma_sound_init_from_file(engine_, file.string().c_str(), MA_SOUND_FLAG_NO_SPATIALIZATION, nullptr, nullptr, &sound_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("[Audio] Failed to load audio file");
        }
    }

    // sound -> [lpf] -> [splitter] -> {endpoint, reverb}. One place for the
    // four lpf/splitter states, so splicing one never unpicks the other.
    // Spatialization lives inside the sound's own node, so panning and
    // attenuation survive the detour.
    void rebuildChain() {

        auto* endpoint = ma_engine_get_endpoint(engine_);
        auto* tail = splitterActive_ ? reinterpret_cast<ma_node*>(&splitter_) : endpoint;

        if (lpfActive_) {
            ma_node_attach_output_bus(&sound_, 0, &lpf_, 0);
            ma_node_attach_output_bus(&lpf_, 0, tail, 0);
        } else {
            ma_node_attach_output_bus(&sound_, 0, tail, 0);
        }

        if (splitterActive_) {
            ma_node_attach_output_bus(&splitter_, 0, endpoint, 0);

            if (auto* reverb = listener_->pimpl_->reverb.get()) {
                ma_node_attach_output_bus(&splitter_, 1, &reverb->base, 0);
            }
        }
    }

    void setReverbSend(float send) {

        if (!splitterActive_) {

            if (send <= 0.f) return;

            // The reverb bus has to exist before the splitter can be pointed at
            // it, so callers never have to order the two calls themselves.
            if (!listener_->pimpl_->reverbNode()) return;

            auto config = ma_splitter_node_config_init(ma_engine_get_channels(engine_));
            if (ma_splitter_node_init(ma_engine_get_node_graph(engine_), &config, nullptr, &splitter_) != MA_SUCCESS) {
                return;
            }

            splitterActive_ = true;
            rebuildChain();
        }

        ma_node_set_output_bus_volume(&splitter_, 1, (std::clamp)(send, 0.f, 1.f));
    }

    void applyVolume() {

        ma_sound_set_volume(&sound_, userVolume_ * occlusionGain_);
    }

    void setOcclusion(float occlusion) {

        // The filter does most of the perceptual work; keep some transmitted
        // energy so an occluded source never goes silent.
        occlusionGain_ = 1.f - 0.7f * occlusion;
        applyVolume();

        if (!lpfActive_ && occlusion <= 0.f) return;

        const auto sampleRate = ma_engine_get_sample_rate(engine_);
        const float cutoff = occlusionCutoff(occlusion, sampleRate);

        if (!lpfActive_) {

            auto config = ma_lpf_node_config_init(ma_engine_get_channels(engine_), sampleRate, cutoff, 2);
            if (ma_lpf_node_init(ma_engine_get_node_graph(engine_), &config, nullptr, &lpf_) != MA_SUCCESS) {
                return;// gain-only fallback
            }

            lpfActive_ = true;
            lpfCutoff_ = cutoff;

            // Ahead of the splitter if there is one: the reverb should hear the
            // muffled version too.
            rebuildChain();

            return;
        }

        if (std::abs(cutoff - lpfCutoff_) > 1.f) {

            auto config = ma_lpf_config_init(ma_format_f32, ma_engine_get_channels(engine_), sampleRate, cutoff, 2);
            ma_lpf_node_reinit(&config, &lpf_);

            lpfCutoff_ = cutoff;
        }
    }

    ~Impl() {
        ma_sound_uninit(&sound_);
        if (splitterActive_) {
            ma_splitter_node_uninit(&splitter_, nullptr);
        }
        if (lpfActive_) {
            ma_lpf_node_uninit(&lpf_, nullptr);
        }
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

    pimpl_->userVolume_ = volume;
    pimpl_->applyVolume();
}

void Audio::setReverbSend(float send) {

    pimpl_->setReverbSend((std::clamp)(send, 0.f, 1.f));
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

void PositionalAudio::setOcclusion(float occlusion) {

    pimpl_->setOcclusion((std::clamp)(occlusion, 0.f, 1.f));
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

    // Into the file-local scratch, not into this object's own `scale`: writing
    // the decomposed WORLD scale back into the LOCAL one re-multiplies it by
    // the parent's every frame, so a sound attached to a scaled node shrank
    // geometrically. Nothing audible depends on it (spatialization reads the
    // translation and the direction) — it is the node that went wrong.
    matrixWorld->decompose(_pos, _quat, _scale);

    _orientation.set(0, 0, -1).applyQuaternion(_quat);

    ma_sound_set_position(&pimpl_->sound_, _pos.x, _pos.y, _pos.z);
    ma_sound_set_direction(&pimpl_->sound_, _orientation.x, _orientation.y, _orientation.z);
}
