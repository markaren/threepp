#ifndef WEBTIDE_PHILLIPSSPECTRUM_HPP
#define WEBTIDE_PHILLIPSSPECTRUM_HPP

#include "shaders.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace webtide {

    /// Generates the initial h0(k) Phillips spectrum texture (dispatched once).
    class PhillipsSpectrum {

    public:
        threepp::WgpuTexture h0;

        PhillipsSpectrum(threepp::WgpuRenderer& renderer, uint32_t textureSize, float tileSize)
            : h0(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA32Float),
              noise_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA32Float),
              uniformBuffer_(renderer, 32), // 5 fields padded to 32 bytes
              pipeline_(renderer, phillipsSpectrumWGSL, "computeSpectrum"),
              textureSize_(textureSize), tileSize_(tileSize) {

            // Generate noise matching original BabylonJS WebTide:
            // gaussianRandom() values stored into Uint8Array (truncate + mod 256),
            // then read by GPU as normalized [0,1] floats.
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

            auto gaussianRandom = [&]() -> float {
                return std::cos(2.0f * 3.14159265f * uniform(rng))
                     * std::sqrt(-2.0f * std::log(uniform(rng)));
            };

            // Replicate JavaScript Uint8Array truncation: trunc to int, mod 256
            auto toUint8 = [](float v) -> uint8_t {
                int n = static_cast<int>(std::trunc(v));
                return static_cast<uint8_t>(((n % 256) + 256) % 256);
            };

            std::vector<float> noiseData(textureSize * textureSize * 4);
            for (uint32_t i = 0; i < textureSize * textureSize; i++) {
                noiseData[i * 4 + 0] = static_cast<float>(toUint8(gaussianRandom())) / 255.0f;
                noiseData[i * 4 + 1] = static_cast<float>(toUint8(gaussianRandom())) / 255.0f;
                noiseData[i * 4 + 2] = 0.0f;
                noiseData[i * 4 + 3] = 0.0f;
            }
            noise_.write(noiseData.data(), noiseData.size() * sizeof(float));

            // Upload uniforms
            struct alignas(16) Params {
                uint32_t textureSize;
                float tileSize;
                float windTheta;
                float windSpeed;
                float smallWaveLengthCutoff;
                float _pad[3];
            } params{};
            params.textureSize = textureSize;
            params.tileSize = tileSize;
            params.windTheta = 0.0f;
            params.windSpeed = 31.0f;
            params.smallWaveLengthCutoff = 0.01f;
            uniformBuffer_.write(&params, sizeof(params));

            // Configure pipeline bindings
            pipeline_.setStorageTexture(0, h0);
            pipeline_.setTexture(1, noise_);
            pipeline_.setUniformBuffer(2, uniformBuffer_);

            // Dispatch once
            uint32_t groups = (textureSize + 7) / 8;
            pipeline_.dispatch(groups, groups, 1);
        }

    private:
        threepp::WgpuTexture noise_;
        threepp::WgpuBuffer uniformBuffer_;
        threepp::WgpuComputePipeline pipeline_;
        uint32_t textureSize_;
        float tileSize_;
    };

}// namespace webtide

#endif//WEBTIDE_PHILLIPSSPECTRUM_HPP
