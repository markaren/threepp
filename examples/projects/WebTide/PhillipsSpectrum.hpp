#ifndef WEBTIDE_PHILLIPSSPECTRUM_HPP
#define WEBTIDE_PHILLIPSSPECTRUM_HPP

#include "shaders.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/renderers/dawn/GPUTexture.hpp"
#include "threepp/renderers/dawn/GPUBuffer.hpp"
#include "threepp/renderers/dawn/ComputePipeline.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace webtide {

    /// Generates the initial h0(k) Phillips spectrum texture (dispatched once).
    class PhillipsSpectrum {

    public:
        threepp::GPUTexture h0;

        PhillipsSpectrum(threepp::DawnRenderer& renderer, uint32_t textureSize, float tileSize)
            : h0(renderer, textureSize, textureSize, threepp::GPUTexture::Format::RGBA32Float),
              noise_(renderer, textureSize, textureSize, threepp::GPUTexture::Format::RGBA32Float),
              uniformBuffer_(renderer, 32), // 5 fields padded to 32 bytes
              pipeline_(renderer, phillipsSpectrumWGSL, "computeSpectrum"),
              textureSize_(textureSize), tileSize_(tileSize) {

            // Generate Gaussian noise texture on CPU (Box-Muller transform)
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);

            std::vector<float> noiseData(textureSize * textureSize * 4);
            for (uint32_t i = 0; i < textureSize * textureSize; i++) {
                noiseData[i * 4 + 0] = dist(rng); // R
                noiseData[i * 4 + 1] = dist(rng); // G
                noiseData[i * 4 + 2] = 0.0f;      // B (unused)
                noiseData[i * 4 + 3] = 0.0f;      // A (unused)
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
        threepp::GPUTexture noise_;
        threepp::GPUBuffer uniformBuffer_;
        threepp::ComputePipeline pipeline_;
        uint32_t textureSize_;
        float tileSize_;
    };

}// namespace webtide

#endif//WEBTIDE_PHILLIPSSPECTRUM_HPP
