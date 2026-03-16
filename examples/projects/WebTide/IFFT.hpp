#ifndef WEBTIDE_IFFT_HPP
#define WEBTIDE_IFFT_HPP

#include "shaders.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/renderers/dawn/GPUTexture.hpp"
#include "threepp/renderers/dawn/GPUBuffer.hpp"
#include "threepp/renderers/dawn/ComputePipeline.hpp"

#include <cmath>
#include <cstring>

namespace webtide {

    /// GPU-based inverse FFT using butterfly operations.
    /// Converts frequency-domain spectrum textures to spatial-domain maps.
    class IFFT {

    public:
        IFFT(threepp::DawnRenderer& renderer, uint32_t textureSize)
            : renderer_(renderer),
              textureSize_(textureSize),
              logSize_(static_cast<uint32_t>(std::log2(textureSize))),
              twiddleTable_(renderer, logSize_, textureSize, threepp::GPUTexture::Format::RGBA32Float),
              settings_(renderer, 16), // step(4) + textureSize(4) + pad(8) = 16
              copyParams_(renderer, 16), // width(4) + height(4) + pad(8) = 16
              twiddlePipeline_(renderer, twiddleFactorsWGSL, "precomputeTwiddleFactorsAndInputIndices"),
              horizontalPipeline_(renderer, horizontalStepIfftWGSL, "horizontalStepInverseFFT"),
              verticalPipeline_(renderer, verticalStepIfftWGSL, "verticalStepInverseFFT"),
              permutePipeline_(renderer, permutationWGSL, "permute"),
              copyPipeline_(renderer, copyTextureWGSL, "main") {

            // Precompute twiddle factors (dispatched once)
            twiddlePipeline_.setStorageTexture(0, twiddleTable_);
            twiddlePipeline_.setUniformBuffer(1, settings_);

            struct { int32_t step; int32_t textureSize; int32_t _pad[2]; } twiddleParams{};
            twiddleParams.textureSize = static_cast<int32_t>(textureSize);
            settings_.write(&twiddleParams, 16);

            twiddlePipeline_.dispatch(logSize_, textureSize / 2 / 8, 1);

            // Set up horizontal/vertical pipeline static bindings
            horizontalPipeline_.setUniformBuffer(0, settings_);
            horizontalPipeline_.setTexture(1, twiddleTable_);

            verticalPipeline_.setUniformBuffer(0, settings_);
            verticalPipeline_.setTexture(1, twiddleTable_);

            // Copy pipeline uniform
            copyPipeline_.setUniformBuffer(2, copyParams_);
        }

        /// Apply the inverse FFT to the input texture, storing result in output.
        /// Both textures will be modified (ping-pong).
        void applyToTexture(threepp::GPUTexture& input, threepp::GPUTexture& output) {
            uint32_t groups = (textureSize_ + 7) / 8;

            bool pingPong = false;
            // Horizontal passes
            for (uint32_t i = 0; i < logSize_; ++i) {
                pingPong = !pingPong;

                struct { int32_t step; int32_t textureSize; int32_t _pad[2]; } params{};
                params.step = static_cast<int32_t>(i);
                params.textureSize = static_cast<int32_t>(textureSize_);
                settings_.write(&params, 16);

                if (pingPong) {
                    horizontalPipeline_.setTexture(2, input);
                    horizontalPipeline_.setStorageTexture(3, output);
                } else {
                    horizontalPipeline_.setTexture(2, output);
                    horizontalPipeline_.setStorageTexture(3, input);
                }

                horizontalPipeline_.dispatch(groups, groups, 1);
            }

            // Vertical passes
            for (uint32_t i = 0; i < logSize_; ++i) {
                pingPong = !pingPong;

                struct { int32_t step; int32_t textureSize; int32_t _pad[2]; } params{};
                params.step = static_cast<int32_t>(i);
                params.textureSize = static_cast<int32_t>(textureSize_);
                settings_.write(&params, 16);

                if (pingPong) {
                    verticalPipeline_.setTexture(2, input);
                    verticalPipeline_.setStorageTexture(3, output);
                } else {
                    verticalPipeline_.setTexture(2, output);
                    verticalPipeline_.setStorageTexture(3, input);
                }

                verticalPipeline_.dispatch(groups, groups, 1);
            }

            // If pingPong is true, result is in output — copy back to input
            if (pingPong) {
                copyTexture(output, input);
            }

            // Permutation (sign-flip)
            permutePipeline_.setTexture(0, input);
            permutePipeline_.setStorageTexture(1, output);
            permutePipeline_.dispatch(groups, groups, 1);

            // Copy result back to input for consistent state
            copyTexture(output, input);
        }

    private:
        threepp::DawnRenderer& renderer_;
        uint32_t textureSize_;
        uint32_t logSize_;

        threepp::GPUTexture twiddleTable_;
        threepp::GPUBuffer settings_;
        threepp::GPUBuffer copyParams_;

        threepp::ComputePipeline twiddlePipeline_;
        threepp::ComputePipeline horizontalPipeline_;
        threepp::ComputePipeline verticalPipeline_;
        threepp::ComputePipeline permutePipeline_;
        threepp::ComputePipeline copyPipeline_;

        void copyTexture(threepp::GPUTexture& src, threepp::GPUTexture& dst) {
            copyPipeline_.setStorageTexture(0, dst);
            copyPipeline_.setTexture(1, src);

            struct { uint32_t width; uint32_t height; uint32_t _pad[2]; } params{};
            params.width = textureSize_;
            params.height = textureSize_;
            copyParams_.write(&params, 16);

            uint32_t groups = (textureSize_ + 7) / 8;
            copyPipeline_.dispatch(groups, groups, 1);
        }
    };

}// namespace webtide

#endif//WEBTIDE_IFFT_HPP
