#ifndef WEBTIDE_DYNAMICSPECTRUM_HPP
#define WEBTIDE_DYNAMICSPECTRUM_HPP

#include "shaders.hpp"
#include "PhillipsSpectrum.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/renderers/dawn/DawnTexture.hpp"
#include "threepp/renderers/dawn/DawnBuffer.hpp"
#include "threepp/renderers/dawn/DawnComputePipeline.hpp"

#include <cstring>

namespace webtide {

    /// Evolves the spectrum in time, producing ht, dht, and displacement textures.
    class DynamicSpectrum {

    public:
        threepp::DawnTexture ht;
        threepp::DawnTexture dht;
        threepp::DawnTexture displacement;

        DynamicSpectrum(threepp::DawnRenderer& renderer, PhillipsSpectrum& initialSpectrum,
                        uint32_t textureSize, float tileSize)
            : ht(renderer, textureSize, textureSize, threepp::DawnTexture::Format::RG32Float),
              dht(renderer, textureSize, textureSize, threepp::DawnTexture::Format::RG32Float),
              displacement(renderer, textureSize, textureSize, threepp::DawnTexture::Format::RG32Float),
              uniformBuffer_(renderer, 16), // textureSize(4) + tileSize(4) + elapsedSeconds(4) + pad(4) = 16
              pipeline_(renderer, dynamicSpectrumWGSL, "computeSpectrum"),
              textureSize_(textureSize), tileSize_(tileSize) {

            pipeline_.setTexture(0, initialSpectrum.h0);
            pipeline_.setStorageTexture(1, ht);
            pipeline_.setStorageTexture(2, dht);
            pipeline_.setStorageTexture(3, displacement);
            pipeline_.setUniformBuffer(4, uniformBuffer_);
        }

        void generate(float elapsedSeconds) {
            struct Params {
                uint32_t textureSize;
                float tileSize;
                float elapsedSeconds;
                float _pad;
            } params{};
            params.textureSize = textureSize_;
            params.tileSize = tileSize_;
            params.elapsedSeconds = elapsedSeconds;
            uniformBuffer_.write(&params, sizeof(params));

            uint32_t groups = (textureSize_ + 7) / 8;
            pipeline_.dispatch(groups, groups, 1);
        }

    private:
        threepp::DawnBuffer uniformBuffer_;
        threepp::DawnComputePipeline pipeline_;
        uint32_t textureSize_;
        float tileSize_;
    };

}// namespace webtide

#endif//WEBTIDE_DYNAMICSPECTRUM_HPP
