#ifndef WEBTIDE_OCEANFOAM_HPP
#define WEBTIDE_OCEANFOAM_HPP

#include "DynamicSpectrum.hpp"
#include "IFFT.hpp"
#include "shaders.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

namespace webtide {

class OceanFoam {
public:
    OceanFoam(threepp::WgpuRenderer& renderer, uint32_t textureSize)
        : jacSpatial_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RG32Float),
          foam0_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA16Float),
          foam1_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA16Float),
          params_(renderer, 16),
          pipeline_(renderer, foamUpdateWGSL, "updateFoam"),
          textureSize_(textureSize)
    {
        // foam textures start zeroed (no foam initially); 4 float16 channels = 8 bytes/pixel
        std::vector<uint8_t> zeros(textureSize * textureSize * 8, 0);
        foam0_.write(zeros.data(), zeros.size());
        foam1_.write(zeros.data(), zeros.size());
    }

    // Returns the texture to use in the water shader (current read-only foam)
    threepp::WgpuTexture& currentFoam() {
        return current_ == 0 ? foam0_ : foam1_;
    }

    // Call once per frame after cascade1 DynamicSpectrum::generate() has run
    void update(DynamicSpectrum& cascade1, IFFT& ifft, float dt, float lambda, float decay) {
        uint32_t groups = (textureSize_ + 7) / 8;

        // 1. IFFT the Jacobian spectrum -> spatial Jacobian components
        ifft.applyToTexture(cascade1.jacDiag, jacSpatial_);
        // After IFFT, jacSpatial_ contains (.r=Jxx, .g=Jzz) in spatial domain

        // 2. Write foam params
        // jacScale normalises the raw IFFT Jacobian by the cascade tile size (same as height).
        struct { float lambda; float decay; float dt; float jacScale; } p{lambda, decay, dt, 1.0f/80.0f};
        params_.write(&p, sizeof(p));

        // 3. Run foam update: read jacSpatial_ + current foam, write to other foam buffer
        int next = 1 - current_;
        threepp::WgpuTexture& foamIn  = current_ == 0 ? foam0_ : foam1_;
        threepp::WgpuTexture& foamOut = current_ == 0 ? foam1_ : foam0_;

        pipeline_.setTexture(0, jacSpatial_);
        pipeline_.setTexture(1, foamIn);
        pipeline_.setStorageTexture(2, foamOut);
        pipeline_.setUniformBuffer(3, params_);
        pipeline_.dispatch(groups, groups, 1);

        current_ = next;
    }

private:
    threepp::WgpuTexture jacSpatial_;
    threepp::WgpuTexture foam0_, foam1_;
    threepp::WgpuBuffer params_;
    threepp::WgpuComputePipeline pipeline_;
    uint32_t textureSize_;
    int current_ = 0;
};

} // namespace webtide
#endif//WEBTIDE_OCEANFOAM_HPP
