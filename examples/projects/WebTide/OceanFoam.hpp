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
        : jacSpatial0_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RG32Float),
          jacSpatial1_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RG32Float),
          jacSpatial2_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RG32Float),
          foam0_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA16Float),
          foam1_(renderer, textureSize, textureSize, threepp::WgpuTexture::Format::RGBA16Float),
          params_(renderer, 32),   // 8 x f32 = 32 bytes
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

    // Call once per frame after all 3 DynamicSpectrum::generate() calls have run.
    // Combines Jacobian contributions from all cascades: J_total = J0 + J1 + J2.
    // Each cascade's raw IFFT Jacobian is normalised by 1/(2*tileSize) before summing
    // so that lambda applies uniformly regardless of cascade scale.
    void update(DynamicSpectrum& cascade0, IFFT& ifft0,
                DynamicSpectrum& cascade1, IFFT& ifft1,
                DynamicSpectrum& cascade2, IFFT& ifft2,
                float dt, float lambda, float decay,
                float tileSize0, float tileSize1, float tileSize2) {
        uint32_t groups = (textureSize_ + 7) / 8;

        // 1. IFFT the Jacobian spectrum for each cascade -> spatial Jacobian (Jxx, Jzz)
        ifft0.applyToTexture(cascade0.jacDiag, jacSpatial0_);
        ifft1.applyToTexture(cascade1.jacDiag, jacSpatial1_);
        ifft2.applyToTexture(cascade2.jacDiag, jacSpatial2_);

        // 2. Write foam params — jacScaleN = 1/(2*tileSizeN), consistent with height IFFT normalization
        struct Params {
            float lambda;
            float decay;
            float dt;
            float jacScale0;
            float jacScale1;
            float jacScale2;
            float pad0;
            float pad1;
        } p{lambda, decay, dt,
            1.0f / (2.0f * tileSize0),
            1.0f / (2.0f * tileSize1),
            1.0f / (2.0f * tileSize2),
            0.0f, 0.0f};
        params_.write(&p, sizeof(p));

        // 3. Run foam update: read 3 Jacobian textures + current foam, write to other foam buffer
        int next = 1 - current_;
        threepp::WgpuTexture& foamIn  = current_ == 0 ? foam0_ : foam1_;
        threepp::WgpuTexture& foamOut = current_ == 0 ? foam1_ : foam0_;

        pipeline_.setTexture(0, jacSpatial0_);
        pipeline_.setTexture(1, jacSpatial1_);
        pipeline_.setTexture(2, jacSpatial2_);
        pipeline_.setTexture(3, foamIn);
        pipeline_.setStorageTexture(4, foamOut);
        pipeline_.setUniformBuffer(5, params_);
        pipeline_.dispatch(groups, groups, 1);

        current_ = next;
    }

private:
    threepp::WgpuTexture jacSpatial0_, jacSpatial1_, jacSpatial2_;
    threepp::WgpuTexture foam0_, foam1_;
    threepp::WgpuBuffer params_;
    threepp::WgpuComputePipeline pipeline_;
    uint32_t textureSize_;
    int current_ = 0;
};

} // namespace webtide
#endif//WEBTIDE_OCEANFOAM_HPP
