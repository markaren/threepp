// GGX-convolved prefiltered radiance environment map (PMREM).
// Each mip level of the output texture is convolved with a GGX lobe whose
// roughness matches `level / (mipLevels-1)`, so materials can sample the
// mip chain by `roughness * maxMip` and get physically-correct rough reflections.
//
// Currently implemented for equirectangular 2D textures (the common case for
// HDR env maps loaded via RGBELoader). Cube maps still use box-filter mipmap
// generation until cube PMREM is implemented.

#ifndef THREEPP_WGPUPMREM_HPP
#define THREEPP_WGPUPMREM_HPP

#include "WgpuState.hpp"
#include <webgpu/webgpu.h>

namespace threepp::wgpu {

    class WgpuPMREM {
    public:
        explicit WgpuPMREM(WgpuState& state);
        ~WgpuPMREM();

        // Prefilter an equirectangular 2D texture's mip chain in-place.
        // The texture must have been created with RenderAttachment usage,
        // mipLevelCount > 1, and mip 0 already populated. Mips 1..N-1 are
        // overwritten with GGX-convolved versions at increasing roughness.
        void prefilter2D(WGPUTexture texture, uint32_t width, uint32_t height,
                         uint32_t mipLevels, WGPUTextureFormat format);

    private:
        WgpuState& state_;

        WGPURenderPipeline pipeline_ = nullptr;
        WGPUPipelineLayout pipelineLayout_ = nullptr;
        WGPUBindGroupLayout bgl_ = nullptr;
        WGPUShaderModule shader_ = nullptr;
        WGPUSampler sampler_ = nullptr;
        WGPUTextureFormat pipelineFormat_ = WGPUTextureFormat_Undefined;

        void ensurePipeline(WGPUTextureFormat format);
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUPMREM_HPP
