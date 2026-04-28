// Mipmap generation for the Wgpu renderer via render-pass blit chain.
// WebGPU has no glGenerateMipmap() equivalent, so each mip level is filled
// by rendering a fullscreen triangle that samples from the previous level.

#ifndef THREEPP_WGPUMIPMAPGENERATOR_HPP
#define THREEPP_WGPUMIPMAPGENERATOR_HPP

#include "WgpuState.hpp"
#include <webgpu/webgpu.h>

namespace threepp::wgpu {

    class WgpuMipmapGenerator {
    public:
        explicit WgpuMipmapGenerator(WgpuState& state);
        ~WgpuMipmapGenerator();

        // Generate mipmaps for a 2D texture.
        // The texture must have been created with WGPUTextureUsage_RenderAttachment
        // and mipLevelCount > 1. Mip level 0 must already contain data.
        void generate2D(WGPUTexture texture, uint32_t width, uint32_t height,
                        uint32_t mipLevels, WGPUTextureFormat format);

        // Generate mipmaps for a cube texture (6 array layers).
        void generateCube(WGPUTexture texture, uint32_t faceSize,
                          uint32_t mipLevels, WGPUTextureFormat format);

    private:
        WgpuState& state_;

        WGPURenderPipeline pipeline_ = nullptr;
        WGPUPipelineLayout pipelineLayout_ = nullptr;
        WGPUBindGroupLayout bgl_ = nullptr;
        WGPUSampler sampler_ = nullptr;
        WGPUShaderModule shader_ = nullptr;
        WGPUTextureFormat pipelineFormat_ = WGPUTextureFormat_Undefined;

        // Ensure the blit pipeline targets the given format.
        void ensurePipeline(WGPUTextureFormat format);

        // Blit one mip level into the next, for a single 2D array layer.
        // srcLevel is sampled, dstLevel is the render attachment.
        void blitLayer(WGPUCommandEncoder encoder,
                       WGPUTexture texture,
                       uint32_t baseArrayLayer,
                       uint32_t srcLevel,
                       WGPUTextureViewDimension viewDim);
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUMIPMAPGENERATOR_HPP
