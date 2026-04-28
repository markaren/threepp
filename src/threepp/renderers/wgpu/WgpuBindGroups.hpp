// Bind group assembly for the Wgpu renderer.
// Builds WGPUBindGroupEntry arrays for standard and custom material draws.
// Uses a persistent internal vector to avoid per-draw heap allocations.

#ifndef THREEPP_WGPUBINDGROUPS_HPP
#define THREEPP_WGPUBINDGROUPS_HPP

#include "WgpuMaterials.hpp"
#include "WgpuShaders.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class ShaderMaterial;
    class Texture;
    class WgpuTexture;
}

namespace threepp::wgpu {

    class WgpuTextures;
    class WgpuShadowMap;

    // Sorted texture list for custom shader bind groups: (name, {textureView, sampler})
    using TextureList = std::vector<std::pair<std::string, std::pair<WGPUTextureView, WGPUSampler>>>;

    // Input parameters for building standard material bind group entries.
    // All members are references/pointers -- no copies.
    struct BindGroupInputs {
        uint64_t features;
        WGPUBuffer transformBuffer;
        WGPUBuffer materialBuffer;
        WGPUBuffer lightBuffer;         // may be null
        size_t lightUniformSize;        // runtime light buffer size
        const MaterialParams& params;
        WgpuTextures& textures;
        WgpuShadowMap* shadowMap;       // may be null
        size_t shadowUniformSize = 0;   // runtime shadow uniform buffer size
        size_t pointShadowUniformSize = 0;
        // Optional per-draw storage buffers
        WGPUBuffer instanceBuffer = nullptr; size_t instanceSize = 0;
        WGPUBuffer morphBuffer = nullptr;    size_t morphSize = 0;
        WGPUBuffer skinBuffer = nullptr;     size_t skinSize = 0;
        WGPUBuffer skinVertexBuffer = nullptr; size_t skinVertexSize = 0;
        WGPUTextureView transmissionTexView = nullptr;
        WGPUSampler transmissionSampler = nullptr;
    };

    class WgpuBindGroups {
    public:
        WgpuBindGroups();

        // Build entries for a standard material draw.
        // Returns a const reference to the internal vector, valid until the next build*() call.
        const std::vector<WGPUBindGroupEntry>& buildStandard(const BindGroupInputs& inputs);

        // Build entries for a custom ShaderMaterial draw.
        const std::vector<WGPUBindGroupEntry>& buildCustom(
                WGPUBuffer transformBuffer, WGPUBuffer lightBuffer,
                size_t lightUniformSize,
                WGPUBuffer customUniformBuffer, uint32_t customUniformSize,
                ShaderMaterial* sm,
                const TextureList& textures);

    private:
        std::vector<WGPUBindGroupEntry> entries_;  // persistent, reused across calls
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUBINDGROUPS_HPP
