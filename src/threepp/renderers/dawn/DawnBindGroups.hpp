// Bind group assembly for the Dawn renderer.
// Builds WGPUBindGroupEntry arrays for standard and custom material draws.
// Uses a persistent internal vector to avoid per-draw heap allocations.

#ifndef THREEPP_DAWNBINDGROUPS_HPP
#define THREEPP_DAWNBINDGROUPS_HPP

#include "DawnMaterials.hpp"
#include "DawnShaders.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class ShaderMaterial;
    class Texture;
    class GPUTexture;
}

namespace threepp::dawn {

    class DawnTextures;
    class DawnShadowMap;

    // Input parameters for building standard material bind group entries.
    // All members are references/pointers -- no copies.
    struct BindGroupInputs {
        uint64_t features;
        WGPUBuffer transformBuffer;
        WGPUBuffer materialBuffer;
        WGPUBuffer lightBuffer;         // may be null
        size_t lightUniformSize;        // runtime light buffer size
        const MaterialParams& params;
        DawnTextures& textures;
        DawnShadowMap* shadowMap;       // may be null
        // Optional per-draw storage buffers
        WGPUBuffer instanceBuffer = nullptr; size_t instanceSize = 0;
        WGPUBuffer morphBuffer = nullptr;    size_t morphSize = 0;
        WGPUBuffer skinBuffer = nullptr;     size_t skinSize = 0;
        WGPUBuffer skinVertexBuffer = nullptr; size_t skinVertexSize = 0;
    };

    class DawnBindGroups {
    public:
        DawnBindGroups();

        // Build entries for a standard material draw.
        // Returns a const reference to the internal vector, valid until the next build*() call.
        const std::vector<WGPUBindGroupEntry>& buildStandard(const BindGroupInputs& inputs);

        // Build entries for a custom ShaderMaterial draw.
        const std::vector<WGPUBindGroupEntry>& buildCustom(
                WGPUBuffer transformBuffer, WGPUBuffer lightBuffer,
                size_t lightUniformSize,
                WGPUBuffer customUniformBuffer, ShaderMaterial* sm);

    private:
        std::vector<WGPUBindGroupEntry> entries_;  // persistent, reused across calls
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNBINDGROUPS_HPP
