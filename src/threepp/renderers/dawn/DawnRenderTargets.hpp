// Render target texture cache for the Dawn renderer.
// Manages creation and lifecycle of color/depth textures for
// off-screen render targets, including MSAA variants.

#ifndef THREEPP_DAWNRENDERTARGETS_HPP
#define THREEPP_DAWNRENDERTARGETS_HPP

#include "DawnState.hpp"

#include <string>
#include <unordered_map>
#include <webgpu/webgpu.h>

namespace threepp {
    class RenderTarget;
}

namespace threepp::dawn {

    struct RTEntry {
        WGPUTexture colorTexture = nullptr;       // resolve target (1x) -- readback source
        WGPUTextureView colorView = nullptr;
        WGPUSampler colorSampler = nullptr;       // for sampling as a texture binding
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        // MSAA textures (only created when sampleCount > 1)
        WGPUTexture msaaColorTexture = nullptr;   // multi-sampled render target
        WGPUTextureView msaaColorView = nullptr;
        WGPUTexture msaaDepthTexture = nullptr;
        WGPUTextureView msaaDepthView = nullptr;
        unsigned int width = 0, height = 0;
        uint32_t sampleCount = 1;
    };

    class DawnRenderTargets {
    public:
        explicit DawnRenderTargets(DawnState& state);

        // Get or create textures for the given render target and sample count.
        RTEntry& getOrCreate(RenderTarget* rt, uint32_t sampleCount);

        // Find an RTEntry by the Texture::id of its color texture (for uniform-based binding).
        RTEntry* findByTextureId(unsigned int texId);

        // Invalidate and release all cached entries.
        void invalidateAll();

        void dispose();

    private:
        DawnState& state_;
        std::unordered_map<std::string, RTEntry> cache_;
        std::unordered_map<unsigned int, std::string> texToRtUuid_;

        static void releaseEntry(RTEntry& e);
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNRENDERTARGETS_HPP
