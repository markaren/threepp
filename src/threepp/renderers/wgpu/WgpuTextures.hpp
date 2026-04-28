// Texture upload, caching, and version tracking for the Wgpu backend.

#ifndef THREEPP_WGPUTEXTURES_HPP
#define THREEPP_WGPUTEXTURES_HPP

#include "WgpuMipmapGenerator.hpp"
#include "WgpuPMREM.hpp"
#include "WgpuState.hpp"

#include <unordered_map>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class Texture;
}

namespace threepp::wgpu {

    struct TextureEntry {
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
        WGPUSampler sampler = nullptr;
        unsigned int version = 0;
    };

    class WgpuTextures {
    public:
        explicit WgpuTextures(WgpuState& state);

        void createDummyTexture();

        TextureEntry& getOrCreateTexture(Texture* tex);

        TextureEntry& getOrCreateCubeTexture(Texture* tex);

        // Same as getOrCreateTexture, but generates mipmaps via GGX-prefiltered
        // convolution (PMREM) instead of box filter. Use for environment maps
        // so rough materials sample properly blurred radiance at high roughness.
        TextureEntry& getOrCreateEnvTexture2D(Texture* tex);

        [[nodiscard]] const TextureEntry& getDummyTexture() const { return dummyTexture_; }
        [[nodiscard]] const TextureEntry& getDummyCubeTexture() const { return dummyCubeTexture_; }

        // RectAreaLight LTC (Linearly Transformed Cosines) lookup textures.
        // Built lazily on first call; both share a single sampler
        // (ltc_1.sampler). 64x64 RGBA float2 data embedded from three.js.
        [[nodiscard]] const TextureEntry& getOrCreateLtc1();
        [[nodiscard]] const TextureEntry& getOrCreateLtc2();

        [[nodiscard]] const TextureEntry* findTexture(unsigned int id) const;

        // Flush any pending mipmap generation work.
        // Must be called BEFORE the main render pass begins each frame so that
        // mipmap command buffers are submitted while no render pass is active.
        void flushPendingMipmaps();

        void dispose();

        [[nodiscard]] size_t count() const { return cache_.size(); }

        WgpuMipmapGenerator& mipmapGen() { return mipmapGen_; }

    private:
        struct PendingMipmap {
            WGPUTexture texture;
            uint32_t width, height, mipLevels;
            bool isCube;
            bool prefiltered;
            WGPUTextureFormat format;
        };
        WgpuState& state_;
        WgpuMipmapGenerator mipmapGen_;
        WgpuPMREM pmrem_;
        std::vector<PendingMipmap> pendingMipmaps_;
        std::unordered_map<unsigned int, TextureEntry> envCache2D_;
        std::unordered_map<unsigned int, TextureEntry> cache_;
        std::unordered_map<unsigned int, TextureEntry> cubeCache_;
        TextureEntry dummyTexture_;
        TextureEntry dummyCubeTexture_;
        TextureEntry ltc1_;
        TextureEntry ltc2_;
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUTEXTURES_HPP
