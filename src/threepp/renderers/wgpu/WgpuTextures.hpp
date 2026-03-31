// Texture upload, caching, and version tracking for the Wgpu backend.

#ifndef THREEPP_WGPUTEXTURES_HPP
#define THREEPP_WGPUTEXTURES_HPP

#include "WgpuMipmapGenerator.hpp"
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

        [[nodiscard]] const TextureEntry& getDummyTexture() const { return dummyTexture_; }
        [[nodiscard]] const TextureEntry& getDummyCubeTexture() const { return dummyCubeTexture_; }

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
        };
        WgpuState& state_;
        WgpuMipmapGenerator mipmapGen_;
        std::vector<PendingMipmap> pendingMipmaps_;
        std::unordered_map<unsigned int, TextureEntry> cache_;
        std::unordered_map<unsigned int, TextureEntry> cubeCache_;
        TextureEntry dummyTexture_;
        TextureEntry dummyCubeTexture_;
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUTEXTURES_HPP
