// Texture upload, caching, and version tracking for the Dawn backend.

#ifndef THREEPP_DAWNTEXTURES_HPP
#define THREEPP_DAWNTEXTURES_HPP

#include "DawnState.hpp"

#include <unordered_map>
#include <webgpu/webgpu.h>

namespace threepp {
    class Texture;
}

namespace threepp::dawn {

    struct TextureEntry {
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
        WGPUSampler sampler = nullptr;
        unsigned int version = 0;
    };

    class DawnTextures {
    public:
        explicit DawnTextures(DawnState& state);

        void createDummyTexture();

        TextureEntry& getOrCreateTexture(Texture* tex);

        TextureEntry& getOrCreateCubeTexture(Texture* tex);

        [[nodiscard]] const TextureEntry& getDummyTexture() const { return dummyTexture_; }
        [[nodiscard]] const TextureEntry& getDummyCubeTexture() const { return dummyCubeTexture_; }

        void dispose();

        [[nodiscard]] size_t count() const { return cache_.size(); }

    private:
        DawnState& state_;
        std::unordered_map<unsigned int, TextureEntry> cache_;
        std::unordered_map<unsigned int, TextureEntry> cubeCache_;
        TextureEntry dummyTexture_;
        TextureEntry dummyCubeTexture_;
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNTEXTURES_HPP
