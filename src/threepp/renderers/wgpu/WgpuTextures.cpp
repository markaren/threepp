
#include "WgpuTextures.hpp"

#include <algorithm>
#include <cmath>
#include <webgpu/webgpu.h>
#include "threepp/constants.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::wgpu;

namespace {
    // Returns the number of mip levels for a texture of the given dimensions.
    uint32_t calcMipLevels(uint32_t w, uint32_t h) {
        return static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(w, h))))) + 1u;
    }

    // Whether the minFilter mode requires mipmaps.
    bool filterUsesMips(Filter f) {
        return f == Filter::NearestMipmapNearest ||
               f == Filter::NearestMipmapLinear  ||
               f == Filter::LinearMipmapNearest  ||
               f == Filter::LinearMipmapLinear;
    }

    WGPUMipmapFilterMode toMipmapFilter(Filter f) {
        switch (f) {
            case Filter::NearestMipmapNearest:
            case Filter::LinearMipmapNearest:
                return WGPUMipmapFilterMode_Nearest;
            case Filter::NearestMipmapLinear:
            case Filter::LinearMipmapLinear:
                return WGPUMipmapFilterMode_Linear;
            default:
                return WGPUMipmapFilterMode_Nearest;
        }
    }
}

WgpuTextures::WgpuTextures(WgpuState& state)
    : state_(state), mipmapGen_(state) {}

void WgpuTextures::createDummyTexture() {
    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"dummy_tex", WGPU_STRLEN} ;
    td.size = {1, 1, 1};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    dummyTexture_.texture = wgpuDeviceCreateTexture(state_.device, &td);
    dummyTexture_.view = wgpuTextureCreateView(dummyTexture_.texture, nullptr);

    unsigned char white[] = {255, 255, 255, 255};
    WGPUTexelCopyTextureInfo dst{};
    dst.texture = dummyTexture_.texture;
    WGPUTexelCopyBufferLayout layout{};
    layout.bytesPerRow = 4;
    layout.rowsPerImage = 1;
    WGPUExtent3D extent = {1, 1, 1};
    wgpuQueueWriteTexture(state_.queue, &dst, white, 4, &layout, &extent);

    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"dummy_sampler", WGPU_STRLEN} ;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_Repeat;
    sd.addressModeV = WGPUAddressMode_Repeat;
    sd.addressModeW = WGPUAddressMode_Repeat;
    sd.maxAnisotropy = 1;
    dummyTexture_.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    // Dummy cube texture (1x1 white per face)
    {
        WGPUTextureDescriptor ctd{};
        ctd.label = WGPUStringView{"dummy_cube", WGPU_STRLEN} ;
        ctd.size = {1, 1, 6};
        ctd.mipLevelCount = 1;
        ctd.sampleCount = 1;
        ctd.dimension = WGPUTextureDimension_2D;
        ctd.format = WGPUTextureFormat_RGBA8Unorm;
        ctd.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        dummyCubeTexture_.texture = wgpuDeviceCreateTexture(state_.device, &ctd);

        for (uint32_t face = 0; face < 6; face++) {
            WGPUTexelCopyTextureInfo cdst{};
            cdst.texture = dummyCubeTexture_.texture;
            cdst.origin = {0, 0, face};
            WGPUTexelCopyBufferLayout clayout{};
            clayout.bytesPerRow = 4;
            clayout.rowsPerImage = 1;
            WGPUExtent3D cextent = {1, 1, 1};
            wgpuQueueWriteTexture(state_.queue, &cdst, white, 4, &clayout, &cextent);
        }

        WGPUTextureViewDescriptor cvd{};
        cvd.label = WGPUStringView{"dummy_cube_view", WGPU_STRLEN} ;
        cvd.format = WGPUTextureFormat_RGBA8Unorm;
        cvd.dimension = WGPUTextureViewDimension_Cube;
        cvd.baseMipLevel = 0;
        cvd.mipLevelCount = 1;
        cvd.baseArrayLayer = 0;
        cvd.arrayLayerCount = 6;
        cvd.aspect = WGPUTextureAspect_All;
        dummyCubeTexture_.view = wgpuTextureCreateView(dummyCubeTexture_.texture, &cvd);

        WGPUSamplerDescriptor csd{};
        csd.label = WGPUStringView{"dummy_cube_samp", WGPU_STRLEN} ;
        csd.magFilter = WGPUFilterMode_Linear;
        csd.minFilter = WGPUFilterMode_Linear;
        csd.addressModeU = WGPUAddressMode_ClampToEdge;
        csd.addressModeV = WGPUAddressMode_ClampToEdge;
        csd.addressModeW = WGPUAddressMode_ClampToEdge;
        csd.maxAnisotropy = 1;
        dummyCubeTexture_.sampler = wgpuDeviceCreateSampler(state_.device, &csd);
    }
}

TextureEntry& WgpuTextures::getOrCreateTexture(Texture* tex) {
    auto it = cache_.find(tex->id);
    if (it != cache_.end() && it->second.version == tex->version()) {
        return it->second;
    }

    // Release old if exists
    if (it != cache_.end()) {
        if (it->second.view) wgpuTextureViewRelease(it->second.view);
        if (it->second.texture) wgpuTextureRelease(it->second.texture);
        if (it->second.sampler) wgpuSamplerRelease(it->second.sampler);
    }

    TextureEntry entry{};
    auto& img = tex->image();
    auto w = img.width;
    auto h = img.height;
    if (w == 0 || h == 0) return dummyTexture_;

    auto& data = img.data<unsigned char>();
    if (data.empty()) return dummyTexture_; // e.g. render target texture with no CPU-side data

    const bool needsMips = tex->generateMipmaps && filterUsesMips(tex->minFilter);
    const uint32_t mipLevels = needsMips ? calcMipLevels(w, h) : 1u;

    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"user_tex", WGPU_STRLEN} ;
    td.size = {w, h, 1};
    td.mipLevelCount = mipLevels;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
             | (needsMips ? WGPUTextureUsage_RenderAttachment : 0u);
    entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
    entry.view = wgpuTextureCreateView(entry.texture, nullptr);

    // Convert RGB to RGBA if needed
    std::vector<unsigned char> rgba;
    const unsigned char* srcData = data.data();
    size_t srcSize = data.size();
    if (data.size() == static_cast<size_t>(w) * h * 3) {
        rgba.resize(static_cast<size_t>(w) * h * 4);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
            rgba[i * 4 + 0] = data[i * 3 + 0];
            rgba[i * 4 + 1] = data[i * 3 + 1];
            rgba[i * 4 + 2] = data[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        srcData = rgba.data();
        srcSize = rgba.size();
    }

    WGPUTexelCopyTextureInfo dst{};
    dst.texture = entry.texture;
    WGPUTexelCopyBufferLayout layout{};
    layout.bytesPerRow = w * 4;
    layout.rowsPerImage = h;
    WGPUExtent3D extent = {w, h, 1};
    wgpuQueueWriteTexture(state_.queue, &dst, srcData, srcSize, &layout, &extent);

    if (needsMips) {
        pendingMipmaps_.push_back({entry.texture, w, h, mipLevels, false});
    }

    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"tex_sampler", WGPU_STRLEN} ;
    sd.magFilter = (tex->magFilter == Filter::Nearest) ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    sd.minFilter = (tex->minFilter == Filter::Nearest || tex->minFilter == Filter::NearestMipmapNearest || tex->minFilter == Filter::NearestMipmapLinear)
        ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    sd.mipmapFilter = needsMips ? toMipmapFilter(tex->minFilter) : WGPUMipmapFilterMode_Nearest;
    auto mapWrap = [](TextureWrapping w) {
        switch (w) {
            case TextureWrapping::Repeat: return WGPUAddressMode_Repeat;
            case TextureWrapping::MirroredRepeat: return WGPUAddressMode_MirrorRepeat;
            default: return WGPUAddressMode_ClampToEdge;
        }
    };
    sd.addressModeU = mapWrap(tex->wrapS);
    sd.addressModeV = mapWrap(tex->wrapT);
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    // Anisotropic filtering: requires linear min+mag and mipmaps to be meaningful.
    {
        auto aniso = static_cast<uint32_t>(tex->anisotropy);
        aniso = std::min(aniso, state_.maxAnisotropy);
        sd.maxAnisotropy = (aniso > 1 && sd.minFilter == WGPUFilterMode_Linear
                            && sd.magFilter == WGPUFilterMode_Linear) ? aniso : 1u;
    }
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cache_[tex->id] = entry;
    return cache_[tex->id];
}

TextureEntry& WgpuTextures::getOrCreateCubeTexture(Texture* tex) {
    auto it = cubeCache_.find(tex->id);
    if (it != cubeCache_.end() && it->second.version == tex->version()) {
        return it->second;
    }

    if (it != cubeCache_.end()) {
        if (it->second.view) wgpuTextureViewRelease(it->second.view);
        if (it->second.texture) wgpuTextureRelease(it->second.texture);
        if (it->second.sampler) wgpuSamplerRelease(it->second.sampler);
    }

    TextureEntry entry{};
    auto& images = tex->images();
    if (images.size() < 6) return dummyCubeTexture_;
    auto w = static_cast<uint32_t>(images[0].width);
    auto h = static_cast<uint32_t>(images[0].height);
    if (w == 0 || h == 0) return dummyCubeTexture_;

    const bool needsMips = tex->generateMipmaps && filterUsesMips(tex->minFilter);
    const uint32_t mipLevels = needsMips ? calcMipLevels(w, h) : 1u;

    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"cube_tex", WGPU_STRLEN} ;
    td.size = {w, h, 6};
    td.mipLevelCount = mipLevels;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
             | (needsMips ? WGPUTextureUsage_RenderAttachment : 0u);
    entry.texture = wgpuDeviceCreateTexture(state_.device, &td);

    // Upload each face
    for (uint32_t face = 0; face < 6; face++) {
        auto& faceImg = images[face];
        auto& data = faceImg.data<unsigned char>();

        // Convert RGB to RGBA if needed
        std::vector<unsigned char> rgba;
        const unsigned char* srcData = data.data();
        size_t srcSize = data.size();
        if (data.size() == w * h * 3) {
            rgba.resize(w * h * 4);
            for (size_t i = 0; i < w * h; i++) {
                rgba[i * 4 + 0] = data[i * 3 + 0];
                rgba[i * 4 + 1] = data[i * 3 + 1];
                rgba[i * 4 + 2] = data[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
            srcData = rgba.data();
            srcSize = rgba.size();
        }

        WGPUTexelCopyTextureInfo dst{};
        dst.texture = entry.texture;
        dst.origin = {0, 0, face};
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = w * 4;
        layout.rowsPerImage = h;
        WGPUExtent3D extent = {w, h, 1};
        wgpuQueueWriteTexture(state_.queue, &dst, srcData, srcSize, &layout, &extent);
    }

    if (needsMips) {
        pendingMipmaps_.push_back({entry.texture, w, h, mipLevels, true});
    }

    WGPUTextureViewDescriptor vd{};
    vd.label = WGPUStringView{"cube_view", WGPU_STRLEN} ;
    vd.format = WGPUTextureFormat_RGBA8Unorm;
    vd.dimension = WGPUTextureViewDimension_Cube;
    vd.baseMipLevel = 0;
    vd.mipLevelCount = mipLevels;
    vd.baseArrayLayer = 0;
    vd.arrayLayerCount = 6;
    vd.aspect = WGPUTextureAspect_All;
    entry.view = wgpuTextureCreateView(entry.texture, &vd);

    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"cube_sampler", WGPU_STRLEN} ;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = needsMips ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cubeCache_[tex->id] = entry;
    return cubeCache_[tex->id];
}

void WgpuTextures::flushPendingMipmaps() {
    if (pendingMipmaps_.empty()) return;
    for (auto& pm : pendingMipmaps_) {
        if (pm.isCube) {
            mipmapGen_.generateCube(pm.texture, pm.width, pm.mipLevels, WGPUTextureFormat_RGBA8Unorm);
        } else {
            mipmapGen_.generate2D(pm.texture, pm.width, pm.height, pm.mipLevels, WGPUTextureFormat_RGBA8Unorm);
        }
    }
    pendingMipmaps_.clear();
}

const TextureEntry* WgpuTextures::findTexture(unsigned int id) const {
    auto it = cache_.find(id);
    return it != cache_.end() ? &it->second : nullptr;
}

void WgpuTextures::dispose() {
    for (auto& [id, te] : cache_) {
        if (te.view) wgpuTextureViewRelease(te.view);
        if (te.texture) wgpuTextureRelease(te.texture);
        if (te.sampler) wgpuSamplerRelease(te.sampler);
    }
    cache_.clear();

    for (auto& [id, te] : cubeCache_) {
        if (te.view) wgpuTextureViewRelease(te.view);
        if (te.texture) wgpuTextureRelease(te.texture);
        if (te.sampler) wgpuSamplerRelease(te.sampler);
    }
    cubeCache_.clear();

    if (dummyTexture_.view) wgpuTextureViewRelease(dummyTexture_.view);
    if (dummyTexture_.texture) wgpuTextureRelease(dummyTexture_.texture);
    if (dummyTexture_.sampler) wgpuSamplerRelease(dummyTexture_.sampler);
    dummyTexture_ = {};

    if (dummyCubeTexture_.view) wgpuTextureViewRelease(dummyCubeTexture_.view);
    if (dummyCubeTexture_.texture) wgpuTextureRelease(dummyCubeTexture_.texture);
    if (dummyCubeTexture_.sampler) wgpuSamplerRelease(dummyCubeTexture_.sampler);
    dummyCubeTexture_ = {};
}
