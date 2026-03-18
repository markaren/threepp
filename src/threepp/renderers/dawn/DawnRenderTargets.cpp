
#include "DawnRenderTargets.hpp"

#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp::dawn;

DawnRenderTargets::DawnRenderTargets(DawnState& state)
    : state_(state) {}

void DawnRenderTargets::releaseEntry(RTEntry& e) {
    if (e.msaaColorView) wgpuTextureViewRelease(e.msaaColorView);
    if (e.msaaColorTexture) wgpuTextureRelease(e.msaaColorTexture);
    if (e.msaaDepthView) wgpuTextureViewRelease(e.msaaDepthView);
    if (e.msaaDepthTexture) wgpuTextureRelease(e.msaaDepthTexture);
    if (e.colorSampler) wgpuSamplerRelease(e.colorSampler);
    if (e.colorView) wgpuTextureViewRelease(e.colorView);
    if (e.colorTexture) wgpuTextureRelease(e.colorTexture);
    if (e.depthView) wgpuTextureViewRelease(e.depthView);
    if (e.depthTexture) wgpuTextureRelease(e.depthTexture);
}

RTEntry& DawnRenderTargets::getOrCreate(threepp::RenderTarget* rt, uint32_t sampleCount) {
    auto it = cache_.find(rt->uuid);
    if (it != cache_.end() && it->second.width == rt->width
        && it->second.height == rt->height && it->second.sampleCount == sampleCount) {
        return it->second;
    }
    if (it != cache_.end()) {
        releaseEntry(it->second);
    }

    RTEntry entry{};
    entry.width = rt->width;
    entry.height = rt->height;
    entry.sampleCount = sampleCount;

    WGPUTextureDescriptor ctd{};
    ctd.label = {.data = "rt_color", .length = 8};
    ctd.size = {rt->width, rt->height, 1};
    ctd.mipLevelCount = 1;
    ctd.sampleCount = 1;
    ctd.dimension = WGPUTextureDimension_2D;
    ctd.format = state_.surfaceFormat;
    ctd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
    entry.colorTexture = wgpuDeviceCreateTexture(state_.device, &ctd);
    entry.colorView = wgpuTextureCreateView(entry.colorTexture, nullptr);

    WGPUSamplerDescriptor sd{};
    sd.label = {.data = "rt_sampler", .length = 10};
    sd.minFilter = WGPUFilterMode_Linear;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.maxAnisotropy = 1;
    entry.colorSampler = wgpuDeviceCreateSampler(state_.device, &sd);

    if (rt->texture) texToRtUuid_[rt->texture->id] = rt->uuid;

    WGPUTextureDescriptor dtd{};
    dtd.label = {.data = "rt_depth", .length = 8};
    dtd.size = {rt->width, rt->height, 1};
    dtd.mipLevelCount = 1;
    dtd.sampleCount = sampleCount;
    dtd.dimension = WGPUTextureDimension_2D;
    dtd.format = WGPUTextureFormat_Depth24Plus;
    dtd.usage = WGPUTextureUsage_RenderAttachment;
    entry.depthTexture = wgpuDeviceCreateTexture(state_.device, &dtd);
    entry.depthView = wgpuTextureCreateView(entry.depthTexture, nullptr);

    if (sampleCount > 1) {
        WGPUTextureDescriptor msaaCtd{};
        msaaCtd.label = {.data = "rt_msaa_color", .length = 13};
        msaaCtd.size = {rt->width, rt->height, 1};
        msaaCtd.mipLevelCount = 1;
        msaaCtd.sampleCount = sampleCount;
        msaaCtd.dimension = WGPUTextureDimension_2D;
        msaaCtd.format = state_.surfaceFormat;
        msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
        entry.msaaColorTexture = wgpuDeviceCreateTexture(state_.device, &msaaCtd);
        entry.msaaColorView = wgpuTextureCreateView(entry.msaaColorTexture, nullptr);
    }

    cache_[rt->uuid] = entry;
    return cache_[rt->uuid];
}

RTEntry* DawnRenderTargets::findByTextureId(unsigned int texId) {
    auto it = texToRtUuid_.find(texId);
    if (it == texToRtUuid_.end()) return nullptr;
    auto it2 = cache_.find(it->second);
    return it2 != cache_.end() ? &it2->second : nullptr;
}

void DawnRenderTargets::invalidateAll() {
    for (auto& [id, entry] : cache_) {
        releaseEntry(entry);
    }
    cache_.clear();
    texToRtUuid_.clear();
}

void DawnRenderTargets::dispose() {
    invalidateAll();
}
