
#include "WgpuRenderTargets.hpp"

#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp::wgpu;

WgpuRenderTargets::WgpuRenderTargets(WgpuState& state)
    : state_(state) {}

void WgpuRenderTargets::releaseEntry(RTEntry& e) {
    if (e.msaaColorView) wgpuTextureViewRelease(e.msaaColorView);
    if (e.msaaColorTexture) wgpuTextureRelease(e.msaaColorTexture);
    if (e.msaaDepthView) wgpuTextureViewRelease(e.msaaDepthView);
    if (e.msaaDepthTexture) wgpuTextureRelease(e.msaaDepthTexture);
    if (e.colorSampler) wgpuSamplerRelease(e.colorSampler);
    if (e.colorView) wgpuTextureViewRelease(e.colorView);
    if (e.colorTexture) wgpuTextureRelease(e.colorTexture);
    if (e.depthResolveSampler) wgpuSamplerRelease(e.depthResolveSampler);
    if (e.depthResolveView) wgpuTextureViewRelease(e.depthResolveView);
    if (e.depthResolveTexture) wgpuTextureRelease(e.depthResolveTexture);
    if (e.depthView) wgpuTextureViewRelease(e.depthView);
    if (e.depthTexture) wgpuTextureRelease(e.depthTexture);
}

RTEntry& WgpuRenderTargets::getOrCreate(threepp::RenderTarget* rt, uint32_t sampleCount) {
    // A user-provided DepthTexture must be sampleable, so the depth attachment has
    // to be single-sampled (an MSAA depth target can't be read as texture_depth_2d).
    // Force 1x for these render targets, regardless of the renderer's MSAA setting.
    if (rt->depthTexture) sampleCount = 1;

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
    ctd.label = WGPUStringView{"rt_color", WGPU_STRLEN} ;
    ctd.size = {rt->width, rt->height, 1};
    ctd.mipLevelCount = 1;
    ctd.sampleCount = 1;
    ctd.dimension = WGPUTextureDimension_2D;
    ctd.format = state_.surfaceFormat;
    ctd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
    entry.colorTexture = wgpuDeviceCreateTexture(state_.device, &ctd);
    entry.colorView = wgpuTextureCreateView(entry.colorTexture, nullptr);

    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"rt_sampler", WGPU_STRLEN} ;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    entry.colorSampler = wgpuDeviceCreateSampler(state_.device, &sd);

    if (rt->texture) texToRtUuid_[rt->texture->id] = rt->uuid;

    WGPUTextureDescriptor dtd{};
    dtd.label = WGPUStringView{"rt_depth", WGPU_STRLEN} ;
    dtd.size = {rt->width, rt->height, 1};
    dtd.mipLevelCount = 1;
    dtd.sampleCount = sampleCount;
    dtd.dimension = WGPUTextureDimension_2D;
    dtd.format = WGPUTextureFormat_Depth24Plus;
    // When the RT carries a user DepthTexture, the depth attachment is read by the
    // resolve pass (textureLoad of texture_depth_2d) — so it needs TextureBinding.
    dtd.usage = WGPUTextureUsage_RenderAttachment
              | (rt->depthTexture ? WGPUTextureUsage_TextureBinding : 0u);
    entry.depthTexture = wgpuDeviceCreateTexture(state_.device, &dtd);
    entry.depthView = wgpuTextureCreateView(entry.depthTexture, nullptr);

    // User DepthTexture: allocate the R32Float resolve target + non-filtering sampler,
    // and register the mapping so the binding path can find it by Texture::id.
    if (rt->depthTexture) {
        WGPUTextureDescriptor rdtd{};
        rdtd.label = WGPUStringView{"rt_depth_resolve", WGPU_STRLEN};
        rdtd.size = {rt->width, rt->height, 1};
        rdtd.mipLevelCount = 1;
        rdtd.sampleCount = 1;
        rdtd.dimension = WGPUTextureDimension_2D;
        rdtd.format = WGPUTextureFormat_R32Float;
        rdtd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        entry.depthResolveTexture = wgpuDeviceCreateTexture(state_.device, &rdtd);
        entry.depthResolveView = wgpuTextureCreateView(entry.depthResolveTexture, nullptr);

        // Nearest, clamp, no compare, no anisotropy => a non-filtering sampler,
        // which is the only kind valid for an R32Float (unfilterable-float) texture.
        WGPUSamplerDescriptor rsd{};
        rsd.label = WGPUStringView{"rt_depth_sampler", WGPU_STRLEN};
        rsd.minFilter = WGPUFilterMode_Nearest;
        rsd.magFilter = WGPUFilterMode_Nearest;
        rsd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        rsd.addressModeU = WGPUAddressMode_ClampToEdge;
        rsd.addressModeV = WGPUAddressMode_ClampToEdge;
        rsd.addressModeW = WGPUAddressMode_ClampToEdge;
        rsd.lodMaxClamp = 1.0f;
        rsd.maxAnisotropy = 1;
        entry.depthResolveSampler = wgpuDeviceCreateSampler(state_.device, &rsd);

        entry.depthTexId = rt->depthTexture->id;
        depthTexToRtUuid_[rt->depthTexture->id] = rt->uuid;
    }

    if (sampleCount > 1) {
        WGPUTextureDescriptor msaaCtd{};
        msaaCtd.label = WGPUStringView{"rt_msaa_color", WGPU_STRLEN} ;
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

RTEntry* WgpuRenderTargets::findByTextureId(unsigned int texId) {
    auto it = texToRtUuid_.find(texId);
    if (it == texToRtUuid_.end()) return nullptr;
    auto it2 = cache_.find(it->second);
    return it2 != cache_.end() ? &it2->second : nullptr;
}

RTEntry* WgpuRenderTargets::findDepthByTextureId(unsigned int texId) {
    auto it = depthTexToRtUuid_.find(texId);
    if (it == depthTexToRtUuid_.end()) return nullptr;
    auto it2 = cache_.find(it->second);
    return it2 != cache_.end() ? &it2->second : nullptr;
}

void WgpuRenderTargets::invalidateAll() {
    for (auto& [id, entry] : cache_) {
        releaseEntry(entry);
    }
    cache_.clear();
    texToRtUuid_.clear();
    depthTexToRtUuid_.clear();
}

void WgpuRenderTargets::dispose() {
    invalidateAll();
}
