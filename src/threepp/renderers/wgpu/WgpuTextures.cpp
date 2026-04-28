
#include "WgpuTextures.hpp"

#include "threepp/lights/ltc/ltc_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <variant>
#include <webgpu/webgpu.h>
#include "threepp/constants.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::wgpu;

namespace {
    uint16_t f32_to_f16(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint16_t sign = static_cast<uint16_t>((x >> 31) << 15);
        int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (exp <= 0) return sign;
        if (exp >= 31) return sign | 0x7C00u;
        return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant >> 13);
    }

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
    : state_(state), mipmapGen_(state), pmrem_(state) {}

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
    sd.lodMaxClamp = 32.0f;
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
        csd.lodMaxClamp = 32.0f;
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

    // Detect float (HDR) vs byte data
    bool isHdr = false;
    try { (void)img.data<float>(); isHdr = true; }
    catch (const std::bad_variant_access&) {}

    const bool needsMips = tex->generateMipmaps && filterUsesMips(tex->minFilter);
    const uint32_t mipLevels = needsMips ? calcMipLevels(w, h) : 1u;

    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"user_tex", WGPU_STRLEN};
    td.size = {w, h, 1};
    td.mipLevelCount = mipLevels;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
             | (needsMips ? WGPUTextureUsage_RenderAttachment : 0u);

    WGPUTexelCopyTextureInfo dst{};
    WGPUTexelCopyBufferLayout layout{};
    WGPUExtent3D extent = {w, h, 1};

    if (isHdr) {
        auto& data = img.data<float>();
        if (data.empty()) return dummyTexture_;
        td.format = WGPUTextureFormat_RGBA16Float;  // filterable; RGBA32Float requires extra device feature
        entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
        entry.view = wgpuTextureCreateView(entry.texture, nullptr);
        std::vector<uint16_t> f16(data.size());
        for (size_t i = 0; i < data.size(); ++i) f16[i] = f32_to_f16(data[i]);
        dst.texture = entry.texture;
        layout.bytesPerRow = w * 4 * sizeof(uint16_t);
        layout.rowsPerImage = h;
        wgpuQueueWriteTexture(state_.queue, &dst, f16.data(), f16.size() * sizeof(uint16_t), &layout, &extent);
    } else {
        auto& data = img.data<unsigned char>();
        if (data.empty()) return dummyTexture_; // render target or uninitialized
        // Hardware sRGB sampling: when a texture is tagged sRGB, allocate
        // RGBA8UnormSrgb so textureSample decodes to linear automatically.
        // Mirrors three.js r166 — color textures sample as linear.
        const bool isSrgb = (tex->colorSpace == ColorSpace::sRGB);
        td.format = isSrgb ? WGPUTextureFormat_RGBA8UnormSrgb : WGPUTextureFormat_RGBA8Unorm;
        entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
        entry.view = wgpuTextureCreateView(entry.texture, nullptr);

        // Convert RGB→RGBA if needed
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
        dst.texture = entry.texture;
        layout.bytesPerRow = w * 4;
        layout.rowsPerImage = h;
        wgpuQueueWriteTexture(state_.queue, &dst, srcData, srcSize, &layout, &extent);
    }

    if (needsMips) {
        WGPUTextureFormat fmt;
        if (isHdr) {
            fmt = WGPUTextureFormat_RGBA16Float;
        } else {
            fmt = (tex->colorSpace == ColorSpace::sRGB)
                ? WGPUTextureFormat_RGBA8UnormSrgb
                : WGPUTextureFormat_RGBA8Unorm;
        }
        pendingMipmaps_.push_back({entry.texture, w, h, mipLevels, false, false, fmt});
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
    sd.lodMaxClamp = 32.0f;
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

    // Hardware sRGB sampling for color cubes (env color cubes typically
    // sample-time decode), matches three.js r166 behavior.
    const bool isSrgb = (tex->colorSpace == ColorSpace::sRGB);
    const WGPUTextureFormat cubeFormat = isSrgb
        ? WGPUTextureFormat_RGBA8UnormSrgb
        : WGPUTextureFormat_RGBA8Unorm;

    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"cube_tex", WGPU_STRLEN} ;
    td.size = {w, h, 6};
    td.mipLevelCount = mipLevels;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = cubeFormat;
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
        pendingMipmaps_.push_back({entry.texture, w, h, mipLevels, true, false, cubeFormat});
    }

    WGPUTextureViewDescriptor vd{};
    vd.label = WGPUStringView{"cube_view", WGPU_STRLEN} ;
    vd.format = cubeFormat;
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
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cubeCache_[tex->id] = entry;
    return cubeCache_[tex->id];
}

void WgpuTextures::flushPendingMipmaps() {
    if (pendingMipmaps_.empty()) return;
    for (auto& pm : pendingMipmaps_) {
        if (pm.prefiltered && !pm.isCube) {
            // Need mip 0 populated first via box-filter down-chain — then overwrite mips 1..N-1
            // with GGX-convolved versions. Simpler: just run PMREM, which writes mips 1..N-1
            // directly from mip 0 (no mip-0 copy needed; mip 0 already has source data).
            pmrem_.prefilter2D(pm.texture, pm.width, pm.height, pm.mipLevels, pm.format);
        } else if (pm.isCube) {
            mipmapGen_.generateCube(pm.texture, pm.width, pm.mipLevels, pm.format);
        } else {
            mipmapGen_.generate2D(pm.texture, pm.width, pm.height, pm.mipLevels, pm.format);
        }
    }
    pendingMipmaps_.clear();
}

TextureEntry& WgpuTextures::getOrCreateEnvTexture2D(Texture* tex) {
    auto it = envCache2D_.find(tex->id);
    if (it != envCache2D_.end() && it->second.version == tex->version()) {
        return it->second;
    }

    if (it != envCache2D_.end()) {
        if (it->second.view) wgpuTextureViewRelease(it->second.view);
        if (it->second.texture) wgpuTextureRelease(it->second.texture);
        if (it->second.sampler) wgpuSamplerRelease(it->second.sampler);
    }

    TextureEntry entry{};
    auto& img = tex->image();
    auto w = img.width;
    auto h = img.height;
    if (w == 0 || h == 0) return dummyTexture_;

    bool isHdr = false;
    try { (void)img.data<float>(); isHdr = true; }
    catch (const std::bad_variant_access&) {}

    // Always generate a full mip chain for env maps so roughness-indexed sampling works.
    const uint32_t mipLevels = calcMipLevels(w, h);

    WGPUTextureDescriptor td{};
    td.label = WGPUStringView{"env_tex", WGPU_STRLEN};
    td.size = {w, h, 1};
    td.mipLevelCount = mipLevels;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment;

    WGPUTexelCopyTextureInfo dst{};
    WGPUTexelCopyBufferLayout layout{};
    WGPUExtent3D extent = {w, h, 1};

    if (isHdr) {
        auto& data = img.data<float>();
        if (data.empty()) return dummyTexture_;
        td.format = WGPUTextureFormat_RGBA16Float;
        entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
        std::vector<uint16_t> f16(data.size());
        for (size_t i = 0; i < data.size(); ++i) f16[i] = f32_to_f16(data[i]);
        dst.texture = entry.texture;
        layout.bytesPerRow = w * 4 * sizeof(uint16_t);
        layout.rowsPerImage = h;
        wgpuQueueWriteTexture(state_.queue, &dst, f16.data(), f16.size() * sizeof(uint16_t), &layout, &extent);
    } else {
        auto& data = img.data<unsigned char>();
        if (data.empty()) return dummyTexture_;
        td.format = WGPUTextureFormat_RGBA8Unorm;
        entry.texture = wgpuDeviceCreateTexture(state_.device, &td);

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
        dst.texture = entry.texture;
        layout.bytesPerRow = w * 4;
        layout.rowsPerImage = h;
        wgpuQueueWriteTexture(state_.queue, &dst, srcData, srcSize, &layout, &extent);
    }

    const auto fmt = isHdr ? WGPUTextureFormat_RGBA16Float : WGPUTextureFormat_RGBA8Unorm;

    // Create an explicit full-mip-chain view. Passing nullptr gives a default
    // view whose mipLevelCount behaviour varies by backend/wgpu-native version —
    // some return only 1 to the shader's textureNumLevels(), which defeats the
    // roughness → mip mapping used for PMREM sampling.
    WGPUTextureViewDescriptor vd{};
    vd.label = WGPUStringView{"env_tex_view", WGPU_STRLEN};
    vd.format = fmt;
    vd.dimension = WGPUTextureViewDimension_2D;
    vd.baseMipLevel = 0;
    vd.mipLevelCount = mipLevels;
    vd.baseArrayLayer = 0;
    vd.arrayLayerCount = 1;
    vd.aspect = WGPUTextureAspect_All;
    entry.view = wgpuTextureCreateView(entry.texture, &vd);

    pendingMipmaps_.push_back({entry.texture, w, h, mipLevels, false, true, fmt});

    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"env_sampler", WGPU_STRLEN};
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_Repeat;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    envCache2D_[tex->id] = entry;
    return envCache2D_[tex->id];
}

const TextureEntry* WgpuTextures::findTexture(unsigned int id) const {
    auto it = cache_.find(id);
    return it != cache_.end() ? &it->second : nullptr;
}

namespace {
    // Upload a 64x64 RGBA16Float LTC LUT. RGBA16Float supports linear filtering
    // by default in WebGPU core; RGBA32Float requires the float32-filterable feature.
    void buildLtcTexture(WGPUDevice device, WGPUQueue queue,
                         const std::array<float, threepp::ltc::LUT_ELEMENTS>& data,
                         const char* label,
                         WGPUTexture& outTex, WGPUTextureView& outView) {
        std::vector<uint16_t> half(data.size());
        for (size_t i = 0; i < data.size(); ++i) half[i] = f32_to_f16(data[i]);

        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{label, WGPU_STRLEN} ;
        td.size = {threepp::ltc::LUT_SIZE, threepp::ltc::LUT_SIZE, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_RGBA16Float;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        outTex = wgpuDeviceCreateTexture(device, &td);
        outView = wgpuTextureCreateView(outTex, nullptr);

        WGPUTexelCopyTextureInfo dst{};
        dst.texture = outTex;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = threepp::ltc::LUT_SIZE * 4 * sizeof(uint16_t);
        layout.rowsPerImage = threepp::ltc::LUT_SIZE;
        WGPUExtent3D extent = {threepp::ltc::LUT_SIZE, threepp::ltc::LUT_SIZE, 1};
        wgpuQueueWriteTexture(queue, &dst, half.data(),
                              half.size() * sizeof(uint16_t),
                              &layout, &extent);
    }
}

const TextureEntry& WgpuTextures::getOrCreateLtc1() {
    if (!ltc1_.texture) {
        buildLtcTexture(state_.device, state_.queue, threepp::ltc::LTC_MAT_1,
                        "ltc_1", ltc1_.texture, ltc1_.view);
        WGPUSamplerDescriptor sd{};
        sd.label = WGPUStringView{"ltc_sampler", WGPU_STRLEN} ;
        sd.magFilter = WGPUFilterMode_Linear;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.lodMaxClamp = 0.0f;
        sd.maxAnisotropy = 1;
        ltc1_.sampler = wgpuDeviceCreateSampler(state_.device, &sd);
    }
    return ltc1_;
}

const TextureEntry& WgpuTextures::getOrCreateLtc2() {
    if (!ltc2_.texture) {
        buildLtcTexture(state_.device, state_.queue, threepp::ltc::LTC_MAT_2,
                        "ltc_2", ltc2_.texture, ltc2_.view);
    }
    return ltc2_;
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

    for (auto& [id, te] : envCache2D_) {
        if (te.view) wgpuTextureViewRelease(te.view);
        if (te.texture) wgpuTextureRelease(te.texture);
        if (te.sampler) wgpuSamplerRelease(te.sampler);
    }
    envCache2D_.clear();

    if (dummyTexture_.view) wgpuTextureViewRelease(dummyTexture_.view);
    if (dummyTexture_.texture) wgpuTextureRelease(dummyTexture_.texture);
    if (dummyTexture_.sampler) wgpuSamplerRelease(dummyTexture_.sampler);
    dummyTexture_ = {};

    if (dummyCubeTexture_.view) wgpuTextureViewRelease(dummyCubeTexture_.view);
    if (dummyCubeTexture_.texture) wgpuTextureRelease(dummyCubeTexture_.texture);
    if (dummyCubeTexture_.sampler) wgpuSamplerRelease(dummyCubeTexture_.sampler);
    dummyCubeTexture_ = {};

    auto releaseLtc = [](TextureEntry& t) {
        if (t.view) wgpuTextureViewRelease(t.view);
        if (t.texture) wgpuTextureRelease(t.texture);
        if (t.sampler) wgpuSamplerRelease(t.sampler);
        t = {};
    };
    releaseLtc(ltc1_);
    releaseLtc(ltc2_);
}
