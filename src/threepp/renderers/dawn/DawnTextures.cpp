
#include "DawnTextures.hpp"

#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::dawn;

DawnTextures::DawnTextures(DawnState& state)
    : state_(state) {}

void DawnTextures::createDummyTexture() {
    WGPUTextureDescriptor td{};
    td.label = {.data = "dummy_tex", .length = 9};
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
    sd.label = {.data = "dummy_sampler", .length = 13};
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
        ctd.label = {.data = "dummy_cube", .length = 10};
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
        cvd.label = {.data = "dummy_cube_view", .length = 15};
        cvd.format = WGPUTextureFormat_RGBA8Unorm;
        cvd.dimension = WGPUTextureViewDimension_Cube;
        cvd.baseMipLevel = 0;
        cvd.mipLevelCount = 1;
        cvd.baseArrayLayer = 0;
        cvd.arrayLayerCount = 6;
        cvd.aspect = WGPUTextureAspect_All;
        dummyCubeTexture_.view = wgpuTextureCreateView(dummyCubeTexture_.texture, &cvd);

        WGPUSamplerDescriptor csd{};
        csd.label = {.data = "dummy_cube_samp", .length = 15};
        csd.magFilter = WGPUFilterMode_Linear;
        csd.minFilter = WGPUFilterMode_Linear;
        csd.addressModeU = WGPUAddressMode_ClampToEdge;
        csd.addressModeV = WGPUAddressMode_ClampToEdge;
        csd.addressModeW = WGPUAddressMode_ClampToEdge;
        csd.maxAnisotropy = 1;
        dummyCubeTexture_.sampler = wgpuDeviceCreateSampler(state_.device, &csd);
    }
}

TextureEntry& DawnTextures::getOrCreateTexture(Texture* tex) {
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

    WGPUTextureDescriptor td{};
    td.label = {.data = "user_tex", .length = 8};
    td.size = {w, h, 1};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
    entry.view = wgpuTextureCreateView(entry.texture, nullptr);

    auto& data = img.data<unsigned char>();

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

    WGPUSamplerDescriptor sd{};
    sd.label = {.data = "tex_sampler", .length = 11};
    sd.magFilter = (tex->magFilter == Filter::Nearest) ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    sd.minFilter = (tex->minFilter == Filter::Nearest || tex->minFilter == Filter::NearestMipmapNearest || tex->minFilter == Filter::NearestMipmapLinear)
        ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
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
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cache_[tex->id] = entry;
    return cache_[tex->id];
}

TextureEntry& DawnTextures::getOrCreateCubeTexture(Texture* tex) {
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

    WGPUTextureDescriptor td{};
    td.label = {.data = "cube_tex", .length = 8};
    td.size = {w, h, 6};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
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

    WGPUTextureViewDescriptor vd{};
    vd.label = {.data = "cube_view", .length = 9};
    vd.format = WGPUTextureFormat_RGBA8Unorm;
    vd.dimension = WGPUTextureViewDimension_Cube;
    vd.baseMipLevel = 0;
    vd.mipLevelCount = 1;
    vd.baseArrayLayer = 0;
    vd.arrayLayerCount = 6;
    vd.aspect = WGPUTextureAspect_All;
    entry.view = wgpuTextureCreateView(entry.texture, &vd);

    WGPUSamplerDescriptor sd{};
    sd.label = {.data = "cube_sampler", .length = 12};
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cubeCache_[tex->id] = entry;
    return cubeCache_[tex->id];
}

void DawnTextures::dispose() {
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
