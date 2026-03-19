
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <webgpu/webgpu.h>

#include <cstring>
#include <vector>

using namespace threepp;

namespace {

    WGPUTextureFormat toWGPUFormat(WgpuTexture::Format fmt) {
        switch (fmt) {
            case WgpuTexture::Format::RGBA32Float: return WGPUTextureFormat_RGBA32Float;
            case WgpuTexture::Format::RG32Float:   return WGPUTextureFormat_RG32Float;
            case WgpuTexture::Format::RGBA8Unorm:  return WGPUTextureFormat_RGBA8Unorm;
        }
        return WGPUTextureFormat_RGBA8Unorm;
    }

    uint32_t bytesPerPixel(WgpuTexture::Format fmt) {
        switch (fmt) {
            case WgpuTexture::Format::RGBA32Float: return 16;
            case WgpuTexture::Format::RG32Float:   return 8;
            case WgpuTexture::Format::RGBA8Unorm:  return 4;
        }
        return 4;
    }

    uint32_t toWGPUUsage(uint32_t usage) {
        uint32_t flags = 0;
        if (usage & WgpuTexture::Storage)        flags |= WGPUTextureUsage_StorageBinding;
        if (usage & WgpuTexture::TextureBinding)  flags |= WGPUTextureUsage_TextureBinding;
        if (usage & WgpuTexture::CopyDst)         flags |= WGPUTextureUsage_CopyDst;
        if (usage & WgpuTexture::CopySrc)         flags |= WGPUTextureUsage_CopySrc;
        return flags;
    }

}// namespace

WgpuTexture::WgpuTexture(WgpuRenderer& renderer, uint32_t width, uint32_t height,
                       Format format, uint32_t usage)
    : device_(static_cast<WGPUDevice>(renderer.nativeDevice())),
      queue_(static_cast<WGPUQueue>(renderer.nativeQueue())),
      width_(width), height_(height), format_(format),
      dimension_(Dimension::D2),
      bytesPerPixel_(::bytesPerPixel(format)) {

    WGPUTextureDescriptor texDesc{};
    texDesc.label = {.data = "gpu_texture", .length = 11};
    texDesc.size = {width, height, 1};
    texDesc.mipLevelCount = 1;
    texDesc.sampleCount = 1;
    texDesc.dimension = WGPUTextureDimension_2D;
    texDesc.format = toWGPUFormat(format);
    texDesc.usage = toWGPUUsage(usage);
    texture_ = wgpuDeviceCreateTexture(device_, &texDesc);

    WGPUTextureViewDescriptor viewDesc{};
    viewDesc.label = {.data = "gpu_tex_view", .length = 12};
    viewDesc.format = toWGPUFormat(format);
    viewDesc.dimension = WGPUTextureViewDimension_2D;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = 1;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = 1;
    viewDesc.aspect = WGPUTextureAspect_All;
    view_ = wgpuTextureCreateView(texture_, &viewDesc);

    WGPUSamplerDescriptor samplerDesc{};
    samplerDesc.label = {.data = "gpu_tex_sampler", .length = 15};
    samplerDesc.addressModeU = WGPUAddressMode_Repeat;
    samplerDesc.addressModeV = WGPUAddressMode_Repeat;
    samplerDesc.addressModeW = WGPUAddressMode_Repeat;
    samplerDesc.magFilter = WGPUFilterMode_Nearest;
    samplerDesc.minFilter = WGPUFilterMode_Nearest;
    samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    samplerDesc.maxAnisotropy = 1;
    sampler_ = wgpuDeviceCreateSampler(device_, &samplerDesc);
}

WgpuTexture::WgpuTexture(WgpuRenderer& renderer, uint32_t width, uint32_t height,
                       Format format, Dimension dimension, uint32_t usage)
    : device_(static_cast<WGPUDevice>(renderer.nativeDevice())),
      queue_(static_cast<WGPUQueue>(renderer.nativeQueue())),
      width_(width), height_(height), format_(format),
      dimension_(dimension),
      bytesPerPixel_(::bytesPerPixel(format)) {

    bool isCube = (dimension == Dimension::Cube);

    WGPUTextureDescriptor texDesc{};
    texDesc.label = {.data = "gpu_texture", .length = 11};
    texDesc.size = {width, height, isCube ? 6u : 1u};
    texDesc.mipLevelCount = 1;
    texDesc.sampleCount = 1;
    texDesc.dimension = WGPUTextureDimension_2D;
    texDesc.format = toWGPUFormat(format);
    texDesc.usage = toWGPUUsage(usage);
    texture_ = wgpuDeviceCreateTexture(device_, &texDesc);

    WGPUTextureViewDescriptor viewDesc{};
    viewDesc.label = {.data = "gpu_tex_view", .length = 12};
    viewDesc.format = toWGPUFormat(format);
    viewDesc.dimension = isCube ? WGPUTextureViewDimension_Cube : WGPUTextureViewDimension_2D;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = 1;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = isCube ? 6 : 1;
    viewDesc.aspect = WGPUTextureAspect_All;
    view_ = wgpuTextureCreateView(texture_, &viewDesc);

    WGPUSamplerDescriptor samplerDesc{};
    samplerDesc.label = {.data = "gpu_tex_sampler", .length = 15};
    samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDesc.magFilter = WGPUFilterMode_Linear;
    samplerDesc.minFilter = WGPUFilterMode_Linear;
    samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    samplerDesc.maxAnisotropy = 1;
    sampler_ = wgpuDeviceCreateSampler(device_, &samplerDesc);
}

WgpuTexture::~WgpuTexture() {
    release();
}

WgpuTexture::WgpuTexture(WgpuTexture&& other) noexcept
    : device_(other.device_), queue_(other.queue_),
      texture_(other.texture_), view_(other.view_), sampler_(other.sampler_),
      width_(other.width_), height_(other.height_), format_(other.format_),
      dimension_(other.dimension_), bytesPerPixel_(other.bytesPerPixel_) {
    other.texture_ = nullptr;
    other.view_ = nullptr;
    other.sampler_ = nullptr;
}

WgpuTexture& WgpuTexture::operator=(WgpuTexture&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        queue_ = other.queue_;
        texture_ = other.texture_;
        view_ = other.view_;
        sampler_ = other.sampler_;
        width_ = other.width_;
        height_ = other.height_;
        format_ = other.format_;
        dimension_ = other.dimension_;
        bytesPerPixel_ = other.bytesPerPixel_;
        other.texture_ = nullptr;
        other.view_ = nullptr;
        other.sampler_ = nullptr;
    }
    return *this;
}

void WgpuTexture::write(const void* data, size_t size) {
    WGPUTexelCopyTextureInfo dst{};
    dst.texture = texture_;
    dst.mipLevel = 0;
    dst.origin = {0, 0, 0};
    dst.aspect = WGPUTextureAspect_All;

    // WebGPU requires bytesPerRow to be a multiple of 256
    uint32_t unpaddedBytesPerRow = width_ * bytesPerPixel_;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;

    if (paddedBytesPerRow == unpaddedBytesPerRow) {
        // No padding needed -- upload directly
        WGPUTexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = unpaddedBytesPerRow;
        layout.rowsPerImage = height_;

        WGPUExtent3D extent = {width_, height_, 1};
        wgpuQueueWriteTexture(queue_, &dst, data, size, &layout, &extent);
    } else {
        // Need row padding -- copy data with padded rows
        size_t paddedSize = static_cast<size_t>(paddedBytesPerRow) * height_;
        std::vector<uint8_t> padded(paddedSize, 0);
        auto* src = static_cast<const uint8_t*>(data);
        for (uint32_t row = 0; row < height_; row++) {
            std::memcpy(padded.data() + row * paddedBytesPerRow,
                        src + row * unpaddedBytesPerRow,
                        unpaddedBytesPerRow);
        }

        WGPUTexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = paddedBytesPerRow;
        layout.rowsPerImage = height_;

        WGPUExtent3D extent = {width_, height_, 1};
        wgpuQueueWriteTexture(queue_, &dst, padded.data(), paddedSize, &layout, &extent);
    }
}

void WgpuTexture::writeFace(uint32_t face, const void* data, size_t size) {
    if (face >= 6) {
        throw std::runtime_error("WgpuTexture::writeFace: face index must be 0-5");
    }

    const auto* srcBytes = static_cast<const unsigned char*>(data);

    // Convert RGB to RGBA if needed (JPEG images are 3 bytes/pixel)
    std::vector<unsigned char> rgba;
    size_t expectedRGB = static_cast<size_t>(width_) * height_ * 3;
    size_t expectedRGBA = static_cast<size_t>(width_) * height_ * 4;
    if (size == expectedRGB && bytesPerPixel_ == 4) {
        rgba.resize(expectedRGBA);
        for (size_t i = 0; i < static_cast<size_t>(width_) * height_; i++) {
            rgba[i * 4 + 0] = srcBytes[i * 3 + 0];
            rgba[i * 4 + 1] = srcBytes[i * 3 + 1];
            rgba[i * 4 + 2] = srcBytes[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        srcBytes = rgba.data();
        size = expectedRGBA;
    }

    WGPUTexelCopyTextureInfo dst{};
    dst.texture = texture_;
    dst.mipLevel = 0;
    dst.origin = {0, 0, face};
    dst.aspect = WGPUTextureAspect_All;

    uint32_t unpaddedBytesPerRow = width_ * bytesPerPixel_;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;

    if (paddedBytesPerRow == unpaddedBytesPerRow) {
        WGPUTexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = unpaddedBytesPerRow;
        layout.rowsPerImage = height_;

        WGPUExtent3D extent = {width_, height_, 1};
        wgpuQueueWriteTexture(queue_, &dst, srcBytes, size, &layout, &extent);
    } else {
        size_t paddedSize = static_cast<size_t>(paddedBytesPerRow) * height_;
        std::vector<uint8_t> padded(paddedSize, 0);
        for (uint32_t row = 0; row < height_; row++) {
            std::memcpy(padded.data() + row * paddedBytesPerRow,
                        srcBytes + row * unpaddedBytesPerRow,
                        unpaddedBytesPerRow);
        }

        WGPUTexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = paddedBytesPerRow;
        layout.rowsPerImage = height_;

        WGPUExtent3D extent = {width_, height_, 1};
        wgpuQueueWriteTexture(queue_, &dst, padded.data(), paddedSize, &layout, &extent);
    }
}

void WgpuTexture::release() {
    if (sampler_) { wgpuSamplerRelease(sampler_); sampler_ = nullptr; }
    if (view_) { wgpuTextureViewRelease(view_); view_ = nullptr; }
    if (texture_) { wgpuTextureRelease(texture_); texture_ = nullptr; }
}
