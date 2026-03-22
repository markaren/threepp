
#include "WgpuReadback.hpp"
#include "WgpuCompat.hpp"

#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>
#else
#include <emscripten.h>
#endif

#include <chrono>
#include <stdexcept>

namespace {
    constexpr auto WGPU_ASYNC_TIMEOUT = std::chrono::seconds(10);
}

std::vector<unsigned char> threepp::wgpu::readRGBPixels(
        WGPUDevice device, WGPUQueue queue,
        WGPUTexture colorTexture,
        uint32_t w, uint32_t h) {

    // Row alignment: WebGPU requires bytesPerRow to be a multiple of 256
    uint32_t bytesPerPixel = 4; // BGRA8
    uint32_t unpaddedBytesPerRow = w * bytesPerPixel;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;
    uint32_t bufferSize = paddedBytesPerRow * h;

    WGPUBufferDescriptor bd{};
    bd.label = WGPU_LABEL("readback_buf");
    bd.size = bufferSize;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer stagingBuf = wgpuDeviceCreateBuffer(device, &bd);

    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = WGPU_LABEL("readback_enc");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    WgpuTexelCopyTextureInfo src{};
    src.texture = colorTexture;

    WgpuTexelCopyBufferInfo dst{};
    dst.buffer = stagingBuf;
    dst.layout.bytesPerRow = paddedBytesPerRow;
    dst.layout.rowsPerImage = h;

    WGPUExtent3D extent = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = WGPU_LABEL("readback_cmd");
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(queue, 1, &cmd);

    struct MapData { bool done = false; WgpuMapAsyncStatus status; } mapData;

#ifdef __EMSCRIPTEN__
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, bufferSize,
        [](WGPUBufferMapAsyncStatus status, void* ud) {
            auto* d = static_cast<MapData*>(ud);
            d->status = status;
            d->done = true;
        }, &mapData);
#else
    WGPUBufferMapCallbackInfo mapCb{};
    mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
    mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
        auto* d = static_cast<MapData*>(ud1);
        d->status = status;
        d->done = true;
    };
    mapCb.userdata1 = &mapData;
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, bufferSize, mapCb);
#endif

    auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
    while (!mapData.done) {
        if (std::chrono::steady_clock::now() > deadline) {
            wgpuBufferRelease(stagingBuf);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            throw std::runtime_error("WgpuRenderer: readRGBPixels buffer map timed out");
        }
#ifdef __EMSCRIPTEN__
        emscripten_sleep(0);
#else
        wgpuDevicePoll(device, true, nullptr);
#endif
    }

    std::vector<unsigned char> result;
    if (mapData.status == WGPU_MAP_ASYNC_STATUS_SUCCESS) {
        auto* mapped = static_cast<const unsigned char*>(wgpuBufferGetConstMappedRange(stagingBuf, 0, bufferSize));
        result.resize(w * h * 3);
        for (uint32_t row = 0; row < h; row++) {
            for (uint32_t col = 0; col < w; col++) {
                const auto* px = mapped + row * paddedBytesPerRow + col * 4;
                size_t outIdx = (row * w + col) * 3;
                // BGRA -> RGB
                result[outIdx + 0] = px[2];
                result[outIdx + 1] = px[1];
                result[outIdx + 2] = px[0];
            }
        }
        wgpuBufferUnmap(stagingBuf);
    }

    wgpuBufferRelease(stagingBuf);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return result;
}