#pragma once

#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>
#else
#include <emscripten.h>
#endif

#include <chrono>
#include <cmath>

// Pull in the compat layer used by the rest of threepp's WGPU code
#ifdef __EMSCRIPTEN__
// Emscripten type aliases (mirrors WgpuCompat.hpp but self-contained for examples)
using WgpuTexelCopyTextureInfo  = WGPUImageCopyTexture;
using WgpuTexelCopyBufferInfo   = WGPUImageCopyBuffer;
using WgpuTexelCopyBufferLayout = WGPUTextureDataLayout;
using WgpuMapAsyncStatus        = WGPUBufferMapAsyncStatus;
#define BUOY_WGPU_LABEL(s) (s)
#define BUOY_MAP_ASYNC_STATUS_SUCCESS WGPUBufferMapAsyncStatus_Success
#else
using WgpuTexelCopyTextureInfo  = WGPUTexelCopyTextureInfo;
using WgpuTexelCopyBufferInfo   = WGPUTexelCopyBufferInfo;
using WgpuTexelCopyBufferLayout = WGPUTexelCopyBufferLayout;
using WgpuMapAsyncStatus        = WGPUMapAsyncStatus;
#define BUOY_WGPU_LABEL(s) WGPUStringView{.data = (s), .length = sizeof(s) - 1}
#define BUOY_MAP_ASYNC_STATUS_SUCCESS WGPUMapAsyncStatus_Success
#endif

namespace webtide {

/// Per-frame GPU readback of wave state at a single world position.
///
/// Reads 8 single texels per frame (one 6-copy command buffer):
///   slot 0 : h0     cascade-0 height         (RG32Float, R=height)
///   slot 1 : h1     cascade-1 height
///   slot 2 : h2     cascade-2 height
///   slot 3 : disp0  cascade-0 displacement    (RG32Float, R=Δx  G=Δz)
///   slot 4 : disp1  cascade-1 displacement
///   slot 5 : disp2  cascade-2 displacement
///   slot 6 : grad1  cascade-1 surface slope   (RG32Float, R=dH/dx  G=dH/dz)
///   slot 7 : grad2  cascade-2 surface slope
///
/// The normalization applied to each channel reproduces waterVertexWGSL exactly,
/// so the buoy tracks the displaced wave surface rather than the undisplaced grid.
class BuoyantSampler {
public:
    explicit BuoyantSampler(threepp::WgpuRenderer& renderer)
        : device_(static_cast<WGPUDevice>(renderer.nativeDevice()))
        , queue_ (static_cast<WGPUQueue> (renderer.nativeQueue()))
    {
        WGPUBufferDescriptor bd{};
        bd.label = BUOY_WGPU_LABEL("buoy_staging");
        bd.size  = kNumSlots * kSlotBytes; // 8 × 256 = 2048 bytes
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        staging_ = wgpuDeviceCreateBuffer(device_, &bd);
    }

    ~BuoyantSampler() { if (staging_) wgpuBufferRelease(staging_); }

    BuoyantSampler(const BuoyantSampler&) = delete;
    BuoyantSampler& operator=(const BuoyantSampler&) = delete;

    struct Result {
        float y   = 0.f;  ///< wave-surface Y     = (h0+h1+h2) × waveScale
        float dx  = 0.f;  ///< total X displacement = (d0+d1+d2) × choppiness
        float dz  = 0.f;  ///< total Z displacement
        float gx  = 0.f;  ///< surface slope dH/dx (for roll)
        float gz  = 0.f;  ///< surface slope dH/dz (for pitch)
    };

    /// Call once per frame after all IFFT compute passes have been submitted.
    /// All six displacement/height/gradient textures must have the CopySrc usage flag.
    Result sample(
        const threepp::WgpuTexture& h0,
        const threepp::WgpuTexture& h1,
        const threepp::WgpuTexture& h2,
        const threepp::WgpuTexture& disp0,
        const threepp::WgpuTexture& disp1,
        const threepp::WgpuTexture& disp2,
        const threepp::WgpuTexture& grad1,
        const threepp::WgpuTexture& grad2,
        float worldX, float worldZ,
        float c0Tile, float c1Tile, float c2Tile,
        float choppiness, float waveScale)
    {
        const uint32_t sz = h0.width();

        // Match the vertex shader's UV → integer texel mapping exactly.
        auto texelOf = [&](float tile) -> std::pair<uint32_t, uint32_t> {
            float u = worldX / tile;
            float v = worldZ / tile;
            u -= std::floor(u);
            v -= std::floor(v);
            if (u < 0.f) u += 1.f;
            if (v < 0.f) v += 1.f;
            return { static_cast<uint32_t>(u * sz) & (sz - 1),
                     static_cast<uint32_t>(v * sz) & (sz - 1) };
        };

        auto [tx0, ty0] = texelOf(c0Tile);
        auto [tx1, ty1] = texelOf(c1Tile);
        auto [tx2, ty2] = texelOf(c2Tile);

        WGPUCommandEncoderDescriptor ed{};
        ed.label = BUOY_WGPU_LABEL("buoy_enc");
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, &ed);

        auto copyTexel = [&](WGPUTexture tex, uint32_t tx, uint32_t ty, uint32_t slot) {
            WgpuTexelCopyTextureInfo src{};
            src.texture = tex;
            src.origin  = {tx, ty, 0};

            WgpuTexelCopyBufferInfo dst{};
            dst.buffer              = staging_;
            dst.layout.offset       = static_cast<uint64_t>(slot) * kSlotBytes;
            dst.layout.bytesPerRow  = kSlotBytes; // ≥ 256, multiple of 256 ✓
            dst.layout.rowsPerImage = 1;

            WGPUExtent3D ext = {1, 1, 1};
            wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ext);
        };

        copyTexel(h0.texture(),    tx0, ty0, 0);
        copyTexel(h1.texture(),    tx1, ty1, 1);
        copyTexel(h2.texture(),    tx2, ty2, 2);
        copyTexel(disp0.texture(), tx0, ty0, 3);  // cascade-0 UV for disp0
        copyTexel(disp1.texture(), tx1, ty1, 4);
        copyTexel(disp2.texture(), tx2, ty2, 5);  // cascade-2 UV for disp2
        copyTexel(grad1.texture(), tx1, ty1, 6);
        copyTexel(grad2.texture(), tx2, ty2, 7);

        WGPUCommandBufferDescriptor cd{};
        cd.label = BUOY_WGPU_LABEL("buoy_cmd");
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
        wgpuQueueSubmit(queue_, 1, &cmd);

        struct MapState { bool done = false; WgpuMapAsyncStatus status{}; } s;

#ifdef __EMSCRIPTEN__
        wgpuBufferMapAsync(staging_, WGPUMapMode_Read, 0, kNumSlots * kSlotBytes,
            [](WGPUBufferMapAsyncStatus st, void* userdata) {
                auto* s   = static_cast<MapState*>(userdata);
                s->status = st;
                s->done   = true;
            }, &s);

        // Emscripten has no wgpuDevicePoll; use emscripten_sleep to yield to the browser
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (!s.done) {
            if (std::chrono::steady_clock::now() > deadline) break;
            emscripten_sleep(1);
        }
#else
        WGPUBufferMapCallbackInfo cb{};
        cb.mode     = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* ud1, void*) {
            auto* s   = static_cast<MapState*>(ud1);
            s->status = st;
            s->done   = true;
        };
        cb.userdata1 = &s;
        wgpuBufferMapAsync(staging_, WGPUMapMode_Read, 0, kNumSlots * kSlotBytes, cb);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (!s.done) {
            if (std::chrono::steady_clock::now() > deadline) break;
            wgpuDevicePoll(device_, true, nullptr);
        }
#endif

        Result r{};
        if (s.done && s.status == BUOY_MAP_ASYNC_STATUS_SUCCESS) {
            // Each 256-byte slot: [float R, float G, <248 bytes padding>]
            // stride = 256/4 = 64 floats between slot starts.
            const auto* f = static_cast<const float*>(
                wgpuBufferGetConstMappedRange(staging_, 0, kNumSlots * kSlotBytes));

            constexpr uint32_t S = kSlotBytes / sizeof(float); // 64

            const float raw_h0   = f[0*S];
            const float raw_h1   = f[1*S];
            const float raw_h2   = f[2*S];
            const float raw_d0x  = f[3*S],   raw_d0z  = f[3*S+1];
            const float raw_d1x  = f[4*S],   raw_d1z  = f[4*S+1];
            const float raw_d2x  = f[5*S],   raw_d2z  = f[5*S+1];
            const float raw_g1x  = f[6*S],   raw_g1z  = f[6*S+1];
            const float raw_g2x  = f[7*S],   raw_g2z  = f[7*S+1];

            // --- Height: identical to waterVertexWGSL (c0Fade=1 assumed near camera) ---
            const float h0n = raw_h0 * (0.25f / c0Tile);
            const float h1n = raw_h1 * (1.0f  / c1Tile);
            const float h2n = raw_h2 * (1.0f  / c2Tile);
            r.y = (h0n + h1n + h2n) * waveScale;

            // --- XZ displacement: sum all three cascades, matches vertex shader ---
            //   d0 *= c0Fade * 0.25 / C0  (c0Fade=1)
            //   d1 *= 1.0 / C1
            //   d2 *= 1.0 / C2
            //   totalDisp *= choppiness
            const float dx = raw_d0x * (0.25f / c0Tile)
                           + raw_d1x * (1.0f  / c1Tile)
                           + raw_d2x * (1.0f  / c2Tile);
            const float dz = raw_d0z * (0.25f / c0Tile)
                           + raw_d1z * (1.0f  / c1Tile)
                           + raw_d2z * (1.0f  / c2Tile);
            r.dx = dx * choppiness;
            r.dz = dz * choppiness;

            // --- Gradient (slope): C1 + C2, skip C0 to avoid micro-jitter ---
            r.gx = raw_g1x * (1.0f / c1Tile) + raw_g2x * (0.25f / c2Tile);
            r.gz = raw_g1z * (1.0f / c1Tile) + raw_g2z * (0.25f / c2Tile);

            wgpuBufferUnmap(staging_);
        }

        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        return r;
    }

private:
    WGPUDevice device_  = nullptr;
    WGPUQueue  queue_   = nullptr;
    WGPUBuffer staging_ = nullptr;

    static constexpr uint32_t kSlotBytes = 256; // WebGPU min bytesPerRow
    static constexpr uint32_t kNumSlots  = 8;   // h0 h1 h2 | d0 d1 d2 | g1 g2
};

} // namespace webtide
