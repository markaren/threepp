#pragma once

#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include <webgpu/webgpu.h>

#include <chrono>
#include <cmath>
#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif

namespace webtide {

/// Per-frame GPU readback of wave state at a single world position.
///
/// On native (wgpu-native), uses synchronous map with wgpuDevicePoll.
/// On Emscripten, uses a one-frame-latency async pattern: submit the copy
/// and map request this frame, read the results next frame.
class BuoyantSampler {
public:
    explicit BuoyantSampler(threepp::WgpuRenderer& renderer)
        : device_(static_cast<WGPUDevice>(renderer.nativeDevice()))
        , queue_ (static_cast<WGPUQueue> (renderer.nativeQueue()))
    {
        WGPUBufferDescriptor bd{};
        bd.label = WGPUStringView{"buoy_staging", sizeof("buoy_staging") - 1};
        bd.size  = kNumSlots * kSlotBytes;
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
        // -----------------------------------------------------------
        // Step 1: Read back previous frame's result (if ready)
        // -----------------------------------------------------------
        Result r = lastResult_; // default to previous result for smoothness


        // Native: synchronous readback
        {
            struct MapState { bool done = false; WGPUMapAsyncStatus status{}; } s;
            submitCopy(h0, h1, h2, disp0, disp1, disp2, grad1, grad2,
                       worldX, worldZ, c0Tile, c1Tile, c2Tile);

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
#ifdef __EMSCRIPTEN__
                emscripten_sleep(1);
#else
                wgpuDevicePoll(device_, true, nullptr);
#endif
            }

            if (s.done && s.status == WGPUMapAsyncStatus_Success) {
                r = decodeResult(c0Tile, c1Tile, c2Tile, choppiness, waveScale);
            }
            wgpuBufferUnmap(staging_);
        }

        lastResult_ = r;


        return r;
    }

private:
    void submitCopy(
        const threepp::WgpuTexture& h0,
        const threepp::WgpuTexture& h1,
        const threepp::WgpuTexture& h2,
        const threepp::WgpuTexture& disp0,
        const threepp::WgpuTexture& disp1,
        const threepp::WgpuTexture& disp2,
        const threepp::WgpuTexture& grad1,
        const threepp::WgpuTexture& grad2,
        float worldX, float worldZ,
        float c0Tile, float c1Tile, float c2Tile)
    {
        const uint32_t sz = h0.width();

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
        ed.label = WGPUStringView{"buoy_enc", sizeof("buoy_enc") - 1};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, &ed);

        auto copyTexel = [&](WGPUTexture tex, uint32_t tx, uint32_t ty, uint32_t slot) {
            WGPUTexelCopyTextureInfo src{};
            src.texture = tex;
            src.origin  = {tx, ty, 0};

            WGPUTexelCopyBufferInfo dst{};
            dst.buffer              = staging_;
            dst.layout.offset       = static_cast<uint64_t>(slot) * kSlotBytes;
            dst.layout.bytesPerRow  = kSlotBytes;
            dst.layout.rowsPerImage = 1;

            WGPUExtent3D ext = {1, 1, 1};
            wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ext);
        };

        copyTexel(h0.texture(),    tx0, ty0, 0);
        copyTexel(h1.texture(),    tx1, ty1, 1);
        copyTexel(h2.texture(),    tx2, ty2, 2);
        copyTexel(disp0.texture(), tx0, ty0, 3);
        copyTexel(disp1.texture(), tx1, ty1, 4);
        copyTexel(disp2.texture(), tx2, ty2, 5);
        copyTexel(grad1.texture(), tx1, ty1, 6);
        copyTexel(grad2.texture(), tx2, ty2, 7);

        WGPUCommandBufferDescriptor cd{};
        cd.label = WGPUStringView{"buoy_cmd", sizeof("buoy_cmd") - 1};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
        wgpuQueueSubmit(queue_, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    Result decodeResult(float c0Tile, float c1Tile, float c2Tile,
                        float choppiness, float waveScale) {
        const auto* f = static_cast<const float*>(
            wgpuBufferGetConstMappedRange(staging_, 0, kNumSlots * kSlotBytes));
        if (!f) return {};

        constexpr uint32_t S = kSlotBytes / sizeof(float); // 64

        const float raw_h0   = f[0*S];
        const float raw_h1   = f[1*S];
        const float raw_h2   = f[2*S];
        const float raw_d0x  = f[3*S],   raw_d0z  = f[3*S+1];
        const float raw_d1x  = f[4*S],   raw_d1z  = f[4*S+1];
        const float raw_d2x  = f[5*S],   raw_d2z  = f[5*S+1];
        const float raw_g1x  = f[6*S],   raw_g1z  = f[6*S+1];
        const float raw_g2x  = f[7*S],   raw_g2z  = f[7*S+1];

        Result r{};
        const float h0n = raw_h0 * (0.25f / c0Tile);
        const float h1n = raw_h1 * (1.0f  / c1Tile);
        const float h2n = raw_h2 * (1.0f  / c2Tile);
        r.y = (h0n + h1n + h2n) * waveScale;

        const float dx = raw_d0x * (0.25f / c0Tile)
                       + raw_d1x * (1.0f  / c1Tile)
                       + raw_d2x * (1.0f  / c2Tile);
        const float dz = raw_d0z * (0.25f / c0Tile)
                       + raw_d1z * (1.0f  / c1Tile)
                       + raw_d2z * (1.0f  / c2Tile);
        r.dx = dx * choppiness;
        r.dz = dz * choppiness;

        r.gx = raw_g1x * (1.0f / c1Tile) + raw_g2x * (0.25f / c2Tile);
        r.gz = raw_g1z * (1.0f / c1Tile) + raw_g2z * (0.25f / c2Tile);
        return r;
    }

    WGPUDevice device_  = nullptr;
    WGPUQueue  queue_   = nullptr;
    WGPUBuffer staging_ = nullptr;

    // Async state for Emscripten (one-frame-latency readback)
    bool mapPending_ = false;
    bool mapDone_ = false;
    WGPUMapAsyncStatus mapStatus_{};
    float pendingC0Tile_ = 0, pendingC1Tile_ = 0, pendingC2Tile_ = 0;
    float pendingChoppiness_ = 0, pendingWaveScale_ = 0;
    Result lastResult_{};

    static constexpr uint32_t kSlotBytes = 256; // WebGPU min bytesPerRow
    static constexpr uint32_t kNumSlots  = 8;   // h0 h1 h2 | d0 d1 d2 | g1 g2
};

} // namespace webtide
