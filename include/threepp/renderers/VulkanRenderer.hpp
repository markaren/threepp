// Vulkan renderer (path-tracer focus). Built against the Vulkan SDK and
// requires VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure for
// the path-tracing pipeline introduced in Phase 5 of the MVP plan.
//
// Phase 0: skeleton. Constructs a VkInstance with validation layers in
// debug builds, picks a physical device, but does not yet present anything.
//
// Co-exists with GLRenderer / WgpuRenderer; selected by the application
// when a Canvas is created with GraphicsAPI::Vulkan.

#ifndef THREEPP_VULKANRENDERER_HPP
#define THREEPP_VULKANRENDERER_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <cstdint>
#include <functional>
#include <memory>

namespace threepp {

    class VulkanRenderer : public Renderer {

    public:
        explicit VulkanRenderer(Canvas& canvas);

        ~VulkanRenderer() override;

        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        void setViewport(const Vector4& v) override;
        void setViewport(int x, int y, int width, int height) override;

        void setScissor(const Vector4& v) override;
        void setScissor(int x, int y, int width, int height) override;
        void setScissorTest(bool boolean) override;

        void setClearColor(const Color& color, float alpha = 1) override;
        void getClearColor(Color& target) const override;
        [[nodiscard]] float getClearAlpha() const override;
        void setClearAlpha(float alpha) override;

        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        [[nodiscard]] std::vector<unsigned char> readRGBPixels() override;

        void dispose() override;

        // ImGui integration handles. All Vulkan types are erased to void* /
        // uint32_t so callers don't need <vulkan/vulkan.h> in this header.
        [[nodiscard]] void* nativeInstance() const;
        [[nodiscard]] void* nativePhysicalDevice() const;
        [[nodiscard]] void* nativeDevice() const;
        [[nodiscard]] void* nativeGraphicsQueue() const;
        [[nodiscard]] uint32_t graphicsQueueFamily() const;
        [[nodiscard]] uint32_t nativeSwapchainFormat() const;// cast to VkFormat
        [[nodiscard]] uint32_t imageCount() const;            // swapchain image count

        // Callback invoked once per frame inside the present render pass,
        // after the path tracer has written the image. The argument is a
        // VkCommandBuffer (type-erased). Set null to disable.
        void setOverlayCallback(std::function<void(void*)> callback);

        // Henyey-Greenstein anisotropy parameter for the fog phase function.
        // g = 0 isotropic, >0 forward-scattering (sun god rays), <0 back-scatter.
        // Clamped to [-0.95, 0.95] internally. No effect when scene.fog is unset.
        // Mirrors WgpuPathTracer::setFogAnisotropy.
        void setFogAnisotropy(float g);
        [[nodiscard]] float getFogAnisotropy() const;

        // World-Y of the water surface for bounding underwater fog to the
        // water column. When the camera is below this Y, the fog distance
        // is capped at the ray's exit through the surface — prevents fog
        // from absorbing sky radiance on rays that leave the water.
        // Default 1e30 (no limit). Set each frame alongside scene.fog.
        void setFogWaterSurfaceY(float y);

        // Samples per pixel per frame for the path tracer. Default 1.
        // Each sample is an independent jittered primary ray; in-frame samples
        // are summed into the accumulator with weight `spp`, so per-pixel FC
        // also advances by `spp`. Clamped to >= 1.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

        // Path-trace render scale. raygen, denoise, and the hybrid raster
        // G-buffer run at (swapchain extent × scale); the temporal AA pass
        // then reconstructs full swapchain resolution by accumulating the
        // jittered low-res samples into a full-res history (TAA upsampling —
        // sharper than a plain blit, since it recovers detail across frames).
        // With TAA disabled the low-res result is instead bilinearly blitted
        // up. The wireframe overlay and ImGui always draw at full resolution.
        // Quadratic FPS lever: 0.5 quarters the path-trace pixel count.
        // Clamped to [0.25, 1.0]. Default 1.0 (no scaling — pixel-identical
        // to the unscaled path, zero added cost).
        //
        // Changing the scale reallocates all render-extent resources and
        // resets accumulation; it issues a vkDeviceWaitIdle internally and so
        // must not be called from inside a render() pass.
        void setRenderScale(float scale);
        [[nodiscard]] float renderScale() const;

        // Spatial denoiser (5×5 à-trous edge-aware filter) applied to the
        // temporally-accumulated radiance before tonemap + sRGB encode. Default
        // on. When off, the compute pass still runs but acts as a tonemap-only
        // pass-through (pixel-identical to the prior in-shader tonemap path).
        void setDenoise(bool enabled);
        [[nodiscard]] bool denoise() const;

        // Extra primary rays fired at detected silhouette pixels (mesh-ID,
        // depth-gradient, or diagonal neighbour mismatch in the raster
        // prepass). 0 disables silhouette MSAA. N gives (N+1)× total
        // samples at edge pixels — default 7 (8× MSAA). Capped internally
        // at 31. Edge pixels are typically 5–15 % of frame; cost scales
        // ~linearly with N on those pixels only.
        void setSilhouetteMsaaExtra(uint32_t extra);
        [[nodiscard]] uint32_t silhouetteMsaaExtra() const;

        // Per-NEE-sample firefly clamp: any direct-lighting contribution whose
        // luminance exceeds `cap` is rescaled down to `cap`. Default 30.0.
        // Pass 0 to disable (stored as a 1e30 sentinel — gates never fire).
        // Mirrors WgpuPathTracer::setFireflyClamp.
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // Manually reset path-tracer frame accumulation. Wipes per-pixel
        // running mean + history (gbuf + accum + ReSTIR DI reservoirs),
        // invalidates reproject state (prev camera + per-mesh prevWorld),
        // and rewinds the sampleIndex counter so the next frame cold-starts
        // from sample 1. Use after lens / focus / lighting changes the
        // renderer can't detect on its own. Issues a vkDeviceWaitIdle
        // internally — must not be called from inside a render() pass.
        // Mirrors WgpuPathTracer::resetAccumulation.
        void resetAccumulation();

        // Hybrid raster + path tracer mode (UE/Omniverse-style). Raster runs
        // a deterministic G-buffer prepass (depth, normal, motion vectors,
        // per-pixel IDs); the path tracer reads that as primary visibility
        // and starts at bounce 1. Eliminates moving-object shake from PT
        // primary jitter; AA happens via TAA on the raster side. Default on.
        void setHybridEnabled(bool enabled);
        [[nodiscard]] bool hybridEnabled() const;

        void setPerSppJitterHybrid(bool enabled);
        [[nodiscard]] bool perSppJitterHybrid() const;

        void setTaaEnabled(bool enabled);
        [[nodiscard]] bool taaEnabled() const;

        // ReSTIR DI master toggle. When on, the path tracer uses
        // streaming RIS + temporal + spatial reuse at primary surfaces — one
        // shadow ray to a single chosen sample replaces the per-light NEE
        // loop, with reservoirs reused across frames and neighbours. When
        // off (default), falls back to classic per-light NEE at primary (one
        // shadow ray per analytic light + one per emissive sample). Bounces
        // and env NEE are unaffected either way.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // ReSTIR GI master toggle (Stage 1a). When on, primary chit launches
        // a BSDF-sampled indirect sub-ray and assembles a single-sample
        // reservoir for the bounce-1 contribution; raygen's step-1 continues
        // from the sub-trace's hit (xs) at bounce 2 to avoid double-counting.
        // At M=1 the result is statistically equivalent to classic MC; the
        // visible variance reduction lands with Stage 1b/1c (temporal +
        // spatial reuse, separate commits). Default off.
        void setRestirGIEnabled(bool enabled);
        [[nodiscard]] bool restirGIEnabled() const;

        // Hybrid-mode raster overlay: post-TAA, draws wireframe-flagged
        // meshes (any material with `wireframe = true`) and Line/LineSegments
        // objects on top of the path-traced image, depth-tested against the
        // raster G-buffer's depth attachment so overlays are correctly
        // occluded by path-traced geometry. Mirrors the WGPU PT overlay path
        // (WgpuPathTracer.cpp:3748).
        //
        // Auto-detected: any Mesh with `material.wireframe == true`,
        // any Line/LineSegments, plus everything on the configured layer.
        // Pass `-1` (default) to disable layer-based selection — material
        // wireframe + Line objects are still drawn. Pass `0..31` to also
        // include any object on that layer (`obj.layers.enable(channel)`).
        // Has no effect when hybridEnabled is false.
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        // Day-1 / debug visualization: blit one G-buffer channel directly
        // to the swapchain, bypassing the path tracer.
        //   0 = off (PT consumes the G-buffer normally)
        //   1 = world-space normal
        //   2 = motion vector (NDC delta in red/green)
        //   3 = per-pixel instanceCustomIndex (raw uint16)
        // Has no effect when hybridEnabled is false.
        void setHybridDebugView(int view);
        [[nodiscard]] int hybridDebugView() const;

        // Per-frame timings (milliseconds). CPU values come from
        // std::chrono around the host-side hot path of the most recent
        // render() call. GPU values come from VkQueryPool timestamps and
        // lag by one in-flight slot (the most recently *retired* frame —
        // GPU is necessarily one or two frames behind the host). For a
        // steady-state renderer the difference is invisible; for a HUD
        // overlay it's fine to read both as "current frame".
        //
        // First frame after construction returns all zeros. GPU fields
        // are zero on frames where the corresponding pass didn't run
        // (e.g. photonEmitMs == 0 when no glass is visible).
        struct FrameTimings {
            float photonEmitMs   = 0.f;// caustic photon trace (when visible)
            float pathTraceMs    = 0.f;// main RT megakernel
            float denoiseMs      = 0.f;// à-trous passes + finalize tonemap
            float taaMs          = 0.f;// hybrid TAA resolve compute
            float rasterGbufMs   = 0.f;// hybrid G-buffer prepass
            float overlayMs      = 0.f;// hybrid overlay depth + draw
            float cpuEnsureSceneMs = 0.f;// ensureSceneBuilt
            float cpuRecordMs      = 0.f;// recordCommandBuffer
            float cpuFrameMs       = 0.f;// total render() wall time
        };
        [[nodiscard]] FrameTimings lastFrameTimings() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
