// Vulkan reference path tracer.
//
// The ground-truth / photorealism sibling of VulkanRenderer. Shades every pixel
// with the RT megakernel (photon caustics + raygen + closest-hit) and an à-trous
// denoiser, with ReSTIR DI/GI and env-CDF importance sampling. Slower and noisier
// than the deferred VulkanRenderer, but physically the reference image.
//
// Shares all of its infrastructure (device/swapchain, acceleration structures,
// scene build, material/geometry buffers, the raster G-buffer prepass, and the
// bloom/TAA/overlay post-stack) with VulkanRenderer through VulkanRendererCore;
// the two differ only in how the scene is shaded.
//
// NOTE: the path-tracer-specific knobs (samples-per-pixel, max bounces, ReSTIR,
// firefly clamp, …) are being migrated onto this class as the peer split lands;
// until then they remain on VulkanRenderer.

#ifndef THREEPP_VULKANPATHTRACER_HPP
#define THREEPP_VULKANPATHTRACER_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/VulkanRendererCore.hpp"

#include <memory>

namespace threepp {

    class VulkanPathTracer : public VulkanRendererCore {

    public:
        explicit VulkanPathTracer(Canvas& canvas);

        ~VulkanPathTracer() override;

        VulkanPathTracer(const VulkanPathTracer&) = delete;
        VulkanPathTracer& operator=(const VulkanPathTracer&) = delete;

        // ── Path-tracer knobs ─────────────────────────────────────────────────

        // Samples per pixel per frame. Default 1. Clamped >= 1.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

        // Extra primary rays fired at detected silhouette pixels. 0 disables.
        void setSilhouetteMsaaExtra(uint32_t extra);
        [[nodiscard]] uint32_t silhouetteMsaaExtra() const;

        // Per-NEE-sample firefly clamp. Default 30.0; 0 disables (1e30 sentinel).
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // Max real scatter events per path. Default 4. Clamped to [1, 16].
        // Changing the value resets accumulation.
        void setMaxBounces(int bounces);
        [[nodiscard]] int maxBounces() const;

        // Manually reset path-tracer frame accumulation. Issues a vkDeviceWaitIdle
        // internally — not callable from inside a render() pass.
        void resetAccumulation();

        void setPerSppJitterHybrid(bool enabled);
        [[nodiscard]] bool perSppJitterHybrid() const;

        // ReSTIR DI master toggle (streaming RIS + temporal + spatial reuse at
        // primary surfaces). Off (default) falls back to per-light NEE.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // ReSTIR DI visibility reuse (Bitterli 2020 §5). Default on.
        void setRestirDIVisibilityReuse(bool enabled);
        [[nodiscard]] bool restirDIVisibilityReuse() const;

        // ReSTIR GI master toggle (Stage 1a). Default off.
        void setRestirGIEnabled(bool enabled);
        [[nodiscard]] bool restirGIEnabled() const;

        // NVIDIA Shader Execution Reordering opt-out. On by default where supported.
        void setSerEnabled(bool enabled);
        [[nodiscard]] bool serEnabled() const;

        // Debug timing knob: raygen exits after the step-0 primary trace.
        void setMeasurePrimaryTraceOnly(bool enabled);
        [[nodiscard]] bool measurePrimaryTraceOnly() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANPATHTRACER_HPP
