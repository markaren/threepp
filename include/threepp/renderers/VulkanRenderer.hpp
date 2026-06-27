// Vulkan deferred renderer. Built against the Vulkan SDK and requires
// VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure (ray-queried
// AO/GI) and VK_KHR_ray_query.
//
// Shades a clean, analytic, noise-free base from the raster material G-buffer
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// plus ray-queried ambient occlusion / global illumination — interactive and
// noise-free, the default for synthetic-perception work. The reference path
// tracer lives in its sibling VulkanPathTracer; both share VulkanRendererCore.
//
// Co-exists with GLRenderer / WgpuRenderer; selected by the application when a
// Canvas is created with GraphicsAPI::Vulkan.

#ifndef THREEPP_VULKANRENDERER_HPP
#define THREEPP_VULKANRENDERER_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/VulkanRendererCore.hpp"

#include <memory>

namespace threepp {

    class VulkanRenderer : public VulkanRendererCore {

    public:
        explicit VulkanRenderer(Canvas& canvas);

        ~VulkanRenderer() override;

        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

        // ── Deferred shading knobs ────────────────────────────────────────────

        // Samples per pixel per frame for the path tracer. Default 1. Clamped >= 1.
        // (Carried while the PT path is being extracted; moves to VulkanPathTracer.)
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

        // Ray-traced env ambient-occlusion / GI. OFF by default — occlusion-testing
        // the IBL makes the HDRI appear to cast shadows. On = soft RT AO/GI (costs
        // occlusion rays; pair with setDenoise for noise).
        void setDeferredAO(bool enabled);
        [[nodiscard]] bool deferredAO() const;

        // Volumetric SPOT-light beams: ray-marched single scattering through a
        // uniform thin haze. `density` is the scattering coefficient σ (1/m; 0 =
        // off, no cost); `anisotropy` is the Henyey-Greenstein g.
        void setDeferredVolumetrics(float density, float anisotropy = 0.55f);

        // Procedural star field on SKY pixels — hash-based points in direction
        // space, pixel-crisp at any resolution/FOV. 0 disables; ~1.0 = night sky.
        void setDeferredStarfield(float intensity);

        // ── Reference-PT knobs (kept on VulkanRenderer until the peer split is
        //    complete; these will move to VulkanPathTracer). ───────────────────

        // Extra primary rays fired at detected silhouette pixels. 0 disables.
        void setSilhouetteMsaaExtra(uint32_t extra);
        [[nodiscard]] uint32_t silhouetteMsaaExtra() const;

        // Per-NEE-sample firefly clamp. Default 30.0; 0 disables (1e30 sentinel).
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // Max real scatter events per path. Default 4. Clamped to [1, 16]. Changing
        // the value resets accumulation.
        void setMaxBounces(int bounces);
        [[nodiscard]] int maxBounces() const;

        // Manually reset path-tracer frame accumulation. Issues a vkDeviceWaitIdle
        // internally — not callable from inside a render() pass.
        void resetAccumulation();

        // Renderer shading strategy (legacy runtime switch). RasterFirst is the
        // deferred path; ReferencePT runs the megakernel path tracer. Being
        // retired in favour of the VulkanPathTracer peer class.
        enum class RenderMode {
            RasterFirst,
            ReferencePT,
        };
        void setRenderMode(RenderMode mode);
        [[nodiscard]] RenderMode renderMode() const;

        void setPerSppJitterHybrid(bool enabled);
        [[nodiscard]] bool perSppJitterHybrid() const;

        // ReSTIR DI master toggle (streaming RIS + temporal + spatial reuse at
        // primary surfaces). Off (default) falls back to per-light NEE.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // ReSTIR DI visibility reuse (Bitterli 2020 §5). Default on; active only
        // while ReSTIR DI is enabled.
        void setRestirDIVisibilityReuse(bool enabled);
        [[nodiscard]] bool restirDIVisibilityReuse() const;

        // ReSTIR GI master toggle (Stage 1a). Default off.
        void setRestirGIEnabled(bool enabled);
        [[nodiscard]] bool restirGIEnabled() const;

        // NVIDIA Shader Execution Reordering (SER) opt-out. On by default where
        // supported. Zero-cost flip (both raygen variants pre-built).
        void setSerEnabled(bool enabled);
        [[nodiscard]] bool serEnabled() const;

        // Debug timing knob: raygen exits after the step-0 primary trace. Image
        // goes black; for a few-frame measurement. Default off.
        void setMeasurePrimaryTraceOnly(bool enabled);
        [[nodiscard]] bool measurePrimaryTraceOnly() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
