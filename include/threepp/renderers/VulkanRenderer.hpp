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

        // Ray-traced env ambient-occlusion / GI. ON by default — soft RT AO/GI
        // (costs occlusion rays; paired with setDenoise, also on by default, to keep
        // the 1-spp gather noise-free). Turn OFF to drop the per-pixel ray cost, or
        // if occlusion-testing the IBL makes a bright HDRI look like it casts shadows.
        void setDeferredAO(bool enabled);
        [[nodiscard]] bool deferredAO() const;

        // Volumetric SPOT-light beams: ray-marched single scattering through a
        // uniform thin haze. `density` is the scattering coefficient σ (1/m; 0 =
        // off, no cost); `anisotropy` is the Henyey-Greenstein g.
        void setDeferredVolumetrics(float density, float anisotropy = 0.55f);

        // Volumetric DIRECTIONAL-light fog: ray-marches the camera→surface air
        // column, tracing an RT shadow ray toward each sun per step so trees and
        // terrain carve real light shafts, and brightens the haze toward the sun
        // via the Henyey-Greenstein phase (driven by setFogAnisotropy). Only
        // contributes when scene.fog is set; opt-in because of the per-step
        // shadow-ray cost. This is what gives an outdoor (sun-lit) deferred scene
        // genuine volume rather than flat distance-faded haze.
        void setVolumetricFog(bool enabled);
        [[nodiscard]] bool volumetricFog() const;

        // Procedural star field on SKY pixels — hash-based points in direction
        // space, pixel-crisp at any resolution/FOV. 0 disables; ~1.0 = night sky.
        void setDeferredStarfield(float intensity);

        // ── Automatic exposure (eye adaptation) ──────────────────────────────
        // When enabled the renderer samples the log2-luma histogram of the
        // rendered frame each tick and drives toneMappingExposure toward the
        // value that maps the scene's weighted-average luminance to 18% gray,
        // using an asymmetric EMA (fast constriction, slow dilation).
        // toneMappingExposure is IGNORED while auto-exposure is active.
        void setAutoExposure(bool enabled);
        [[nodiscard]] bool autoExposure() const;

        // EV per second for brightness adaptation (default 2.0).
        // Dilation (scene-darkens) is applied at 0.5× this speed.
        void setAutoExposureSpeed(float evPerSecond);

        // Exposure clamp in EV relative to 1.0 (default -3 to +3 EV).
        // E.g. setAutoExposureRange(-2, 4) limits to 0.25× .. 16× exposure.
        void setAutoExposureRange(float minEV, float maxEV);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
