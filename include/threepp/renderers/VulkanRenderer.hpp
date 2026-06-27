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

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
