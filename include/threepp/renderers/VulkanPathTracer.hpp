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

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANPATHTRACER_HPP
