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

        // Samples per pixel per frame for the path tracer. Default 2.
        // Each sample is an independent jittered primary ray; in-frame samples
        // are summed into the accumulator with weight `spp`, so per-pixel FC
        // also advances by `spp`. Clamped to >= 1.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

        // Spatial denoiser (5×5 à-trous edge-aware filter) applied to the
        // temporally-accumulated radiance before tonemap + sRGB encode. Default
        // on. When off, the compute pass still runs but acts as a tonemap-only
        // pass-through (pixel-identical to the prior in-shader tonemap path).
        void setDenoise(bool enabled);
        [[nodiscard]] bool denoise() const;

        // Per-NEE-sample firefly clamp: any direct-lighting contribution whose
        // luminance exceeds `cap` is rescaled down to `cap`. Default 20.0.
        // Pass 0 to disable (stored as a 1e30 sentinel — gates never fire).
        // Mirrors WgpuPathTracer::setFireflyClamp.
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
