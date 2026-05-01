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

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
