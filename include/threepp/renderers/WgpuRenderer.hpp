
#ifndef THREEPP_WGPURENDERER_HPP
#define THREEPP_WGPURENDERER_HPP

#include "threepp/renderers/Renderer.hpp"
#include "threepp/canvas/Canvas.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace threepp {

    struct WgpuInfo {
        struct {
            size_t geometries = 0;
            size_t textures = 0;
        } memory;

        struct {
            size_t frame = 0;
            size_t calls = 0;
            size_t triangles = 0;
            size_t lines = 0;
            size_t points = 0;
        } render;
    };

    class WgpuRenderer : public Renderer {

    public:
        explicit WgpuRenderer(Canvas& canvas);

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        void setViewport(const Vector4& v) override;
        void setViewport(int x, int y, int width, int height) override;
        void getViewport(Vector4& target) const;

        void setScissor(const Vector4& v) override;
        void setScissor(int x, int y, int width, int height) override;
        void getScissor(Vector4& target) const;
        void setScissorTest(bool boolean) override;
        [[nodiscard]] bool getScissorTest() const;

        void setClearColor(const Color& color, float alpha = 1) override;
        void getClearColor(Color& target) const override;
        [[nodiscard]] float getClearAlpha() const override;
        void setClearAlpha(float alpha) override;

        void clear(bool color = true, bool depth = true, bool stencil = true) override;
        void clearColor() override;
        void clearDepth() override;
        void clearStencil() override;

        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;
        [[nodiscard]] int getActiveCubeFace() const;
        [[nodiscard]] int getActiveMipmapLevel() const;

        std::vector<unsigned char> readRGBPixels() override;
        void readPixels(const Vector2& position, const std::pair<int, int>& size, std::vector<unsigned char>& data);

        void copyFramebufferToTexture(const Vector2& position, Texture& texture, int level = 0) override;

        void copyTextureToImage(Texture& texture) override;

        void writeFramebuffer(const std::filesystem::path& filename);

        [[nodiscard]] bool renderTargetFlipY() const override { return true; }

        [[nodiscard]] const WgpuInfo& info() const;

        /// Access the underlying WGPUDevice handle (type-erased).
        /// Cast with static_cast<WGPUDevice>(renderer.nativeDevice()).
        [[nodiscard]] void* nativeDevice() const;

        /// Access the underlying WGPUQueue handle (type-erased).
        [[nodiscard]] void* nativeQueue() const;

        /// Access the underlying WGPUInstance handle (type-erased).
        [[nodiscard]] void* nativeInstance() const;

        /// Set MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
        /// Must be 1 or 4. Invalidates cached pipelines.
        void setSampleCount(uint32_t count);
        [[nodiscard]] uint32_t getSampleCount() const;

        /// Set maximum light counts per type. Rebuilds the light GPU buffer
        /// and invalidates all cached pipelines (shaders encode array sizes).
        /// Call before the first render, or between frames.
        void setMaxLights(int maxDir, int maxPoint, int maxSpot, int maxHemi);

        /// Register a callback invoked at the end of each render pass (before encoder end).
        /// Receives the WGPURenderPassEncoder as void*. Used for ImGui overlay rendering.
        void setOverlayCallback(std::function<void(void*)> callback);

        /// Access the native color texture of the current render target (type-erased).
        /// Returns nullptr if no render target is set.
        [[nodiscard]] void* nativeRenderTargetTexture() const;

        /// Surface texture format as uint32_t (cast to WGPUTextureFormat).
        /// Avoids exposing WebGPU types in the public header.
        [[nodiscard]] uint32_t nativeSurfaceFormat() const;

        void resetState();
        void dispose() override;

        ~WgpuRenderer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPURENDERER_HPP
