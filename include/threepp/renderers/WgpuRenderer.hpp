
#ifndef THREEPP_WGPURENDERER_HPP
#define THREEPP_WGPURENDERER_HPP

#include "threepp/renderers/Renderer.hpp"
#include "threepp/canvas/Canvas.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace threepp {

    class WgpuPathTracer;

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
        /// When true, render() routes scene drawing through the embedded
        /// WgpuPathTracer instead of the raster pipeline.  The path tracer is
        /// lazily constructed on first use; toggle freely between frames.
        /// Configure path-tracer-specific settings via pathTracer().
        bool usePathTracer = false;

        explicit WgpuRenderer(Canvas& canvas);

        void render(Object3D& scene, Camera& camera) override;

        /// Embedded path tracer.  Lazily constructed at the current renderer
        /// size on first call.  Use this to set exposure, denoiser params,
        /// max bounces, etc.  Toggle activation with usePathTracer.
        WgpuPathTracer& pathTracer();

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        /// Set the pixel-ratio value that appears in shader uniforms (_pad field)
        /// WITHOUT reconfiguring the surface. Use when you need shaders to see a
        /// scale factor but want the surface to stay at its current resolution.
        void setPixelRatioHint(float value);

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

        /// Access the underlying WGPUSurface handle (type-erased).
        /// Returns nullptr for headless canvases.
        [[nodiscard]] void* nativeSurface() const;

        /// Set MSAA sample count (1 = no MSAA, 4 = 4x MSAA). Default: 1.
        /// Must be 1 or 4. Invalidates cached pipelines.
        void setSampleCount(uint32_t count);
        [[nodiscard]] uint32_t getSampleCount() const;

        /// Set maximum light counts per type. Rebuilds the light GPU buffer
        /// and invalidates all cached pipelines (shaders encode array sizes).
        /// Call before the first render, or between frames.
        void setMaxLights(int maxDir, int maxPoint, int maxSpot, int maxHemi);

        /// Set shadow map configuration. Rebuilds shadow GPU resources and
        /// invalidates cached pipelines (shaders encode shadow array sizes).
        /// Call before the first render, or between frames.
        void setShadowConfig(uint32_t mapSize, int maxShadowLights, int maxShadowPointLights);

        /// Register a callback invoked at the end of each render pass (before encoder end).
        /// Receives the WGPURenderPassEncoder as void*. Used for ImGui overlay rendering.
        void setOverlayCallback(std::function<void(void*)> callback);

        /// Access the native color texture of the current render target (type-erased).
        /// Returns nullptr if no render target is set.
        [[nodiscard]] void* nativeRenderTargetTexture() const;

        /// Surface texture format as uint32_t (cast to WGPUTextureFormat).
        /// Avoids exposing WebGPU types in the public header.
        [[nodiscard]] uint32_t nativeSurfaceFormat() const;

        /// Current frame's depth texture view (WGPUTextureView, type-erased).
        /// Valid after the first render() call in a frame; nullptr otherwise.
        [[nodiscard]] void* nativeFrameDepthView() const;

        /// Active render command encoder (WGPUCommandEncoder, type-erased).
        /// Valid after the first render() call in a frame; nullptr otherwise.
        [[nodiscard]] void* nativeRenderCommandEncoder() const;

        /// MSAA sample count of the current frame's depth buffer (1 or getSampleCount()).
        /// Returns 1 when a tone-mapping intermediate RT is in use.
        [[nodiscard]] uint32_t nativeFrameDepthSampleCount() const;

        /// Finalize the current frame: run any deferred post-processing (tone
        /// mapping blit, ImGui overlay) and present the surface texture.
        /// Normally called automatically by the Canvas animate loop. Call
        /// explicitly when using render() outside of animate(), or before
        /// reconfiguring the surface.
        void endFrame();

        /// Submit any pending render commands to the GPU queue. No-op when a
        /// canvas frame is active (commands are batched until endFrame()).
        /// Call between render-target renders and subsequent GPU reads
        /// (e.g. post-processing passes) to ensure proper ordering.
        void flush();

        void resetState();
        void dispose() override;

        ~WgpuRenderer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPURENDERER_HPP
