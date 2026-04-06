#ifndef THREEPP_WGPUPATHTRACER_HPP
#define THREEPP_WGPUPATHTRACER_HPP

#include <memory>
#include <utility>

namespace threepp {

    class WgpuRenderer;
    class Object3D;
    class Camera;

    /// GPU path tracer / ray tracer built on WebGPU compute shaders.
    /// Wraps a WgpuRenderer for device access and final display blit.
    class WgpuPathTracer {

    public:
        /// Create a path tracer.
        /// @param renderer  WgpuRenderer that owns the WebGPU device and surface.
        /// @param size      Initial viewport size in pixels (width, height).
        explicit WgpuPathTracer(WgpuRenderer& renderer, std::pair<int, int> size);

        ~WgpuPathTracer();

        WgpuPathTracer(const WgpuPathTracer&) = delete;
        WgpuPathTracer& operator=(const WgpuPathTracer&) = delete;

        /// Trace / accumulate one frame and blit to screen.
        void render(Object3D& scene, Camera& camera);

        /// Scale applied to environment/sky light contribution. Default: 1.0.
        /// Set < 1.0 to reduce env light influence on path-traced results.
        void setEnvIntensity(float intensity);
        [[nodiscard]] float envIntensity() const;

        /// Maximum number of bounces for path tracing. Default: 8.
        /// Higher values improve light transport accuracy at the cost of performance.
        void setMaxBounces(int bounces);
        [[nodiscard]] int maxBounces() const;

        /// Exposure multiplier applied during tone mapping. Default: 1.0.
        /// Adjusting this does not reset accumulation.
        void setExposure(float exposure);
        [[nodiscard]] float exposure() const;

        /// Per-contribution firefly clamp on indirect MIS paths (luminance cap).
        /// Default: 8.0 — matches production renderers (Arnold/Cycles/RenderMan).
        /// Pass a very large value (or <= 0 for auto-disable) when unbiased HDR
        /// is required — e.g. ML training data, light-transport validation.
        /// Primary-ray emissive hits are never clamped.
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        /// Enable/disable the à-trous wavelet denoiser (path tracer mode only). Default: true.
        void setDenoiserEnabled(bool enabled);
        [[nodiscard]] bool denoiserEnabled() const;

        /// Enable/disable ReSTIR DI for direct illumination. Default: false.
        void setReSTIREnabled(bool enabled);
        [[nodiscard]] bool restirEnabled() const;

        /// Enable/disable hybrid rasterization: rasterize primary G-buffer, then
        /// path-trace bounces 1+. Reduces primary BVH traversal cost, especially
        /// valuable for fast camera motion (driving simulators). Default: false.
        /// Transmissive surfaces fall back to full ray tracing.
        void setHybridMode(bool enabled);
        [[nodiscard]] bool hybridMode() const;

        /// Resize accumulation textures. Resets accumulation.
        void setSize(std::pair<int, int> size);
        [[nodiscard]] std::pair<int, int> size() const;

        /// Render resolution scale factor. Default: 1.0 (native).
        /// Set < 1.0 to render at lower resolution for higher FPS (e.g. 0.5 = half res).
        void setPixelScale(float scale);
        [[nodiscard]] float pixelScale() const;

        /// Enable/disable foveated rendering during camera motion. Default: true.
        /// Renders center at full resolution, periphery at reduced resolution.
        /// Converges to full quality when camera is stationary (lossless).
        void setFoveatedRendering(bool enabled);
        [[nodiscard]] bool foveatedRendering() const;

        [[nodiscard]] int frameCount() const;
        void resetAccumulation();

        /// Objects on this layer are rasterized as overlay (bypassing path tracing entirely).
        /// Useful for gizmos like TransformControls. Set to -1 to disable (default).
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        /// Force a full scene rebuild (geometry + materials) on the next render call.
        void markDirty();

        void dispose();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPUPATHTRACER_HPP
