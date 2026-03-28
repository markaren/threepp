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
        enum class Mode {
            Raytracer,  ///< Deterministic GGX shading, 4x RGSS AA, single mirror bounce
            PathTracer  ///< Monte Carlo path tracing with progressive accumulation
        };

        /// Create a path tracer.
        /// @param renderer  WgpuRenderer that owns the WebGPU device and surface.
        /// @param size      Initial viewport size in pixels (width, height).
        explicit WgpuPathTracer(WgpuRenderer& renderer, std::pair<int, int> size);

        ~WgpuPathTracer();

        WgpuPathTracer(const WgpuPathTracer&) = delete;
        WgpuPathTracer& operator=(const WgpuPathTracer&) = delete;

        /// Trace / accumulate one frame and blit to screen.
        void render(Object3D& scene, Camera& camera);

        void setMode(Mode mode);
        [[nodiscard]] Mode mode() const;

        /// Set samples per pixel for raytracer mode (1, 2, or 4). Default: 1.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

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

        /// Ambient light factor added per bounce. Default: 0.03.
        void setAmbientFactor(float factor);
        [[nodiscard]] float ambientFactor() const;

        /// Max radiance clamp to suppress fireflies. Default: 10.0. Set 0 to disable.
        void setFireflyClamp(float clamp);
        [[nodiscard]] float fireflyClamp() const;

        /// Per-pixel frame count cap when a shadow/bounce ray hits a moved mesh. Default: 6.
        /// Lower values make shadows on static surfaces refresh faster after object movement.
        void setMovedPixelFC(float fc);
        [[nodiscard]] float movedPixelFC() const;

        /// Enable/disable the à-trous wavelet denoiser (path tracer mode only). Default: true.
        void setDenoiserEnabled(bool enabled);
        [[nodiscard]] bool denoiserEnabled() const;

        /// Resize accumulation textures. Resets accumulation.
        void setSize(std::pair<int, int> size);
        [[nodiscard]] std::pair<int, int> size() const;

        [[nodiscard]] int frameCount() const;
        void resetAccumulation();

        /// Objects on this layer are rasterized as overlay (bypassing path tracing entirely).
        /// Useful for gizmos like TransformControls. Set to -1 to disable (default).
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        void dispose();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPUPATHTRACER_HPP
