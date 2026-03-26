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

        /// Enable/disable the à-trous wavelet denoiser (path tracer mode only). Default: true.
        void setDenoiserEnabled(bool enabled);
        [[nodiscard]] bool denoiserEnabled() const;

        /// Resize accumulation textures. Resets accumulation.
        void setSize(std::pair<int, int> size);
        [[nodiscard]] std::pair<int, int> size() const;

        [[nodiscard]] int frameCount() const;
        void resetAccumulation();

        void dispose();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPUPATHTRACER_HPP
