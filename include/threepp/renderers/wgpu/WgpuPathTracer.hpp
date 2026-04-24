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
    struct LensSettings {
        float fStop = 0.0f;             // <= 0 disables DOF (pinhole)
        float focusDistance = 5.0f;     // world units
        int   apertureBlades = 0;       // 0 = circular, 3..8 = polygonal bokeh
        float apertureRotation = 0.0f;  // radians
    };

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

        /// Scene-wide multiplier applied to every emissive material's output.
        /// Default: 1.0. Useful for quickly brightening/dimming all area lights
        /// (e.g. RectAreaLight quads + emissive meshes) in an indoor PT scene
        /// without touching individual material values. Unbiased — scales the
        /// per-hit emissive contribution; CDF probabilities are unaffected.
        void setEmissiveIntensity(float intensity);
        [[nodiscard]] float emissiveIntensity() const;

        /// Henyey-Greenstein anisotropy for volumetric fog single-scattering.
        /// Range: [-0.95, 0.95]. 0 = isotropic (uniform glow), >0 = forward
        /// scattering (god rays / light shafts when looking toward lights),
        /// <0 = backward scattering. Default: 0. Only affects rendering when
        /// Scene::fog is set (FogExp2 or Fog).
        void setFogAnisotropy(float g);
        [[nodiscard]] float fogAnisotropy() const;

        /// Maximum number of bounces for path tracing. Default: 8.
        /// Higher values improve light transport accuracy at the cost of performance.
        void setMaxBounces(int bounces);
        [[nodiscard]] int maxBounces() const;

        /// Enable CPU-side wall-clock timings for the path tracer frame.
        /// When enabled, the renderer serializes CPU and GPU via wgpuDevicePoll
        /// between submits to measure per-stage cost (pre / bounces / post).
        /// Prints a rolling 60-frame summary to stderr.  Adds ~0.5 ms overhead
        /// per submit — use for diagnostics, not production.
        /// Also togglable via env var WGPU_PATHTRACER_TIMINGS=1.
        void setTimingsEnabled(bool enabled);
        [[nodiscard]] bool timingsEnabled() const;

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

        /// Enable/disable ReSTIR DI for direct illumination. Default: true.
        void setReSTIREnabled(bool enabled);
        [[nodiscard]] bool restirEnabled() const;

        /// Enable/disable ReSTIR GI for indirect illumination (bounce 1). Default: false.
        /// Reuses secondary hit points across frames/neighbors for lower-variance GI.
        void setReSTIRGIEnabled(bool enabled);
        [[nodiscard]] bool restirGiEnabled() const;

        /// Enable/disable the two-level TLAS/BLAS acceleration structure. Default: false.
        /// Experimental — plumbing only in the current build; single-level BVH remains
        /// the active traversal path until the shader rewrite lands.
        void setTlasEnabled(bool enabled);
        [[nodiscard]] bool tlasEnabled() const;

        /// Samples per pixel per frame. Default: 1. Higher values reduce noise
        /// at the cost of proportionally more RT time per frame.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

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

        /// AOV visualization mode. 0=off (normal rendering), 1=depth, 2=normals,
        /// 3=albedo, 4=instance ID, 5=roughness, 6=adaptive bounce (red=reduced, blue=full).
        /// 10+: post-bounce diagnostic AOVs read from pathStateBuf by rt_accum_main.
        ///   10=diffRadFinal, 11=specRadFinal, 12=touchedMoved, 13=flagBits(RGB),
        ///   14=b0Point.fract, 15=primaryDepth, 16=primaryMeshIdx, 17=primaryMatIdx,
        ///   18=b0Alpha.  Intended for debugging; may be removed in future.
        void setAOVMode(int mode);
        [[nodiscard]] int aovMode() const;

        /// Texture atlas tile resolution. Default: 1024. Set to 2048 for higher
        /// quality textures.
        void setTextureResolution(int size);
        [[nodiscard]] int textureResolution() const;

        /// Force a full scene rebuild (geometry + materials) on the next render call.
        void markDirty();

        /// Physical camera lens settings. fStop <= 0 produces a pinhole camera (default).
        /// Any change resets accumulation.
        void setLens(const LensSettings& lens);
        [[nodiscard]] const LensSettings& lens() const;

        /// Convenience: set focusDistance to the world-space distance from camera to target.
        void focusOn(const Camera& camera, const Object3D& target);

        void dispose();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_WGPUPATHTRACER_HPP
