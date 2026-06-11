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
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace threepp {

    class Mesh;

    class VulkanRenderer : public Renderer {

    public:
        explicit VulkanRenderer(Canvas& canvas);

        ~VulkanRenderer() override;

        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        // The actual surface (swapchain) size in PIXELS. Differs from size()
        // when the OS display scaling is not 100%: size() is the logical
        // window size, this is what readRGBPixels()/writeFramebuffer()
        // operate at. Use it for any pixel-space math on read-back frames.
        [[nodiscard]] WindowSize framebufferSize() const;

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

        // Save the last presented frame to disk (.png / .jpg / .jpeg / .bmp),
        // creating parent directories as needed — same convenience GLRenderer
        // and WgpuRenderer expose. Wraps readRGBPixels(); call after render().
        // Throws on unsupported extension or write failure.
        void writeFramebuffer(const std::filesystem::path& filename) override;

        // Toggle scene-only swapchain capture. When enabled, the renderer
        // snapshots the post-TAA / pre-overlay swapchain image into a
        // host-visible staging buffer each frame. `readSceneRGBPixels`
        // returns that snapshot — *without* any sprite / ImGui overlays
        // composited on top, which is what sensor pipelines (event
        // cameras, photometric trackers) need to avoid feedback-looping
        // through their own on-screen visualisation.
        //
        // Pays a per-frame swapchain-sized copy when enabled (~0.5 ms at
        // 1080p on a 4070). No cost when disabled.
        void setSceneCaptureEnabled(bool enabled);
        [[nodiscard]] bool sceneCaptureEnabled() const;
        [[nodiscard]] std::vector<unsigned char> readSceneRGBPixels();

        // ── GPU event camera (DVS) detector ───────────────────────────────
        // Lightweight compute pass that runs immediately after the
        // scene-capture copy: reads the post-TAA scene buffer, updates a
        // persistent per-pixel log-luminance reference, and paints a
        // black/white/grey accumulator image (positive event → white,
        // negative → black, no recent event → mid-grey with exponential
        // decay). The accumulator is copied to a 3-slot host-visible ring
        // buffer at the end of the dispatch.
        //
        // readEventCameraVisualisation() returns the RGBA8 bytes of the
        // OLDEST ring slot (guaranteed complete — no vkDeviceWaitIdle
        // needed), so the call is a plain memcpy. Two-frame display lag
        // is the cost; capture overhead drops from ~25 ms (CPU readback
        // + per-pixel loop) to <2 ms (compute dispatch + small memcpy).
        //
        // Enabling the event camera implicitly enables scene capture —
        // the detector consumes that buffer as its input.
        struct EventCameraParams {
            float    threshold        = 0.15f;
            float    decay            = 0.85f;
            float    minLuma          = 0.005f;
            uint32_t maxEventsPerPixel = 5;
            // Microsecond timestamp tagged onto every event emitted this
            // frame. Set this per-frame from your sim or wall clock; all
            // events from one render() share this value (matches real DVS
            // "event packet" semantics — sub-frame timing isn't available).
            uint32_t frameTimeUs       = 0u;
        };

        // 16-byte event record. Matches the GPU layout byte-for-byte.
        struct Event {
            uint32_t x;
            uint32_t y;
            int32_t  polarity;  // +1 (bright) / -1 (dark)
            uint32_t t_us;
        };
        void setEventCameraEnabled(bool enabled);
        [[nodiscard]] bool eventCameraEnabled() const;
        void setEventCameraParams(const EventCameraParams& params);
        [[nodiscard]] EventCameraParams eventCameraParams() const;
        [[nodiscard]] std::vector<unsigned char> readEventCameraVisualisation() const;

        // Zero-allocation variant — writes the accumulator's RGBA8 bytes
        // straight into the caller's buffer. Returns the byte count written
        // (width × height × 4 on success, 0 on failure). The buffer must
        // have capacity ≥ width × height × 4. Bypasses the per-frame
        // std::vector allocation AND the caller-side memcpy that the
        // returning-vector overload forces, so it's the right call from a
        // tight events-only animate() loop targeting 500 Hz+.
        size_t readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const;

        // Pin the sensor's pixel resolution. The event_shade pass nearest-
        // samples the (typically larger) gbuf into a sensor-res luma buffer,
        // and event_detect runs at the sensor extent — so smaller values cut
        // detector + readback work by the squared scale factor. Pass (0, 0)
        // to match the swapchain (default).
        //
        // Typical real DVS hardware sits in the 128² – 640×480 range; values
        // outside [16, swapchainExtent] are clamped. Issues a vkDeviceWaitIdle
        // internally to recreate the detector's persistent images at the new
        // resolution, so don't call this inside render().
        void setEventCameraResolution(uint32_t width, uint32_t height);
        [[nodiscard]] std::pair<uint32_t, uint32_t> eventCameraResolution() const;

        // Sparse event-stream readback. Writes up to `cap` events into
        // `dst` and returns the count written; `overflowed` (if non-null)
        // is set to true when the shader dropped events for that frame
        // because the GPU buffer hit capacity (~256k events). Two-frame
        // display latency, no GPU wait. Pair this with frameTimeUs in
        // EventCameraParams set on the frame you want timestamped.
        //
        // This is the "real sensor" output — analogous to what hardware
        // DVS readout interfaces produce. Suitable for ROS bridges,
        // .aedat exports, ML training pipelines, etc.
        size_t readEventStreamInto(Event* dst, size_t cap,
                                   bool* overflowed = nullptr) const;

        // Events-only render mode. When on, recordCommandBuffer runs the
        // raster G-buffer prepass and then short-circuits — no photon
        // emit, no PT raygen, no denoise, no TAA, no upscale, no overlay
        // draws. The swapchain is cleared to black; event_shade and
        // event_detect run in the outer flow exactly as in the normal
        // path, so the screen-space sprite overlay (event accumulator)
        // remains the only visible content.
        //
        // Designed for high-rate event-camera sampling (~500 Hz target)
        // where the renderer's visual output isn't displayed at all —
        // only the event readout matters. The raster G-buffer prepass
        // always runs (event_shade reads its normal + ids attachments).
        // Has no effect when the event camera is off.
        void setEventsOnlyMode(bool enabled);
        [[nodiscard]] bool eventsOnlyMode() const;

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

        // World-Y of the water surface for bounding underwater fog to the
        // water column. When the camera is below this Y, the fog distance
        // is capped at the ray's exit through the surface — prevents fog
        // from absorbing sky radiance on rays that leave the water.
        // Default 1e30 (no limit). Set each frame alongside scene.fog.
        void setFogWaterSurfaceY(float y);

        // Samples per pixel per frame for the path tracer. Default 1.
        // Each sample is an independent jittered primary ray; in-frame samples
        // are summed into the accumulator with weight `spp`, so per-pixel FC
        // also advances by `spp`. Clamped to >= 1.
        void setSamplesPerPixel(int spp);
        [[nodiscard]] int samplesPerPixel() const;

        // Path-trace render scale. raygen, denoise, and the hybrid raster
        // G-buffer run at (swapchain extent × scale); the temporal AA pass
        // then reconstructs full swapchain resolution by accumulating the
        // jittered low-res samples into a full-res history (TAA upsampling —
        // sharper than a plain blit, since it recovers detail across frames).
        // With TAA disabled the low-res result is instead bilinearly blitted
        // up. The wireframe overlay and ImGui always draw at full resolution.
        // Quadratic FPS lever: 0.5 quarters the path-trace pixel count.
        // Clamped to [0.25, 1.0]. Default 1.0 (no scaling — pixel-identical
        // to the unscaled path, zero added cost).
        //
        // Changing the scale reallocates all render-extent resources and
        // resets accumulation; it issues a vkDeviceWaitIdle internally and so
        // must not be called from inside a render() pass.
        void setRenderScale(float scale);
        [[nodiscard]] float renderScale() const;

        // Denoiser toggle for the ACTIVE render mode — one switch, both paths.
        // PT/hybrid: 5×5 à-trous edge-aware filter on the temporally-accumulated
        // radiance before tonemap + sRGB encode (off = tonemap-only pass-through,
        // pixel-identical to the prior in-shader tonemap path). RasterFirst:
        // SVGF à-trous on the ray-traced diffuse-indirect (AO/GI) channel only
        // (off = raw noisy GI; the raster base is deterministic either way).
        // Default on. Switching RenderMode needs no re-toggle.
        void setDenoise(bool enabled);
        [[nodiscard]] bool denoise() const;

        // HDR bloom, added in linear HDR *before* the tone-map curve. A
        // soft-knee bright pass (see setBloomThreshold) keeps only highlights,
        // which are blurred and added back additively, so darks/mid-tones stay
        // crisp and bright areas glow. 0 disables the bloom passes (the
        // composite still owns tone map + sRGB). Typical intensity: 0.2–0.8.
        void setBloomIntensity(float intensity);
        [[nodiscard]] float bloomIntensity() const;

        // Deprecated alias of setDenoise()/denoise() — the denoise toggle is
        // unified across render modes (it used to control only the RasterFirst
        // GI denoiser, so mode switches required toggling two flags).
        void setDeferredDenoise(bool enabled);
        [[nodiscard]] bool deferredDenoise() const;

        // RasterFirst ray-traced env ambient-occlusion / GI. OFF by default —
        // occlusion-testing the IBL makes the HDRI appear to cast shadows. Off =
        // flat env IBL (the directional light still casts shadows). On = soft RT
        // AO/GI (costs occlusion rays; pair with setDeferredDenoise for noise).
        void setDeferredAO(bool enabled);
        [[nodiscard]] bool deferredAO() const;

        // RasterFirst volumetric SPOT-light beams: ray-marched single scattering
        // through a uniform thin haze — searchlight / lighthouse beams, visible
        // against sky and surfaces alike. `density` is the scattering coefficient
        // σ (1/m; typical 0.005–0.05, 0 = off, no cost); `anisotropy` is the
        // Henyey-Greenstein g (forward-peaked ≈ 0.5–0.8 reads as atmospheric
        // haze). RasterFirst only; the PT path has its own fog volumetrics.
        void setDeferredVolumetrics(float density, float anisotropy = 0.55f);

        // RasterFirst procedural star field on SKY pixels — hash-based points
        // evaluated in direction space, so they stay pixel-crisp at any
        // resolution/FOV (env-texture stars are sub-texel features that
        // magnify into blobs). 0 disables (no cost); ~1.0 = naked-eye night
        // sky. Pair with a dark environment.
        void setDeferredStarfield(float intensity);

        // ── PhysX soft-body zero-copy interop (CUDA → Vulkan) ────────────────
        // Re-backs `mesh`'s per-frame tet-position buffer (the tet_skinning.comp
        // input) with EXPORTED external device memory and registers `deviceCopy`
        // to be invoked each frame in place of the CPU upload. The caller (the
        // PhysX soft-body glue) imports the returned OS handle into CUDA and has
        // `deviceCopy` issue a device→device copy from PhysX's deformed-position
        // buffer — the tet data then never touches the host.
        //   • Call AFTER the first render (the mesh's tet state must exist) —
        //     returns an empty handle until then; poll.
        //   • Returns an empty handle when the device lacks the external-memory
        //     extension. On Windows the handle is a Win32 NT handle owned by the
        //     renderer (valid until disable/teardown; CUDA's import duplicates
        //     it). On POSIX it is an fd cast into the pointer, ownership
        //     transferring to CUDA on successful import.
        //   • If the CUDA import fails, call disableSoftBodyInterop to fall back
        //     to the CPU upload path.
        struct SoftBodyInteropHandle {
            void*  osHandle  = nullptr;
            size_t sizeBytes = 0;
        };
        SoftBodyInteropHandle enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy);
        void disableSoftBodyInterop(const Mesh& mesh);

        // Bloom bright-pass cutoff in linear-HDR luma: only scene values above
        // this (with a soft knee below it) contribute to the glow. Higher =
        // only the very brightest highlights bloom. Typical: 0.8–2.0.
        void setBloomThreshold(float threshold);
        [[nodiscard]] float bloomThreshold() const;

        // Bloom input clamp in linear-HDR luma (UE's bloom "max brightness"):
        // each bright-pass tap is capped here, so a tiny ultra-bright specular
        // highlight (an analytic light mirrored in smooth metal) can't pulse
        // the halo radius as the TAA jitter swings its per-frame intensity by
        // orders of magnitude. <= 0 disables (default — unclamped legacy
        // look). Typical: 8–32; lower = more stable but dims very-bright halos.
        void setBloomClamp(float clampMax);
        [[nodiscard]] float bloomClamp() const;

        // Post-TAA RCAS sharpen strength — restores high-frequency detail the
        // temporal resolve softens (contrast-limited, so it won't ring or
        // amplify fireflies). 0 disables the sharpen pass. Typical: 0.2–0.5.
        void setSharpenStrength(float amount);
        [[nodiscard]] float sharpenStrength() const;

        // Extra primary rays fired at detected silhouette pixels (mesh-ID,
        // depth-gradient, or diagonal neighbour mismatch in the raster
        // prepass). 0 disables silhouette MSAA. N gives (N+1)× total
        // samples at edge pixels — default 7 (8× MSAA). Capped internally
        // at 31. Edge pixels are typically 5–15 % of frame; cost scales
        // ~linearly with N on those pixels only.
        void setSilhouetteMsaaExtra(uint32_t extra);
        [[nodiscard]] uint32_t silhouetteMsaaExtra() const;

        // Per-NEE-sample firefly clamp: any direct-lighting contribution whose
        // luminance exceeds `cap` is rescaled down to `cap`. Default 30.0.
        // Pass 0 to disable (stored as a 1e30 sentinel — gates never fire).
        // Mirrors WgpuPathTracer::setFireflyClamp.
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // Max real scatter events per path. Default 4. Clamped internally to
        // [1, 16]. Diffuse / glossy primaries adaptively cap themselves at 3 / 4
        // in raygen, so raising this mainly affects deeper metal / mirror /
        // glass paths. Changing the value resets accumulation.
        // Mirrors WgpuPathTracer::setMaxBounces.
        void setMaxBounces(int bounces);
        [[nodiscard]] int maxBounces() const;

        // Manually reset path-tracer frame accumulation. Wipes per-pixel
        // running mean + history (gbuf + accum + ReSTIR DI reservoirs),
        // invalidates reproject state (prev camera + per-mesh prevWorld),
        // and rewinds the sampleIndex counter so the next frame cold-starts
        // from sample 1. Use after lens / focus / lighting changes the
        // renderer can't detect on its own. Issues a vkDeviceWaitIdle
        // internally — must not be called from inside a render() pass.
        // Mirrors WgpuPathTracer::resetAccumulation.
        void resetAccumulation();

        // Renderer shading strategy. Both modes share the raster G-buffer
        // prepass (depth, normal, motion vectors, per-pixel IDs) and the whole
        // post chain (denoise / bloom / TAA); they diverge only in how the
        // surface is shaded.
        //
        //   ReferencePT — the path tracer shades everything (the existing
        //                 infra): the raster prepass supplies primary
        //                 visibility and raygen path-traces all direct +
        //                 indirect light. Ground-truth quality, but carries the
        //                 full PT noise and cost. Kept as the reference / high-
        //                 fidelity option.
        //
        //   RasterFirst — raster shades a clean, analytic, noise-free base
        //                 (direct analytic lights + IBL) and the path tracer
        //                 contributes only additive accents (reflections, GI,
        //                 caustics) on top. The intended default once built
        //                 out: most of the look of PT without the noise or the
        //                 per-pixel ray cost.
        //
        // NOTE: RasterFirst is being landed in stages. Until the deferred
        // shading path exists it falls back to ReferencePT behaviour, so the
        // two modes currently render identically. Default is ReferencePT and
        // flips to RasterFirst once the base + accent passes are in.
        enum class RenderMode {
            RasterFirst,
            ReferencePT,
        };
        void setRenderMode(RenderMode mode);
        [[nodiscard]] RenderMode renderMode() const;

        void setPerSppJitterHybrid(bool enabled);
        [[nodiscard]] bool perSppJitterHybrid() const;

        // ReSTIR DI master toggle. When on, the path tracer uses
        // streaming RIS + temporal + spatial reuse at primary surfaces — one
        // shadow ray to a single chosen sample replaces the per-light NEE
        // loop, with reservoirs reused across frames and neighbours. When
        // off (default), falls back to classic per-light NEE at primary (one
        // shadow ray per analytic light + one per emissive sample). Bounces
        // and env NEE are unaffected either way.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // ReSTIR DI visibility reuse (Bitterli 2020 §5). Shadow-tests the
        // RIS-selected candidate before temporal/spatial reuse and discards it
        // if occluded, so per-pixel history converges onto visible lights — the
        // main DI quality win in occluded interiors. Costs one extra shadow ray
        // per RIS pixel; only active while ReSTIR DI is enabled. Default on.
        void setRestirDIVisibilityReuse(bool enabled);
        [[nodiscard]] bool restirDIVisibilityReuse() const;

        // ReSTIR GI master toggle (Stage 1a). When on, primary chit launches
        // a BSDF-sampled indirect sub-ray and assembles a single-sample
        // reservoir for the bounce-1 contribution; raygen's step-1 continues
        // from the sub-trace's hit (xs) at bounce 2 to avoid double-counting.
        // At M=1 the result is statistically equivalent to classic MC; the
        // visible variance reduction lands with Stage 1b/1c (temporal +
        // spatial reuse, separate commits). Default off.
        void setRestirGIEnabled(bool enabled);
        [[nodiscard]] bool restirGIEnabled() const;

        // NVIDIA Shader Execution Reordering (SER) opt-out. SER warp-reorders
        // path-tracer threads by hit material before the closest-hit
        // invocation — a 10-30% speed-up on incoherent diffuse bounces on
        // supported hardware (Ada / Ampere). On by default wherever the driver
        // advertises VK_NV_ray_tracing_invocation_reorder. Pass false to force
        // the plain-traceRayEXT fallback raygen (A/B perf comparison, or to
        // sidestep a driver-specific SER issue). No-op on unsupported devices
        // (already on the fallback) and while ReSTIR GI is on (SER is
        // auto-disabled there). Zero-cost flip: both raygen variants are
        // pre-built, so this only selects which one runs — no pipeline rebuild,
        // and the rendered result is unchanged. serEnabled() returns the user
        // preference, not whether SER is actually active.
        void setSerEnabled(bool enabled);
        [[nodiscard]] bool serEnabled() const;

        // Hybrid-mode raster overlay: post-TAA, draws wireframe-flagged
        // meshes (any material with `wireframe = true`) and Line/LineSegments
        // objects on top of the path-traced image, depth-tested against the
        // raster G-buffer's depth attachment so overlays are correctly
        // occluded by path-traced geometry. Mirrors the WGPU PT overlay path
        // (WgpuPathTracer.cpp:3748).
        //
        // Auto-detected: any Mesh with `material.wireframe == true`,
        // any Line/LineSegments, plus everything on the configured layer.
        // Pass `-1` (default) to disable layer-based selection — material
        // wireframe + Line objects are still drawn. Pass `0..31` to also
        // include any object on that layer (`obj.layers.enable(channel)`).
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        // Day-1 / debug visualization: blit one G-buffer channel directly
        // to the swapchain, bypassing the path tracer.
        //   0 = off (PT consumes the G-buffer normally)
        //   1 = world-space normal
        //   2 = motion vector (NDC delta in red/green)
        //   3 = per-pixel instanceCustomIndex (raw uint16)
        //   4 = albedo (linear base colour in rgb; metalness in alpha)
        void setHybridDebugView(int view);
        [[nodiscard]] int hybridDebugView() const;

        // Debug timing knob: when on, raygen exits the bounce loop and the
        // spp loop immediately after the step-0 primary traceRayEXT (no NEE,
        // no env, no RIS / GI, no bounce continuation, no accumulator write).
        // `lastFrameTimings().pathTraceMs` then measures roughly the primary-
        // trace + dispatch overhead. Compare against the same field with the
        // toggle off to estimate what fraction of PT time the primary trace
        // accounts for — i.e. the upper bound on savings if bounce 0 were
        // shaded from the raster G-buffer instead of traced. The image goes
        // black while on; intended for a few-frame measurement, not gameplay.
        // Default off.
        void setMeasurePrimaryTraceOnly(bool enabled);
        [[nodiscard]] bool measurePrimaryTraceOnly() const;


        // ── Path-traced LIDAR scanner ─────────────────────────────────────
        // Synchronously trace `beams.size()` rays against the same TLAS the
        // path tracer uses, evaluate a back-scatter LIDAR equation at the
        // first hit (Lambertian luminance damped by metal specular loss ×
        // |cos θ| × Beer-Lambert atmospheric attenuation / r²), and return
        // per-beam (position, normal, distance, intensity, hit-instance)
        // tuples. Lets simulation users obtain physically-grounded LIDAR
        // returns from the same GPU-accelerated geometry the renderer
        // sees, rather than rasterising depth into cube faces and reading
        // pixels back (the existing GL `LidarSensor` path).
        //
        // Intensity is normalised so that a perpendicular, 1.0-albedo
        // surface at `params.referenceRange` reads as 1.0; returns below
        // `params.detectorThreshold` are dropped (instanceId = -1).
        //
        // Submits its own command buffer + fence and blocks until results
        // come back. Typical cadence for a real LIDAR is 10-30 Hz, so the
        // round-trip overhead is acceptable. Calling between render()
        // invocations is safe; calling concurrently with render() is not.
        //
        // Beam / return / params types live in helpers/LidarTypes.hpp so
        // GL-side LidarSensor and Vulkan-side PathTracedLidarSensor can
        // share the same output container.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& results,
                       const LidarParams& params = {});


        // Per-frame timings (milliseconds). CPU values come from
        // std::chrono around the host-side hot path of the most recent
        // render() call. GPU values come from VkQueryPool timestamps and
        // lag by one in-flight slot (the most recently *retired* frame —
        // GPU is necessarily one or two frames behind the host). For a
        // steady-state renderer the difference is invisible; for a HUD
        // overlay it's fine to read both as "current frame".
        //
        // First frame after construction returns all zeros. GPU fields
        // are zero on frames where the corresponding pass didn't run
        // (e.g. photonEmitMs == 0 when no glass is visible).
        struct FrameTimings {
            float photonEmitMs   = 0.f;// caustic photon trace (when visible)
            float pathTraceMs    = 0.f;// main RT megakernel
            float denoiseMs      = 0.f;// à-trous passes + finalize tonemap
            float taaMs          = 0.f;// hybrid TAA resolve compute
            float rasterGbufMs   = 0.f;// hybrid G-buffer prepass
            float overlayMs      = 0.f;// hybrid overlay depth + draw
            float cpuEnsureSceneMs = 0.f;// ensureSceneBuilt
            float cpuRecordMs      = 0.f;// recordCommandBuffer
            float cpuFrameMs       = 0.f;// total render() wall time
        };
        [[nodiscard]] FrameTimings lastFrameTimings() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
