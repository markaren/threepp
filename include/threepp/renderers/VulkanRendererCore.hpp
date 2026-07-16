// Shared base for the Vulkan renderer.
//
// Owns (through the derived leaf's pImpl) all of the Vulkan infrastructure —
// device/swapchain context, acceleration structures, scene build + visibility,
// material/geometry buffers, the raster G-buffer prepass, and the bloom / TAA /
// overlay post-stack — and implements the shared public API once. The concrete
// leaf, VulkanRenderer, injects its deferred G-buffer shade through the virtual
// recordSceneDispatch hook (see VulkanRenderer.cpp). Kept as a separate base so
// the shared machinery is reusable should another shading strategy be added.

#ifndef THREEPP_VULKANRENDERERCORE_HPP
#define THREEPP_VULKANRENDERERCORE_HPP

#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <utility>
#include <vector>

namespace threepp {

    class Mesh;

    class VulkanRendererCore : public Renderer {

    public:
        ~VulkanRendererCore() override;

        VulkanRendererCore(const VulkanRendererCore&) = delete;
        VulkanRendererCore& operator=(const VulkanRendererCore&) = delete;

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        // The actual surface (swapchain) size in PIXELS (differs from size() when
        // OS display scaling is not 100%). Use it for pixel-space math on
        // read-back frames.
        [[nodiscard]] WindowSize framebufferSize() const;

        // Pixel ratio is unsupported on the Vulkan backend: the swapchain is
        // sized in native device pixels, so getTargetPixelRatio() always
        // returns 1 and setPixelRatio() warns once and ignores the value.
        // Use setRenderScale for resolution scaling.
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

        // Unsupported on the Vulkan backend (the deferred pipeline rewrites
        // every attachment each render()); warns once and returns.
        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        // Offscreen render targets are unsupported (swapchain-only renderer).
        // getRenderTarget() returns nullptr (= default framebuffer, matching
        // three.js semantics); a non-null setRenderTarget() warns once and is
        // ignored — use readGBufferAOV/readRGBPixels for capture instead.
        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        [[nodiscard]] std::vector<unsigned char> readRGBPixels() override;

        // Save the last presented frame to disk (.png/.jpg/.jpeg/.bmp), creating
        // parent dirs as needed. Wraps readRGBPixels(); call after render().
        void writeFramebuffer(const std::filesystem::path& filename) override;

        // Scene-only swapchain capture (post-TAA / pre-overlay) into a host-visible
        // staging buffer. readSceneRGBPixels() returns it without sprite/ImGui
        // overlays — what sensor pipelines need to avoid feedback-looping.
        void setSceneCaptureEnabled(bool enabled);
        [[nodiscard]] bool sceneCaptureEnabled() const;
        [[nodiscard]] std::vector<unsigned char> readSceneRGBPixels();

        // ── G-buffer AOV readback (deferred / hybrid raster prepass) ──────
        // The raster G-buffer already computes high-precision per-pixel geometry
        // and material data every frame. readGBufferAOV copies one attachment
        // from the most recently rendered frame into `out` as its NATIVE GPU
        // format — no 8-bit swapchain round-trip, no id hashing — so ML / sensor
        // pipelines get lossless depth, normals, recoverable integer instance
        // ids and motion vectors. Call after render().
        //
        // On success `out` is resized to width*height*bytesPerPixel (tightly
        // packed, row-major, top-left origin) and true is returned. Returns
        // false before the first frame or when the backend has no G-buffer.
        // width/height come back as the render-scaled G-buffer extent (see
        // setRenderScale), which may be smaller than framebufferSize().
        //
        // Per-AOV element layout (bytesPerPixel):
        //   Depth  (4)  1x float32 : reversed-Z NDC depth in [0,1] (1=near, 0=far)
        //   Normal (8)  4x float16 : world normal encoded n*0.5+0.5 in xyz, roughness in w
        //   Motion (8)  4x float16 : screen-space motion (prevNDC - currNDC) in xy, prevDepth in z
        //   Ids    (8)  4x uint16  : x = instanceCustomIndex+1 (0 = sky/no-hit), y = meshId, z = flag bits
        //   Albedo (4)  4x unorm8  : linear base colour in rgb, metalness in a
        enum class GBufferAOV { Depth, Normal, Motion, Ids, Albedo };
        [[nodiscard]] bool readGBufferAOV(GBufferAOV aov, std::vector<uint8_t>& out,
                                          int& width, int& height, int& bytesPerPixel);

        // ── Segmentation labels for the Ids AOV ──────────────────────────
        // The Ids attachment's .y channel is a STABLE per-object instance id,
        // auto-assigned on first draw (unlike .x, the per-frame visible index,
        // it survives add/remove/hide/LOD). setObjectInstanceId overrides it for
        // a specific object; setObjectClassId tags an object with an 8-bit
        // semantic class written to .z bits 8..15 — so a single readback yields
        // both instance- and semantic-segmentation ground truth. Both take
        // effect on the next render. instanceId is truncated to 16 bits, classId
        // clamped to [0, 255]. Keyed by Object3D::id, so calling before the
        // object is added to the scene is fine.
        void setObjectInstanceId(const Object3D& obj, uint32_t instanceId);
        void setObjectClassId(const Object3D& obj, uint32_t classId);

        // ── GPU event camera (DVS) detector ───────────────────────────────
        struct EventCameraParams {
            float    threshold        = 0.15f;
            float    decay            = 0.85f;
            float    minLuma          = 0.005f;
            uint32_t maxEventsPerPixel = 5;
            uint32_t frameTimeUs       = 0u;
        };

        // 16-byte event record. Matches the GPU layout byte-for-byte.
        struct Event {
            uint32_t x;
            uint32_t y;
            int32_t  polarity;// +1 (bright) / -1 (dark)
            uint32_t t_us;
        };
        void setEventCameraEnabled(bool enabled);
        [[nodiscard]] bool eventCameraEnabled() const;
        void setEventCameraParams(const EventCameraParams& params);
        [[nodiscard]] EventCameraParams eventCameraParams() const;
        [[nodiscard]] std::vector<unsigned char> readEventCameraVisualisation() const;
        size_t readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const;
        void setEventCameraResolution(uint32_t width, uint32_t height);
        [[nodiscard]] std::pair<uint32_t, uint32_t> eventCameraResolution() const;
        size_t readEventStreamInto(Event* dst, size_t cap, bool* overflowed = nullptr) const;
        void setEventsOnlyMode(bool enabled);
        [[nodiscard]] bool eventsOnlyMode() const;

        void dispose() override;

        // ImGui integration handles (Vulkan types erased to void* / uint32_t).
        [[nodiscard]] void* nativeInstance() const;
        [[nodiscard]] void* nativePhysicalDevice() const;
        [[nodiscard]] void* nativeDevice() const;
        [[nodiscard]] void* nativeGraphicsQueue() const;
        [[nodiscard]] uint32_t graphicsQueueFamily() const;
        [[nodiscard]] uint32_t nativeSwapchainFormat() const;// cast to VkFormat
        [[nodiscard]] uint32_t imageCount() const;            // swapchain image count

        // Callback invoked once per frame inside the present render pass, after
        // the scene has been written. The argument is a VkCommandBuffer
        // (type-erased). Set null to disable.
        void setOverlayCallback(std::function<void(void*)> callback);

        // Henyey-Greenstein anisotropy for the fog phase function. Clamped to
        // [-0.95, 0.95]. No effect when scene.fog is unset.
        void setFogAnisotropy(float g);
        [[nodiscard]] float getFogAnisotropy() const;

        // World-Y of the water surface. Bounds the underwater MURK
        // (setUnderwaterMurk) to the water column below this Y; the air fog
        // (scene.fog) is NOT clipped by it. Default 1e30 (no limit).
        void setFogWaterSurfaceY(float y);

        // Underwater murk — a homogeneous absorption/tint medium clipped to BELOW
        // setFogWaterSurfaceY (the water body's own attenuation). Phase 2 fog
        // unification decouples this from scene.fog: scene.fog is now the AIR
        // medium (haze / god rays, unclipped) and this is the SEPARATE below-water
        // medium, so a scene can hold clear air above the waterline and murk below
        // (the fjord). density = σ_t (1/m; 0 = off, the default); color = inscatter
        // tint. Pair it with setFogWaterSurfaceY to set the clip plane.
        void setUnderwaterMurk(float density, const Color& color);
        [[nodiscard]] std::pair<float, Color> underwaterMurk() const;

        // Render scale. The scene shade + hybrid raster G-buffer run at (swapchain
        // extent × scale); TAA reconstructs full resolution. Clamped to
        // [0.25, 1.0]. Default 1.0. Issues a vkDeviceWaitIdle internally — not
        // callable from inside render().
        void setRenderScale(float scale);
        [[nodiscard]] float renderScale() const;

        // ── AMD FidelityFX FSR 3.1 upscaler runtime toggle ───────────────
        // Only meaningful when built with -DTHREEPP_WITH_FSR (Windows/Vulkan) and
        // the FSR context created (fsrAvailable() == true); otherwise this is a
        // no-op and the built-in TAA temporal upsampler always runs. Frame-to-frame
        // switchable (no device idle) — flipping resets the temporal history so the
        // switched-to path doesn't inherit the other's accumulation. Default on.
        void setFsr(bool enabled);
        [[nodiscard]] bool fsr() const;          // FSR is the active upscaler now
        [[nodiscard]] bool fsrAvailable() const; // compiled in + context created

        // ── NVIDIA DLSS Super Resolution runtime toggle ──────────────────
        // Only meaningful when built with -DTHREEPP_WITH_DLSS (Windows/Vulkan)
        // and the NGX feature created on an RTX GPU (dlssAvailable() == true);
        // otherwise a no-op. DLSS OUTRANKS FSR when both are available and
        // enabled; setDlss(false) hands the frame back to FSR (if on) or the
        // built-in TAA. Frame-to-frame switchable — flipping resets the
        // temporal history. Default on.
        void setDlss(bool enabled);
        [[nodiscard]] bool dlss() const;          // DLSS is the active upscaler now
        [[nodiscard]] bool dlssAvailable() const; // compiled in + feature created

        // Material-texture anisotropic filtering. Default AUTO (0): 16× when
        // the raster is UNJITTERED (setGbufferMsaa>1 without an upscaler,
        // event camera) — sharpness has no temporal cost there — and
        // ISOTROPIC trilinear whenever the raster is JITTERED (built-in TAA,
        // DLSS, FSR). Rationale: anisotropic filtering re-sharpens
        // grazing-angle textures back to pixel frequency, which no temporal
        // resolve can hold still — measured as the dominant carrier of the
        // "whole scene shimmers at a distance" residual on terrain- and
        // Bistro-class content; isotropic LOD selection prefilters it into
        // stability (a mild grazing-angle softness trade). Pass 1..16 to
        // force a fixed level in every mode; 0 restores AUTO. Runtime-safe.
        void setTextureAnisotropy(float aniso);
        [[nodiscard]] float textureAnisotropy() const;// the override; 0 = auto

        // Selects the demodulated filtered-lighting pipeline. Default ON.
        //   ON  — the stochastic 1-spp GI is cleaned by an SVGF variance-guided
        //         à-trous filter, direct shadows use a denoised soft-shadow
        //         visibility ratio, and the traced reflection is turned into a
        //         roughness-driven gloss by the reflection reconstruction pass.
        //   OFF — the inline DETERMINISTIC path: deterministic AO/GI, inline
        //         shadowed light loops, and SHARP mirror reflections (rough
        //         metals lose their gloss). This toggle therefore changes MATERIAL
        //         APPEARANCE, not just noise.
        // Intended as an A/B discriminator between the two shading paths
        // (equivalent to THREEPP_DENOISE=0), NOT a quality/perf dial. Probe GI
        // requires it ON.
        void setDenoise(bool enabled);
        [[nodiscard]] bool denoise() const;

        // Deprecated alias of setDenoise()/denoise().
        void setDeferredDenoise(bool enabled);
        [[nodiscard]] bool deferredDenoise() const;

        // Per-NEE-sample firefly clamp used by the deferred highlight suppress.
        // Default 30.0; 0 disables (1e30 sentinel).
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // Angular RADIUS (degrees) of directional lights for the deferred
        // renderer's soft sun shadows: the primary shadow ray is jittered within
        // this cone, so thin occluders (slats, railings, foliage) cast a narrow
        // stable penumbra instead of a per-frame hard-shadow coin flip that TAA
        // cannot converge. The real sun subtends ~0.27°; default 0.5. 0 restores
        // the exact hard 1-ray shadow.
        void setSunAngularRadius(float degrees);
        [[nodiscard]] float sunAngularRadius() const;

        // ReSTIR DI master toggle (streaming RIS + temporal + spatial reuse at
        // primary surfaces) for the deferred shade's next-event estimation. Off
        // (default) falls back to per-light NEE.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // Normal-map vMF/Toksvig specular AA (deferred G-buffer raster path).
        // Recovers the normal-map minification variance already baked into a filtered
        // (mip/trilinear) tap's shortened vector length and folds it into the
        // G-buffer roughness BEFORE the geometric spec-AA pass reads it, so
        // a high-frequency normal map shading a rough dielectric doesn't
        // moire/shimmer under TAA jitter at a distance. No-op at mip 0
        // (nLen ~= 1) and inert on materials without a normal map. ON by
        // default (strictly-better, no-op-when-unused); setter is a manual
        // override / debug escape.
        void setNormalMapToksvig(bool enabled);
        [[nodiscard]] bool normalMapToksvig() const;

        // ── Automatic mesh LOD (ON by default) ────────────────────────────
        // Per-geometry chains of simplified INDEX buffers (vertex positions/
        // normals/UVs untouched — index-only), generated in the background
        // with meshoptimizer and selected per-entry per-frame by projected
        // screen-space error (sub-pixel at the switch, so transitions are
        // invisible). Consumed identically by the raster G-buffer and the
        // ray-tracing TLAS. This is a PERFORMANCE feature for geometry-bound
        // scenes (measured: fjord flythrough +32% FPS, per-pixel-bound
        // Bistro neutral) — measurement showed it does NOT meaningfully
        // reduce TAA edge shimmer (that is sub-pixel coverage + the 1-spp
        // shading floor, not triangle density).
        // Exempt (always LOD0): skinned/displaced/grass/morphed/tet meshes,
        // overlay/particle entries, emissive materials, meshes with
        // Object3D::autoLod == false (self-managed LOD — TileTerrain sets it
        // on its quadtree tiles), and anything under a manual threepp::LOD
        // node (its levels are hand-authored already).
        // setAutoLod(false) is the manual override / debug escape. Toggle
        // semantics: takes effect at the next render() (every entry snaps
        // back to full detail on disable); generated chains and in-flight
        // background jobs are KEPT across a disable/enable cycle (results are
        // geometry-version-guarded, so nothing stale is ever consumed), and
        // autoLodStats() reflects the most recent render, not the setter.
        void setAutoLod(bool enabled);
        [[nodiscard]] bool autoLod() const;

        // Auto-LOD screen-space error budget τ, in RENDER-scale pixels.
        // Selection picks the coarsest level whose projected simplification
        // error stays under τ (hysteresis raises a level only under 0.8·τ).
        // Default 0.75 px — validated visually lossless under TAA. Raising
        // it is the perf lever: 1.5-2 px trades (mostly sub-animation-noise)
        // silhouette error for triangle throughput in geometry-bound scenes.
        // Clamped to [0.1, 8]; takes effect at the next render().
        void setAutoLodError(float px);
        [[nodiscard]] float autoLodError() const;

        // Debug/harness stats snapshot from the last render()'s LOD
        // selection pass. entriesPerLevel[0] is LOD0 (unsimplified); [1..4]
        // are the generated chain levels; [5] is a defensive catch-all
        // beyond the 4-level cap (should always read 0).
        struct AutoLodStats {
            uint32_t entriesPerLevel[6] = {};
            uint64_t indexBytes = 0;// resident LOD index-buffer bytes (all levels, all geometries)
            uint64_t blasBytes  = 0;// resident LOD BLAS storage bytes
            uint32_t chainsReady  = 0;// unique geometries with a finalized chain
            uint32_t chainsQueued = 0;// unique geometries with a chain job in flight
        };
        [[nodiscard]] AutoLodStats autoLodStats() const;

        // HDR bloom, added in linear HDR before the tone-map curve. 0 disables.
        void setBloomIntensity(float intensity);
        [[nodiscard]] float bloomIntensity() const;

        // Bloom bright-pass cutoff in linear-HDR luma.
        void setBloomThreshold(float threshold);
        [[nodiscard]] float bloomThreshold() const;

        // Bloom input clamp in linear-HDR luma. <= 0 disables.
        void setBloomClamp(float clampMax);
        [[nodiscard]] float bloomClamp() const;

        // Post-TAA RCAS sharpen strength. 0 disables.
        void setSharpenStrength(float amount);
        [[nodiscard]] float sharpenStrength() const;

        // Camera motion blur (post-TAA, per-pixel motion vectors incl.
        // skinning/deformation). The value is the shutter open fraction of
        // the frame interval — 0.5 is the filmic 180° shutter, 1.0 a full
        // frame of smear. 0 (default) disables the passes entirely.
        void setMotionBlur(float shutterFraction);
        [[nodiscard]] float motionBlur() const;

        // ── Physical camera exposure ─────────────────────────────────────
        // Derive exposure from real camera parameters instead of
        // toneMappingExposure:
        //   EV100 = log2(aperture²/shutter · 100/iso) − evCompensation
        //   exposure = 1 / (1.2 · 2^EV100)
        // Defaults (f/16, 1/125 s, ISO 100) are the sunny-16 daylight
        // setting: a 100,000 lux sun-lit scene lands at mid-gray. Pair with
        // setPhysicalLightUnits so intensities are real photometric units.
        // While enabled the renderer PRE-EXPOSES the HDR scene target
        // (stores are scaled by the exposure) so fp16 never overflows at
        // daylight magnitudes; bloomThreshold then operates on exposed
        // values (~1.0 = a few stops above mid-gray). toneMappingExposure
        // is ignored; the deferred renderer's auto-exposure composes on top
        // as EV compensation around the physical exposure. OFF by default.
        void setPhysicalCamera(bool enabled);
        [[nodiscard]] bool physicalCamera() const;

        // Camera exposure triplet (effective while physicalCamera is on):
        // aperture = f-number N, shutter in seconds, iso = sensitivity.
        void setCameraExposure(float aperture, float shutterSeconds, float iso);
        struct CameraExposure {
            float aperture;
            float shutterSeconds;
            float iso;
        };
        [[nodiscard]] CameraExposure cameraExposure() const;

        // Exposure compensation in EV (+1 doubles screen brightness);
        // effective while physicalCamera is on. Default 0.
        void setExposureCompensation(float ev);
        [[nodiscard]] float exposureCompensation() const;

        // ── Physical light units ─────────────────────────────────────────
        // When enabled, analytic light intensities are photometric:
        //   DirectionalLight.intensity = lux (sunlight ≈ 100,000)
        //   PointLight.intensity = lumens (candela = Φ/4π at upload)
        //   SpotLight.intensity  = lumens (Φ/π — Frostbite's convention:
        //                          brightness invariant under cone edits)
        //   RectAreaLight.intensity / emissive = nits (used as-is)
        // Default off: intensities keep the legacy arbitrary units.
        void setPhysicalLightUnits(bool enabled);
        [[nodiscard]] bool physicalLightUnits() const;

        // ── White balance + colour grade (post composite) ────────────────
        // Scene-illuminant white balance: temperatureK on the Planckian
        // locus (1667–25000 K; 6500 = neutral/D65 = off), tint shifts
        // green (−) / magenta (+), ±1 ≈ ±0.05 in CIE y. Bradford-adapted
        // to D65 in linear space before the tone curve.
        void setWhiteBalance(float temperatureK, float tint = 0.0f);
        // {temperatureK, tint} as last set (default {6500, 0} = neutral).
        [[nodiscard]] std::pair<float, float> whiteBalance() const;

        // Lift/gamma/gain colour wheels + saturation + contrast, applied to
        // the tone-mapped sRGB-encoded result via a 33³ LUT baked on the
        // CPU at set time (call at UI rate, not per frame). Defaults are
        // identity = off.
        struct ColorGrade {
            Vector3 lift{0.f, 0.f, 0.f};  // adds toward shadows
            Vector3 gamma{1.f, 1.f, 1.f}; // mid-tone power (per channel)
            Vector3 gain{1.f, 1.f, 1.f};  // scales toward highlights
            float saturation = 1.f;
            float contrast   = 1.f;// pivot 0.5 (display-referred)
        };
        void setColorGrade(const ColorGrade& grade);

        // ── Depth of field (thin lens, post) ─────────────────────────────
        // Half-res near/far scatter-as-gather bokeh on the HDR scene before
        // bloom/TAA (defocused highlights still bloom). The circle of
        // confusion is CAMERA-derived, not a strength slider: aperture is
        // setCameraExposure's f-number (the aperture drives bokeh size even
        // while physicalCamera is off — exposure and DoF are independent
        // consumers of the same triplet), focal length comes from the
        // camera's vertical FOV on a 24 mm full-frame sensor, and the focus
        // plane sits at setFocusDistance. Wider aperture (smaller f-number)
        // or a longer lens (narrower FOV) → shallower depth of field, as on
        // a real camera. OFF by default (zero cost).
        void setDepthOfField(bool enabled);
        [[nodiscard]] bool depthOfField() const;

        // Focus plane distance in scene units/meters (default 10).
        void setFocusDistance(float meters);
        [[nodiscard]] float focusDistance() const;

        // ── Two-phase GPU occlusion culling ──────────────────────────────
        // Splits the raster G-buffer into: draw last frame's visible set →
        // build a farthest-depth pyramid from it → test every object's
        // world AABB on the GPU → draw only the newly visible. Hidden
        // objects stop paying vertex/fragment cost entirely, with no
        // popping: a wrongly-skipped object is caught by the test and drawn
        // in phase 2 of the SAME frame (camera cuts cost one slow frame,
        // never a wrong one). Deformers (skinned/displaced/soft-body) always
        // draw — their CPU-side bounds are stale. Works with setGbufferMsaa
        // (the depth pyramid reduces the raw MS attachment's samples). Wins
        // scale with how much geometry is actually hidden (interiors, city
        // blocks, walls); scenes with little occlusion pay two small compute
        // dispatches + a depth-pyramid build. OFF by default. Ignored while
        // split-screen scissor is active.
        void setOcclusionCulling(bool enabled);
        [[nodiscard]] bool occlusionCulling() const;

        // ── PhysX soft-body zero-copy interop (CUDA → Vulkan) ────────────────
        struct SoftBodyInteropHandle {
            void*  osHandle  = nullptr;
            size_t sizeBytes = 0;
        };
        SoftBodyInteropHandle enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy);
        void disableSoftBodyInterop(const Mesh& mesh);

        // Hybrid-mode raster overlay: post-TAA wireframe / Line / layer-tagged
        // meshes drawn over the shaded image, depth-tested against the raster
        // G-buffer. -1 (default) disables layer selection.
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        // Day-1 / debug visualization: blit one G-buffer channel to the swapchain.
        //   0 = off, 1 = normal, 2 = motion, 3 = instance id, 4 = albedo
        void setHybridDebugView(int view);
        [[nodiscard]] int hybridDebugView() const;

        // ── Path-traced LIDAR scanner ─────────────────────────────────────
        // Synchronously trace beams against the same TLAS, evaluate a back-scatter
        // LIDAR equation at the first hit, and return per-beam tuples. Submits its
        // own command buffer + fence and blocks until results come back.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& results,
                       const LidarParams& params = {});

        // Per-frame timings (milliseconds). See FrameTimings.
        struct FrameTimings {
            float pathTraceMs    = 0.f;// deferred shade compute
            float denoiseMs      = 0.f;// deferred SVGF denoise passes
            float taaMs          = 0.f;// TAA resolve compute
            float rasterGbufMs   = 0.f;// raster G-buffer prepass
            float gbufResolveMs  = 0.f;// MSAA dominant-sample resolve (0 unless setGbufferMsaa > 1)
            float shadeBMs       = 0.f;// MSAA dispatch B: per-sample edge shading (0 unless setGbufferMsaa > 1)
            float overlayMs      = 0.f;// hybrid overlay depth + draw
            float dofMs          = 0.f;// thin-lens depth of field (0 unless setDepthOfField)
            float cpuEnsureSceneMs = 0.f;// ensureSceneBuilt
            float cpuRecordMs      = 0.f;// recordCommandBuffer
            float cpuFrameMs       = 0.f;// total render() wall time
        };
        [[nodiscard]] FrameTimings lastFrameTimings() const;

    protected:
        VulkanRendererCore() = default;

        // The shared implementation struct. Defined in VulkanRenderer.cpp; each
        // leaf's pImpl derives from it. Opaque to callers of this header.
        struct CoreImpl;

        // Hands the shared base a pointer to the leaf's CoreImpl sub-object so
        // shared behaviour can run without knowing the concrete leaf type.
        [[nodiscard]] virtual CoreImpl* coreImpl() const = 0;
        [[nodiscard]] CoreImpl* core() const { return coreImpl(); }

        // dispose() frees the leaf-owned pImpl; the leaf implements this because
        // it owns the unique_ptr.
        virtual void disposeImpl() = 0;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERERCORE_HPP
