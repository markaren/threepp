// Vulkan deferred renderer. Built against the Vulkan SDK and requires
// VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure (ray-queried
// AO/GI) and VK_KHR_ray_query.
//
// Shades a clean, analytic, noise-free base from the raster material G-buffer
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// plus ray-queried ambient occlusion / global illumination - interactive and
// noise-free, the default for synthetic-perception work.
//
// Owns (through pImpl) all of the Vulkan infrastructure: device/swapchain
// context, acceleration structures, scene build + visibility, material/geometry
// buffers, the raster G-buffer prepass, the deferred shade, and the bloom / TAA
// / overlay post-stack.
//
// Co-exists with GLRenderer; selected by the application when a Canvas is
// created with GraphicsAPI::Vulkan.

#ifndef THREEPP_VULKANRENDERER_HPP
#define THREEPP_VULKANRENDERER_HPP

#include "threepp/cameras/LensDistortion.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace threepp {

    class Mesh;
    class ParticleField;

    class VulkanRenderer: public Renderer {

    public:
        explicit VulkanRenderer(Canvas& canvas);

        ~VulkanRenderer() override;

        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

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
        //   SplatDepth (4) 1x float32 : Gaussian-splat VIEW DISTANCE in world
        //                  units (positive, NOT reversed-Z NDC), 0 where no
        //                  splat cloud owns the pixel. Readable whenever the
        //                  AOV is ALLOCATED — setSplatDepthAov, or the
        //                  renderer's own overlay-occlusion latch, which turns
        //                  it on (as Median) the first frame a scene holds both
        //                  clouds and overlay content. Which statistic it
        //                  carries is setSplatDepthAov's mode, Median when the
        //                  latch is the only reason it exists.
        enum class GBufferAOV { Depth, Normal, Motion, Ids, Albedo, SplatDepth };
        [[nodiscard]] bool readGBufferAOV(GBufferAOV aov, std::vector<uint8_t>& out,
                                          int& width, int& height, int& bytesPerPixel);

        // Batched form: read SEVERAL attachments of the same frame with ONE
        // device wait, one command buffer (N copy regions) and one staging
        // allocation, instead of paying the full drain per AOV — the multi-AOV
        // dataset path calls this once per captured frame. `out` receives one
        // entry per readable requested AOV (duplicates collapsed, unreadable
        // ones — e.g. SplatDepth without setSplatDepthAov — omitted); match by
        // the entry's `aov` field, layouts exactly as documented above. Returns
        // false when nothing could be read at all.
        struct AOVReadback {
            GBufferAOV aov{};
            std::vector<uint8_t> data;
            int width = 0, height = 0, bytesPerPixel = 0;
        };
        [[nodiscard]] bool readGBufferAOVs(const std::vector<GBufferAOV>& aovs,
                                           std::vector<AOVReadback>& out);

        // ── ParticleField density volume readback (TEST / DEBUG) ─────────
        // Copy a ParticleField's world-space density volume back to the host as
        // raw FIXED-POINT sigma_t: `out` is resolution^3 uint32 in x-fastest
        // order, each value sigma_t * 4096 (12 fractional bits — see
        // shaders/particle_density.glsl for why that scale).
        //
        // This exists because the density representation's headline guarantee is
        // DETERMINISM — the volume is accumulated with integer atomics, whose
        // adds are associative, so the same particles produce the same bits
        // however the GPU orders them — and that guarantee is not checkable
        // through the rendered image, whose GI/ReSTIR/TAA are stochastic per
        // frame index by design. Reading the volume is what turns the claim into
        // an assertion.
        //
        // DRAINS THE DEVICE (vkDeviceWaitIdle) and allocates a staging buffer of
        // resolution^3 * 4 bytes. Not a per-frame call. Returns false when the
        // field has no volume (density representation off, or never rendered).
        [[nodiscard]] bool readParticleDensityVolume(const ParticleField& field,
                                                     std::vector<uint32_t>& out,
                                                     uint32_t& resolution);

        // ── Gaussian-splat depth AOV ─────────────────────────────────────
        // A splat cloud is composited by a compute tile rasterizer and is in no
        // acceleration structure, so it writes no G-buffer depth and every
        // ray-traced consumer — the RT sensors, reflections, shadows — passes
        // straight through it. The raster already accumulates the alpha-weighted
        // expected view distance per pixel (it is what makes the cloud's motion
        // vectors computable); this switch exports it.
        //
        // What Expected is: for each pixel a cloud owns, sum(dist * alpha * T) /
        // (1 - T) — the depth of the cloud's opacity centroid along that ray.
        // Only where accumulated coverage exceeds 0.5, because below that the
        // geometry behind the translucent fringe is the better answer; the rest
        // reads 0. Nearest cloud wins where several overlap.
        //
        // What Expected is NOT: a surface. The expected value sits behind the
        // visible front of a splat by roughly the cloud's own thickness along
        // the ray, so it localizes a wall well and a foliage canopy poorly. For
        // picking and for coarse occupancy it is the right number; for metric
        // ranging against thin structure it is biased, knowingly.
        //
        // Median is the unbiased-for-surfaces statistic: the view distance at
        // which accumulated transmittance crosses 0.5, interpolated between the
        // two splats that straddle the crossing. Same coverage gate, same
        // nearest-cloud-wins rule, same image — only the statistic changes, and
        // only for the AOV (motion vectors and per-splat fog keep using the
        // expected value they have always used). It is what depth fusion wants
        // (plans/splat-surface-bake.md).
        //
        // OFF by default and a SETUP knob: turning the AOV on or off
        // reallocates the render targets (a full-res r32f per frame in flight),
        // so set it once before the render loop rather than per frame. Changing
        // only the statistic is a per-frame flag and reallocates nothing.
        // Primary view only — splats are not drawn into secondary views at all.
        enum class SplatDepthMode { Off, Expected, Median };
        void setSplatDepthAov(SplatDepthMode mode);
        // Expected, the statistic every pre-mode caller got.
        void setSplatDepthAov(bool enabled);
        [[nodiscard]] bool splatDepthAov() const;
        [[nodiscard]] SplatDepthMode splatDepthAovMode() const;

        // ── Multi-view: N cameras per frame ──────────────────────────────
        // A camera rig — a robot's cameras, a multi-sensor capture setup —
        // wants several viewpoints of the SAME simulated instant. Rendering
        // them by calling render() N times gives N different instants and pays
        // N times for one scene. addView instead attaches a persistent extra
        // view: every render() then produces the primary AND every added view
        // from one scene build, in a single queue submission.
        //
        // Each view gets its own G-buffer, its own temporal history and its own
        // camera state, so one camera cutting or moving cannot smear another.
        // World-space work — acceleration structures, lights, materials,
        // textures, probe GI — is built once and shared.
        //
        // Views are PERSISTENT. addView is expensive (it drains the device and
        // allocates a full deferred chain, reporting the cost); rendering an
        // existing view every frame is not. Do not add and remove per frame.
        //
        // Secondary views are deliberately plainer than the primary: native
        // resolution with the built-in temporal resolve, no DLSS/FSR, no
        // occlusion culling, no UI overlay, no depth of field, no lens or
        // sensor model. They are measurement cameras, not the display.
        //
        // Returns a handle (> 0), or 0 if the view could not be created —
        // notably when render() has not run yet, since a view shares the
        // primary's render pass and pipelines.
        uint32_t addView(Camera& camera, int width, int height);
        // Destroys the view and frees everything it owns. Returns false for an
        // unknown handle. Handles are never reused, so a stale one is inert
        // rather than dangerous.
        bool removeView(uint32_t handle);
        // Repoint a view at a different camera. Treated as a CUT: the view's
        // temporal history is dropped rather than reprojected across a
        // discontinuity that never happened in world space.
        bool setViewCamera(uint32_t handle, Camera& camera);
        // This view's most recent frame as tightly-packed RGB8, row-major,
        // TOP-LEFT origin — the same convention as readRGBPixels (the Vulkan
        // readback is already top-down; there is no flip). Empty on an unknown
        // handle. Reads the view's own colour image, never the swapchain.
        [[nodiscard]] std::vector<unsigned char> readViewRGBPixels(uint32_t handle);
        // Pixel size of a view's output, as passed to addView. False if unknown.
        bool viewSize(uint32_t handle, int& width, int& height) const;

        // Show this view inside the primary's frame, with its top-left corner
        // at (x, y) in window pixels. The view's image is already resolved, on
        // the device and in the swapchain's format, so this is a single image
        // copy in the frame's own command buffer — no readback, no upload, no
        // texture, and no second submission.
        //
        // 1:1 only: `width`/`height` must equal the size the view was added at,
        // and a mismatch draws NOTHING rather than a filtered rescale. A
        // secondary resamples an image the view's temporal resolve already
        // produced exactly once, and sizing the view to its rect costs nothing.
        // Pass a rect that runs off the window edge and it is clipped.
        //
        // Composited after the scene capture (which stays a clean picture of
        // the primary camera alone) and before the UI overlay, so ImGui and
        // screen-space sprites still draw on top of it. False on an unknown
        // handle.
        bool setViewDisplayRect(uint32_t handle, int x, int y, int width, int height);
        // Back to a measurement camera: still rendered, still readable, no
        // longer drawn into the frame.
        bool hideView(uint32_t handle);

        // readGBufferAOV for a specific view. Handle 0 means the primary, so
        // this is the general form and readGBufferAOV is the shorthand.
        //
        // This is what makes a multi-camera rig useful for training data rather
        // than just for looking at: every camera yields not only colour but
        // lossless depth and the Ids attachment, whose .y channel is a stable
        // per-object instance id and whose .z carries the 8-bit semantic class
        // (see setObjectInstanceId / setObjectClassId). One readback per camera
        // gives depth, instance segmentation and semantic segmentation ground
        // truth for the same simulated instant, from N viewpoints.
        //
        // A stale or unknown handle returns false rather than falling back to
        // the primary: labels attributed to the wrong camera are worse than no
        // labels. Same layout contract as readGBufferAOV — tightly packed,
        // row-major, top-left origin, native GPU format.
        [[nodiscard]] bool readViewGBufferAOV(uint32_t viewHandle, GBufferAOV aov,
                                              std::vector<uint8_t>& out,
                                              int& width, int& height, int& bytesPerPixel);

        // Batched per-view form — the implementation the two single-AOV reads
        // above are a batch-of-one of. Same contract as readGBufferAOVs.
        [[nodiscard]] bool readViewGBufferAOVs(uint32_t viewHandle,
                                               const std::vector<GBufferAOV>& aovs,
                                               std::vector<AOVReadback>& out);

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
        // What the detector looks at.
        //   Shaded — a deterministic Lambert proxy shaded from the raster
        //            G-buffer (event_shade.comp): directional lights + ambient
        //            + emissive, metalness-damped; no specular, shadows, GI,
        //            transmission or point/spot lights. Cheap and noise-free
        //            (raster jitter is gated off so a static scene emits
        //            nothing), but it only sees silhouettes and diffuse
        //            texture — water glitter, backlit cloth and light flashes
        //            do not fire. The default, and what eventsOnlyMode uses.
        //   Final  — the presented frame: post-TAA/upscale/tonemap, the same
        //            pixels readRGBPixels returns, box-averaged down to the
        //            sensor resolution (a DVS pixel integrates its photodiode
        //            area). Everything the picture shows fires, at the cost of
        //            inheriting the picture's temporal residue (denoiser,
        //            auto-exposure drift). Raster jitter stays ON (TAA resolves
        //            it). Ignored — Shaded is used — while eventsOnlyMode is
        //            on, since no final frame exists then.
        // Switching while enabled re-latches the per-pixel reference on the
        // next frame (which then emits nothing) instead of firing a
        // whole-frame burst on the proxy-vs-final luma jump. Flip it at a cut.
        enum class EventCameraSource { Shaded, Final };
        void setEventCameraEnabled(bool enabled);
        [[nodiscard]] bool eventCameraEnabled() const;
        void setEventCameraSource(EventCameraSource source);
        [[nodiscard]] EventCameraSource eventCameraSource() const;
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

        // ── Deterministic frame clock ─────────────────────────────────────
        // Pins every wall-clock read the frame path makes — the TAA blend dt,
        // the shade's animation timeSec, DLSS/FSR frame deltas, ocean foam
        // decay, deform timestamps, the cloud clock — to an app-supplied
        // simulation time. Call once per frame BEFORE render() with a
        // monotonically non-decreasing value; stepping it by a fixed dt makes
        // the rendered output replayable bit-for-bit across runs (raster AOVs
        // already are; the beauty frame additionally needs this because its
        // temporal-blend weights are otherwise functions of real frame time).
        // Negative disables and returns to the wall clock (the default).
        // Sensor pipelines should drive this with the same sim clock that
        // stamps their measurements (see extras/sensors/Sensor.hpp).
        void setSimTime(double seconds);

        // Hard restart of every temporal accumulator, on every live view: the
        // TAA history, the reprojection validity of the deferred shade's
        // per-frame histories (GI, shadow, reflections, volumetric fog) and the
        // ReSTIR reservoirs. The next render() starts them from nothing, as the
        // first frame does. Meant for the start of a capture once a scene has
        // finished STREAMING: terrain tiles that land at run-dependent frames
        // put run-varying content into every history, and an exponential
        // average never returns to bit-equality within float precision, so a
        // replayable capture restarts the histories after the scene is stable.
        // Costs one frame of temporal convergence. Takes effect on the next
        // render().
        void resetTemporalHistory();
        [[nodiscard]] double simTime() const;

        // Debug/audit readback of the temporal-resolve endpoints for the LAST
        // completed frame: `input` = the TAA input image (the shade → bloom →
        // post-composite product; BGRA8 at the render extent), `history` = the
        // history slot that frame wrote (RGBA16F at the output extent).
        // Splits "the shading diverged" from "the temporal resolve diverged"
        // in the determinism audit (examples/vulkan/vulkan_aov_audit.cpp).
        // Full device sync per call — an audit instrument, not a capture path.
        bool readTaaDebugImages(std::vector<uint8_t>& input, int& inW, int& inH,
                                std::vector<uint8_t>& history, int& histW, int& histH);

        // Same instrument one stage earlier: the linear-HDR scene image the
        // shade/denoise chain wrote this frame (bloom's sceneHdr; RGBA16F at
        // the render extent), BEFORE bloom and the post composite touch it.
        // taa.input diverging while sceneHdr is exact indicts bloom/post;
        // sceneHdr diverging under --no-denoise indicts the shade dispatch
        // itself. Full device sync per call — audit instrument, not capture.
        bool readSceneHdrDebug(std::vector<uint8_t>& hdr, int& w, int& h);

        // The finest split: FNV-1a hash of each deferred-shade temporal image
        // for the last completed frame — {indirect, momentsSq, reflect,
        // reflAux, shadowVis, directU}. The first name whose hash differs
        // between two same-seed runs is the pass the divergence enters at.
        // directU is the control: analytic direct light, no rays, no history —
        // if IT diverges the shade dispatch itself is non-deterministic.
        // Empty result before the first frame. Full device sync per call.
        std::vector<std::pair<std::string, uint64_t>> debugHashShadeImages();

        // Raw dump of the probe-GI SH-L1 store (kProbeCount × 4 × vec4) for
        // byte-level divergence forensics: which probe, which SH band, how
        // large. Audit instrument; full device sync per call.
        bool readProbeShDebug(std::vector<uint8_t>& sh);

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

        // ── Orthographic camera: 3D view or 2D overlay? ──────────────────
        // An OrthographicCamera means two different things to this backend. By
        // default (false) a standalone render() with one is the HUD/2D path:
        // the scene's Sprites, Lines, Points and Meshes are drawn as flat
        // unlit fills over the swapchain — what an SVG layer or a screen-space
        // overlay wants, and what every 2D user of this renderer already gets.
        //
        // Set true when the ortho camera is a real 3D VIEW (an editor's axis
        // views, an isometric game camera). The frame then takes the same
        // deferred path a perspective camera does — G-buffer, analytic lights,
        // ray-traced shadows, GI, reflections, fog, tone mapping — so the two
        // projections shade identically instead of one falling back to flat
        // colour. Parallel rays are reconstructed per pixel, which costs one
        // uniform-branch in the shading passes and nothing when off.
        //
        // Only the STANDALONE render() call is affected. The HUD pattern (a
        // perspective render(), then a second render() with an ortho camera
        // over a HUD scene) still composes overlay-only onto the open frame
        // regardless of this flag. Depth of field is skipped under an ortho
        // camera: a parallel projection has no lens, so there is no circle of
        // confusion to compute.
        void setOrthographicSceneRendering(bool enabled);
        [[nodiscard]] bool orthographicSceneRendering() const;

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

        // Material-texture anisotropic filtering. Default AUTO (0), which is
        // 16× in EVERY mode — jittered or not. Pass 1..16 to force a fixed
        // level (1 = isotropic trilinear); 0 restores AUTO. Runtime-safe, and
        // also settable at startup with THREEPP_VK_ANISO=<n>.
        //
        // An earlier AUTO policy dropped to isotropic whenever the raster was
        // jittered (built-in TAA, DLSS, FSR), on the theory that re-sharpening
        // grazing-angle detail back to pixel frequency was the dominant carrier
        // of the "whole scene shimmers at a distance" residual. Later triage
        // attributed that shimmer to other sources; the isotropic fallback only
        // mip-blurred ground and facade textures at distance for no stability
        // gain, so it was withdrawn. setTextureAnisotropy(1) restores it.
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
        // primary surfaces) for the deferred shade's next-event estimation.
        // ON by default — it is what makes many-light and emissive-geometry
        // scenes converge at 1 spp, so it is the path the renderer is tuned and
        // golden-tested against. setRestirDIEnabled(false) bypasses the primary
        // RIS branch and falls back to the legacy per-light NEE loops: cheaper
        // with a handful of lights, markedly noisier with many, and a useful
        // A/B when triaging a reservoir-feedback artifact.
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

        // Debug/harness stats for the graduated per-frame dynamic-geometry
        // path (a plain mesh whose attributes are rewritten + needsUpdate()ed
        // every frame graduates to a frame-cb BLAS refit after a short dirty
        // streak). Counters are cumulative since renderer construction. The
        // determinism audit asserts they MOVED — a scene that was supposed to
        // exercise the refit path but didn't would otherwise certify nothing.
        struct DynamicGeomStats {
            uint32_t graduated = 0;     // records promoted to the per-frame path
            uint64_t refitsRecorded = 0;// BLAS ops recorded into frame command buffers
            uint64_t fullRebuilds = 0;  // of those, periodic MODE_BUILD rebalances
        };
        [[nodiscard]] DynamicGeomStats dynamicGeomStats() const;

        // Top-level acceleration structure build bookkeeping. Counters only —
        // nothing here feeds a decision. The distinction that matters to a
        // determinism audit is fullRebuilds vs updates: a MODE_BUILD re-derives
        // the TLAS from scratch and is free to order instances differently,
        // while a MODE_UPDATE refits the existing topology in place.
        struct TlasStats {
            uint64_t fullRebuilds = 0;// MODE_BUILD (structural + promoted refits)
            uint64_t updates = 0;     // MODE_UPDATE refits
            uint32_t instances = 0;   // instance count of the most recent build/update
        };
        [[nodiscard]] TlasStats tlasStats() const;

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

        // ── Camera intrinsics readout ────────────────────────────────────
        // The pinhole intrinsics the renderer is currently projecting with,
        // in PIXELS of the render extent — the same pixel grid readGBufferAOV
        // returns, which under setRenderScale is smaller than
        // framebufferSize(). Origin is top-left, matching both the AOV
        // readback and OpenCV, so these drop straight into a projectPoints /
        // undistort call.
        //
        // There is no setter: the sensor and lens live on the camera, where
        // threepp already models them — set a real camera the real way,
        //   cam.filmGauge = 6.3f;        // sensor WIDTH in mm (1/2.3")
        //   cam.setFocalLength(4.8f);    // lens in mm  → drives fov
        // and this reports back the fx/fy/cx/cy an OpenCV calibration of that
        // camera would produce. The depth-of-field CoC reads the same film
        // gauge, so exposure, framing and defocus finally share one sensor.
        //
        // Valid after the first render(); zero-sized before it.
        struct CameraIntrinsics {
            float fx = 0.f, fy = 0.f;// focal length in pixels
            float cx = 0.f, cy = 0.f;// principal point in pixels, top-left origin
            uint32_t width = 0, height = 0;
        };
        [[nodiscard]] CameraIntrinsics cameraIntrinsics() const;

        // ── Lens distortion ──────────────────────────────────────────────
        // Bend the image the way a real lens does, using the coefficients the
        // real lens was calibrated with. Models and conventions are OpenCV's
        // exactly (see cameras/LensDistortion.hpp), so the output of
        // cv::calibrateCamera — or a ROS camera_info — drops straight in:
        //
        //   LensDistortion d;
        //   d.model = LensModel::BrownConrady;
        //   d.k1 = -0.28f; d.k2 = 0.09f; d.p1 = 0.0006f;
        //   renderer.setLensDistortion(d);
        //
        // Applied to BOTH the displayed image and the sensor readback
        // (readGBufferAOV / readSceneRGBPixels), because a distorted colour
        // frame paired with undistorted depth/segmentation labels is worse
        // than no distortion at all. Integer AOVs (Ids) and depth resample
        // NEAREST — interpolating an instance id invents objects, and
        // interpolating depth across a silhouette invents surfaces.
        //
        // BORDER: the warp is a gather over what was actually rendered, and
        // the frustum still comes from the camera, so a pixel whose ideal ray
        // falls outside the rendered frame clamps to the frame edge and
        // smears. BARREL distortion (k1 < 0) is the case that hits this: it
        // maps scene points inward, so filling the output corners needs
        // content from OUTSIDE the rendered field (measured: k1 = -0.1 on a
        // 384x256 frame wants pixel (398, 265) at the corner). Pincushion
        // (k1 > 0) gathers inward and is unaffected. Until overscan exists,
        // render with a wider FOV than the lens' nominal one and accept the
        // reframing, or crop the smeared border, if the corners must be real.
        //
        // OFF by default (LensModel::None) — zero cost, output unchanged.
        void setLensDistortion(const LensDistortion& distortion);
        [[nodiscard]] LensDistortion lensDistortion() const;

        // Render the scene with the frustum widened by this factor so the warp
        // has real geometry to gather into the output corners, instead of the
        // clamped, smeared edge described above. 1 (default) = off.
        //
        // This is the fix for the barrel-distortion border: 1.15-1.3 covers
        // typical wide-lens coefficients. The cost is rendering a wider field
        // at the same resolution, so effective detail drops by roughly the
        // factor (a 1.25 overscan renders 1.25x the field into the same
        // pixels). Only meaningful with a lens set; ignored otherwise.
        //
        // The widened frustum applies to EVERYTHING this frame — culling,
        // motion vectors and the G-buffer AOVs included — so colour and labels
        // keep describing one camera. cameraIntrinsics() still reports the
        // OUTPUT camera (the one you configured), not the widened one.
        void setLensOverscan(float factor);
        [[nodiscard]] float lensOverscan() const;

        // ── Image-sensor noise ───────────────────────────────────────────
        // Photon and electronic noise, applied last (post-TAA/upscale — see
        // below). Modelling ISO as pure exposure gain, which is all the
        // physical camera did on its own, gets the brightness of a high-ISO
        // frame right and the character of it completely wrong; this closes
        // that gap for synthetic training data.
        //
        // Parameters are in ELECTRONS, the sensor's own units, so a datasheet
        // is enough to configure one: full-well capacity sets where shot
        // noise sits relative to saturation, read noise is the floor, dark
        // current is thermal, PRNU is the fixed pattern burned into the
        // silicon. ISO gain (from setCameraExposure) divides the electron
        // count, so higher ISO really does get noisier, by the physics rather
        // than by a slider.
        //
        // Deterministic: the same seed replays the same noise, so an episode
        // is reproducible. resetSensorNoise() restarts the sequence.
        //
        // Applied AFTER the temporal resolve by necessity — TAA and the
        // upscalers average successive frames, so noise injected earlier is
        // quietly filtered back out and the sim reports a noise model it is
        // not producing.
        //
        // OFF by default — zero cost, output unchanged.
        struct SensorNoise {
            bool  enabled = false;
            float fullWellElectrons = 20000.f;      // saturation capacity
            float readNoiseElectrons = 3.0f;        // RMS floor
            float darkCurrentElectronsPerSec = 5.0f;// × the exposure time
            float prnuPercent = 0.5f;               // fixed-pattern gain sigma, %
            uint32_t seed = 1u;
        };
        void setSensorNoise(const SensorNoise& noise);
        [[nodiscard]] SensorNoise sensorNoise() const;
        // Restart the noise sequence (call on episode reset so two episodes
        // with the same seed produce the same frames).
        void resetSensorNoise();

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
        // camera's vertical FOV on the camera's OWN sensor
        // (PerspectiveCamera::getFilmHeight — set filmGauge for anything
        // other than 35 mm film; see cameraIntrinsics), and the focus plane
        // sits at setFocusDistance. Wider aperture (smaller f-number) or a
        // longer lens (narrower FOV) → shallower depth of field, as on a
        // real camera; a SMALLER sensor at the same framing → deeper depth
        // of field, also as on a real camera (which is why phone cameras
        // need to fake it). OFF by default (zero cost).
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

        // ── GPU-particle zero-copy interop (CUDA → Vulkan), plan F6 ──────────
        // Export an Ownership::Interop ParticleField's positions allocation and
        // arm the per-frame device-to-device copy that fills it. Import the
        // returned handle once (CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 /
        // _OPAQUE_FD, CUDA_EXTERNAL_MEMORY_DEDICATED — the export is a dedicated
        // allocation) and copy straight out of the sim: for PhysX PBD that is
        // PxParticleBuffer::getPositionInvMasses(), and ParticlePos is
        // byte-identical to PxVec4, so it is one cuMemcpyDtoD with no repack.
        //
        // `deviceCopy` runs once per frame inside render(), before recording,
        // and MUST be synchronous (cuMemcpyDtoDAsync + cuStreamSynchronize) —
        // that host ordering is what sequences the foreign write against the
        // frame that reads it, in the absence of a shared semaphore.
        //
        // CALL IT AFTER THE FIRST render(): the field's device state and this
        // renderer's field pass are both created on the frame the field is
        // first seen. Same polling pattern as enableSoftBodyInterop; a null
        // handle means "not yet, or not on this device".
        //
        // Returns {} when the device has no external-memory extension. The
        // field is then in ParticleField::hostFallback() and wants submit(),
        // and the renderer has said so on stderr.
        struct ParticleFieldInteropHandle {
            void*  osHandle  = nullptr;
            size_t sizeBytes = 0;
            // Config::attributes: the field's per-particle vec4 appearance
            // buffer, exported by the SAME call and with the same layout (rgb =
            // linear HDR radiance, a reserved). Null when the field was created
            // without attributes. It is handed out here rather than through a
            // second entry point so that a field can never end up in a state
            // where the positions imported and the colours did not — the two
            // allocations are created together and exported together.
            void*  attrHandle    = nullptr;
            size_t attrSizeBytes = 0;
        };
        ParticleFieldInteropHandle enableParticleFieldInterop(ParticleField& field,
                                                              std::function<void()> deviceCopy);

        // ── Mesh-vertex zero-copy interop (CUDA → Vulkan) ────────────────────
        // Export a plain Mesh's BLAS position + normal buffers and arm the
        // per-frame device-to-device copy that fills them, so an external GPU
        // producer (NVIDIA Warp, PhysX, torch) can write `position`/`normal` in
        // place with no host round trip. Import the returned handles once
        // (CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 / _OPAQUE_FD,
        // CUDA_EXTERNAL_MEMORY_DEDICATED) and write straight into them; the
        // layout is tightly-packed float xyz, i.e. a 12-byte stride, NOT the
        // vec4-padded layout the soft-body tet path uses.
        //
        // What the renderer does with them: the exports are copy SOURCES that
        // no shader ever binds. Each frame the renderer runs `deviceCopy`, then
        // copies posBytes/nrmBytes into the mesh's own BLAS buffers and refits
        // the BLAS. It has to be a copy rather than a buffer substitution
        // because five of the seven consumers of a mesh's vertex buffer reach
        // it by device address, and an exported dedicated allocation cannot
        // carry one (it is allocated without VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT).
        //
        // `deviceCopy` runs once per frame inside render(), post-fence and
        // pre-record, and MUST be synchronous (e.g. wp.synchronize_device(), or
        // cuMemcpyDtoDAsync + cuStreamSynchronize) — that host ordering is what
        // sequences the foreign write against the frame that reads it, in the
        // absence of a shared semaphore.
        //
        // `validate` (default true) runs a small compute pass over the exported
        // positions before the copy, rewriting non-finite vertices into
        // zero-area degenerates. It replaces the CPU finiteness scan that
        // guards every other route into a BLAS build and cannot run here (the
        // host attribute array is stale under interop). Leave it on unless the
        // producer is trusted: a NaN reaching vkCmdBuildAccelerationStructures
        // is VK_ERROR_DEVICE_LOST on NVIDIA — a GPU reset, not an error return.
        //
        // CALL IT AFTER THE FIRST render(): the mesh's BlasRecord is created on
        // the frame the mesh is first seen. Same polling pattern as
        // enableSoftBodyInterop; a null handle means "not yet, or not on this
        // device", never an exception.
        //
        // FIXED-CAPACITY ALLOCATION, VARIABLE DRAW. A producer whose triangle
        // count varies allocates for its maximum once and publishes the live
        // count with `BufferGeometry::setDrawRange`: the raster path clamps to
        // the range, the BLAS is built over [0, start + count), and the interop
        // copies are trimmed to match. No zero-area degenerate tail is needed —
        // that was the contract before the mesh path honoured drawRange.
        // Changing the attribute *counts* after enabling still tears the
        // BLAS record down (and with it the allocation the foreign API imported),
        // so the renderer disables interop for that mesh with a warning instead.
        struct VertexInteropHandle {
            void*  posHandle = nullptr;
            size_t posBytes  = 0;
            void*  nrmHandle = nullptr;
            size_t nrmBytes  = 0;
        };
        // REGISTRATION-TIME call, not a per-frame one: arming (and disarming)
        // pays a device drain plus a full structural scene rebuild on the next
        // frame — toggling it per frame would stall the renderer hard. And
        // never call it from inside another mesh's deviceCopy callback: enable
        // may replace a blasCache record while the frame that invoked the
        // callback is iterating records, which is a use-after-free, not a
        // wrong picture.
        // stableCorrespondence: leave true when vertex i is the same surface
        // point every frame (a deforming fixed-topology mesh) — per-vertex
        // motion vectors then come from the previous frame's positions. Pass
        // FALSE for a producer that re-triangulates each frame (a marching-
        // cubes soup: one changed cell shifts every later vertex slot), where
        // that history is noise: the mesh then reprojects as world-static and
        // the temporal passes (TAA/upscaler, reflection denoiser) stop
        // flickering on the regions that changed.
        VertexInteropHandle enableVertexInterop(const Mesh& mesh,
                                                std::function<void()> deviceCopy,
                                                bool validate = true,
                                                bool stableCorrespondence = true);
        // Release the exports and return the mesh to the normal CPU-driven
        // attribute path. The caller must have stopped (or never started) the
        // foreign writes into the exported memory first.
        void disableVertexInterop(const Mesh& mesh);

        // ── Frame zero-copy interop (Vulkan → CUDA), "frames out" ────────────
        // The reverse direction of enableVertexInterop: instead of a foreign
        // producer writing geometry the renderer reads, the renderer publishes
        // what it just DREW into external-memory buffers a foreign consumer
        // reads. A vision-observation RL policy, an on-GPU dataset writer or a
        // learned post-process gets the frame as a device pointer, with no
        // staging buffer, no host memcpy and no vkDeviceWaitIdle — the three
        // costs readGBufferAOVs pays per call.
        //
        // What it is, mechanically: one exported TRANSFER_DST buffer per
        // requested channel, and one vkCmdCopyImageToBuffer per channel
        // recorded into the frame's OWN command buffer, at the point in the
        // frame where the source image is final (the scene-capture point for
        // Color, after every G-buffer consumer for the AOVs). No shader binds
        // the exports; they are copy destinations and nothing else. A frame
        // with no armed view records nothing extra at all.
        //
        // Channels. Color is the scene image — post-TAA, pre-overlay, the same
        // picture readSceneRGBPixels captures — in the SWAPCHAIN's format for
        // the primary and in the view's own colorTarget (same format) for a
        // secondary. `bgra` says which byte order that format is, since the
        // surface may not offer B8G8R8A8_UNORM and the context falls back. The
        // rest are the raster G-buffer attachments, byte-for-byte the images
        // readGBufferAOV copies, with the element layouts documented there:
        //   Color      (4)  4x unorm8  : bgra when `bgra`, else rgba
        //   Depth      (4)  1x float32 : reversed-Z NDC depth (1=near, 0=far)
        //   Normal     (8)  4x float16 : n*0.5+0.5 in xyz, roughness in w
        //   Motion     (8)  4x float16 : NDC motion in xy, prevDepth in z
        //   Ids        (8)  4x uint16  : visible index+1, meshId, flags|class
        //   Albedo     (4)  4x unorm8  : linear base colour in rgb, metal in a
        //   SplatDepth (4)  1x float32 : splat view distance; needs setSplatDepthAov
        // Rows are TIGHTLY PACKED (bufferRowLength 0), row-major, top-left
        // origin — an export is exactly width*height*bytesPerPixel of image,
        // and `sizeBytes` is the ALLOCATION size, which the driver may round up
        // past that. No conversion, no tonemapping, no lens warp: the AOV
        // HOST readback warps its output when a lens is set (readViewGBufferAOVs)
        // and this path deliberately does not, because the warp is a CPU
        // resample. Exports carry the pinhole G-buffer; a lensed pipeline that
        // needs matched labels stays on the host path.
        enum class FrameChannel { Color, Depth, Normal, Motion, Ids, Albedo, SplatDepth };

        struct FrameInteropExport {
            FrameChannel channel{};
            void*    osHandle = nullptr;// renderer-owned NT handle / fresh fd,
            size_t   sizeBytes = 0;     // same rules as VertexInteropHandle
            uint32_t width = 0, height = 0, bytesPerPixel = 0;
            bool     bgra = false;// Color only: the swapchain is B8G8R8A8
        };

        // REGISTRATION-TIME call, after the first render(): the G-buffer and
        // the swapchain images are allocated by then and their extents are what
        // the exports are sized from. Returns one entry per exportable
        // requested channel — duplicates collapse, unreadable channels are
        // SKIPPED rather than failing the batch (SplatDepth without
        // setSplatDepthAov; Color on a surface without TRANSFER_SRC swapchain
        // usage), so match entries by their `channel` field. An EMPTY vector
        // means nothing could be exported at all: before the first render, on a
        // stale view handle, or on a device without the external-memory
        // extension — in which case one line says so on stderr and the fallback
        // is the readGBufferAOVs host path everyone already has.
        //
        // Re-enabling a view that is already armed replaces its channel set and
        // returns fresh handles for the SAME allocations (Windows) or freshly
        // minted fds (POSIX), so a consumer may re-import without a disable.
        //
        // Import the handles once with CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 /
        // _OPAQUE_FD and CUDA_EXTERNAL_MEMORY_DEDICATED (the exports are
        // dedicated allocations) — python/threepp/cuda_interop.py does exactly
        // that, and threepp.torch_frames wraps it as torch tensors.
        std::vector<FrameInteropExport> enableFrameInterop(uint32_t viewHandle,
                                                           const std::vector<FrameChannel>& channels);
        // Release this view's exports. Windows NT handles are closed here, so
        // the consumer must have released its imports FIRST — freeing the
        // Vulkan allocation under a live CUDA mapping reports as nothing at all.
        // Unknown or unarmed handles are a no-op.
        void disableFrameInterop(uint32_t viewHandle);

        // Is this view's frame interop still armed? The one way a consumer
        // learns about an invalidation (below) without reading stderr: a false
        // here means the exports it imported are dead and the tensors built on
        // them describe memory the renderer no longer writes.
        [[nodiscard]] bool frameInteropActive(uint32_t viewHandle) const;

        // Block until the most recently submitted frame — and therefore the
        // copies recorded at its tail — has completed on the GPU. This waits
        // ONE frame fence, not vkDeviceWaitIdle: submissions on a single
        // VkQueue signal fences in submission order, so retiring the last frame
        // retires every earlier one too. The frame loop waits the same fence
        // again later, and a fence wait is idempotent, so this costs nothing
        // beyond the wait itself. Returns false before the first frame.
        bool syncFrameInterop();

        // ── Sync contract (the mirror of enableVertexInterop's) ──────────────
        // Host ordering is the ONLY cross-API synchronization here, direction
        // reversed: render() → syncFrameInterop() → read the tensors → next
        // render(). There is no shared timeline semaphore (deliberately out of
        // scope; see plans/frame-interop-torch.md).
        //
        // SINGLE-BUFFERED. The exports are one allocation per channel, not a
        // ring: the tensors are live views of the renderer's own memory and the
        // next render() overwrites them in place. A consumer that needs to keep
        // a frame — a replay buffer, an async training batch — clones it
        // (tensor.clone()) while the fence is still held. Reading during a
        // render() that is in flight is torn data, intermittently, by
        // construction, exactly as skipping wp.synchronize_device() is on the
        // inbound path.
        //
        // ── Invalidation ─────────────────────────────────────────────────────
        // Anything that REALLOCATES the source images — a window resize,
        // setRenderScale, setGbufferMsaa, toggling setSplatDepthAov, removing
        // the view — destroys the images the exports copy from. The rule is the
        // one vertex interop uses for an attribute-count change: the renderer
        // DISABLES frame interop for that view, with one line on stderr, rather
        // than reallocating under a live foreign import. The consumer notices
        // (its next enableFrameInterop returns fresh handles) and re-imports.
        // No silent reallocation under an import, ever. Fixed-size secondary
        // views and fixed-size headless windows are the supported steady state,
        // and are what the ML use cases actually run.

        // Hybrid-mode raster overlay: post-TAA wireframe / Line / layer-tagged
        // meshes drawn over the shaded image, depth-tested against the raster
        // G-buffer. -1 (default) disables layer selection.
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        // ── Sensor-only surfaces ─────────────────────────────────────────
        // Geometry the SENSORS may perceive and the camera may not. A surface
        // baked out of a Gaussian-splat scan (threepp::splats::bakeSurface) is
        // the case this exists for: the splat rasterizer already draws that
        // scan in the picture, so the mesh must return lidar ranges and depth
        // WITHOUT appearing next to the cloud it approximates.
        //
        // A mesh opts in by enabling kSensorOnlyLayer (splats::makeSensorMesh
        // does it for a baked surface). From then on, unconditionally: the
        // primary view never rasterizes it, no radiance trace — reflection,
        // refraction, shadow, GI, emissive NEE — has it in its cull mask, and
        // it does not move the GI probe grid. What setSensorOnlySurfaces(true)
        // then adds is PERCEPTION: scanLidar's beams hit it, and the secondary
        // views that ASK for it (setViewSensorSurfaces) rasterize it. OFF, the
        // default, its TLAS instance carries mask 0 and nothing at all sees
        // it — a scene that never opts in senses and renders exactly as it did
        // before the feature existed.
        //
        // This bool is the scene master. On the RASTER side it is necessary
        // and not sufficient: see setViewSensorSurfaces. On the LIDAR side it
        // is the whole decision, because a beam list is already a per-consumer
        // choice — whoever calls scanLidar asked for those beams.
        static constexpr unsigned kSensorOnlyLayer = 31u;
        void setSensorOnlySurfaces(bool enabled);
        [[nodiscard]] bool sensorOnlySurfaces() const;

        // ── Overlays exempt from splat occlusion ─────────────────────────
        // An overlay (wireframe mesh, Line/LineSegments, Points) that ENABLES
        // this layer draws BEFORE the splat depth stamp: it is depth-tested
        // against scene geometry only, and a Gaussian-splat cloud never
        // occludes it. For an overlay that IS a picture of a splat surface —
        // the editor's baked-surface preview hugs the cloud's front within a
        // voxel by construction — this is the only stable choice: the stamp
        // is re-expressed from the jittered splat raster's depth AOV, so at a
        // grazing view from a distance both its coverage gate and its depth
        // wobble per frame by more than the gap between the preview and the
        // cloud, and any depth test between the two flickers (measured:
        // SplatOverlayFlicker_probe). ENABLE the bit (Layers::enable), do not
        // set it exclusively — the object still needs its camera layer.
        // Ordinary overlays should not use this: a grid behind a cloud is
        // behind it, and this layer would put it on top.
        static constexpr unsigned kSplatUnoccludedOverlayLayer = 30u;

        // Per-view raster permission for sensor-only surfaces. OFF by default,
        // for every view, because "secondary" does not mean "sensor": an RGB
        // camera preview (CameraSensor) and an editor viewport pane are
        // addView views too, and a bake shell standing untextured in front of
        // the splats it approximates is a defect in both. A DEPTH consumer —
        // one reading GBufferAOV::Depth off this view — is the caller that
        // should turn it on.
        //
        // A view rasterizes the surface only if this is set AND the scene
        // called setSensorOnlySurfaces(true). Raster state only: it takes
        // effect on the next render and invalidates nothing. False for an
        // unknown handle, and for handle 0 — the primary never draws sensor
        // surfaces at all.
        //
        // A view that serves BOTH RGB and depth for one sensor cannot have it
        // both ways; leave it off (the picture stays correct) and read depth
        // from a lidar scan or a second view.
        bool setViewSensorSurfaces(uint32_t handle, bool enabled);
        [[nodiscard]] bool viewSensorSurfaces(uint32_t handle) const;

        // Per-view permission to rasterize SplatClouds. OFF by default, for
        // every view: the splat pass was primary-only until this existed, and
        // the reason was cost, not correctness. The radix sort scales with
        // SPLAT COUNT rather than view size, so an opted-in view is a second
        // full sort — trivial on a 60k-splat object scan (~0.3 ms), ~8-13 ms on
        // a 5M-splat town whatever the sensor's resolution. Which views can
        // afford that is the caller's call, which is why this is per view.
        //
        // On, the view composites the clouds with the SAME deterministic
        // compute rasterizer the primary uses, at the same point in its frame
        // (linear HDR, before its bloom/tonemap/TAA tail) — an RGB CameraSensor
        // pointed at a scan sees the scan. It draws what the app selected this
        // frame: SplatCloud::setSubmitRanges is per-cloud state, so a
        // dynamic-LOD app's secondary views inherit the PRIMARY's level
        // selection rather than choosing their own.
        //
        // The splat depth AOV stays primary-only (a secondary view's AOV image
        // is 1x1), and so does the debug checksum. False for an unknown handle,
        // and for handle 0 — the primary always draws splats. At most
        // three views may hold the flag at once; a fourth is refused on stderr
        // and renders without splats. Takes effect at the next frame boundary.
        bool setViewSplats(uint32_t handle, bool enabled);
        [[nodiscard]] bool viewSplats(uint32_t handle) const;

        // Day-1 / debug visualization: blit one G-buffer channel to the swapchain.
        //   0 = off, 1 = normal, 2 = motion, 3 = instance id, 4 = albedo
        void setHybridDebugView(int view);
        [[nodiscard]] int hybridDebugView() const;

        // ── Path-traced LIDAR scanner ─────────────────────────────────────
        // Synchronously trace beams against the same TLAS, evaluate a back-scatter
        // LIDAR equation at the first hit, and return per-beam tuples. Submits its
        // own command buffer + fence and blocks until results come back.
        //
        // `cleanResults`, with LidarParams::pairedCleanTrace set, additionally
        // receives the SAME beams traced with the ParticleField density medium
        // switched off — the ground-truth degradation reference. Same layout,
        // same length; pair row i with row i and call lidarDegradation().
        // Without the flag it is left empty and the trace is unchanged.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& results,
                       const LidarParams& params = {},
                       std::vector<LidarReturn>* cleanResults = nullptr);

        // The same scan, PIPELINED — fire on one frame, take delivery on a
        // later one. Use this from a frame loop; scanLidar() above is for a
        // caller with no frame to keep (a test, an offline capture).
        //
        // Blocking on a readback does not cost the trace, it costs every frame
        // already queued on the GPU: the fence sits behind them. Measured on an
        // RTX 4070 at two frames in flight, a 1.2 ms VLP-16 trace cost a 28 ms
        // stall — a hitch a 10 Hz sensor delivers ten times a second. So:
        //
        //   const int scan = scanLidarBegin(beams, params);// frame N: submits
        //   ...render frame N, present...
        //   if (scanLidarReady(scan))            // frame N+1: poll, no wait
        //       scanLidarCollect(scan, results); // a memcpy by now
        //
        // Results are therefore one frame old, which is the pose the beams
        // were actually fired from — a sensor's own latency, not an error.
        //
        // The handle keeps several sensors independent: a rig with a LIDAR and
        // a depth camera both due on the same frame each get their own slot, so
        // which frame a sensor scans on never depends on what else is in the
        // scene. A few scans may be outstanding at once; beyond that Begin
        // returns kNoLidarScan and the caller retries next frame. On a
        // non-Vulkan renderer there is no equivalent, which is why LidarSensor
        // keeps the synchronous path for raster backends.
        static constexpr int kNoLidarScan = -1;
        [[nodiscard]] int scanLidarBegin(const std::vector<LidarBeam>& beams,
                                         const LidarParams& params = {});
        // Whether that dispatch has finished. A fence poll, never a wait;
        // false for a handle that is not outstanding.
        [[nodiscard]] bool scanLidarReady(int handle) const;
        // Take delivery, waiting if it somehow has not finished. False when the
        // handle is not outstanding, in which case `results` is left empty.
        // `cleanResults` takes the paired clean leg, as in scanLidar above.
        bool scanLidarCollect(int handle, std::vector<LidarReturn>& results,
                              std::vector<LidarReturn>* cleanResults = nullptr);

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
            float froxelMs       = 0.f;// froxel volumetrics: inject + integrate (0 unless a medium is active)
            float splatMs        = 0.f;// Gaussian-splat tile rasterizer (0 unless the scene has a SplatCloud)
            // The three stages inside splatMs, for the FIRST splat cloud of the
            // frame only (see TimingPass). They partition the pass: per-splat
            // work, the sort, and the tile composite — which is the split that
            // decides whether an optimisation belongs in front of the sort or
            // behind it.
            float splatProjectMs = 0.f;// project + cull + prefix sum + expand
            float splatSortMs    = 0.f;// the radix passes and their scans
            float splatRasterMs  = 0.f;// tile ranges + composite
            // GPU per-instance world-matrix expansion (0 unless
            // setGpuInstanceExpansion and the scene has instanced geometry).
            float instanceExpandMs = 0.f;
            // ParticleField density scatter — the clear + per-particle splat
            // into the world-space density volume (0 unless a ParticleField in
            // the scene has DensityRepr enabled with live particles). Recorded
            // once per frame however many cameras look at the dust.
            float particleDensityMs = 0.f;
            // ParticleField device emitter — the closed-form position +
            // prevPosition write for every Ownership::Renderer field (0 unless
            // the scene has one with live particles). Like the scatter above it
            // is recorded once per frame however many cameras look at the field,
            // and it is the ENTIRE per-frame cost of such a field: there is no
            // CPU counterpart to pair it with.
            float particleEmitMs = 0.f;
            // Ocean / DisplacedMesh per-frame update, split into its four
            // stages (0 unless the scene has a DisplacedMesh; first displaced
            // mesh of the frame only): the cascade spectrum + IFFT chains, the
            // water_displace vertex/normal dispatch, the world-foam
            // accumulator, and the in-place BLAS refit/rebuild.
            float oceanFftMs      = 0.f;
            float oceanDisplaceMs = 0.f;
            float oceanFoamMs     = 0.f;
            float oceanBlasMs     = 0.f;
            // Per-frame TLAS refit recorded on the frame command buffer
            // (0 on frames with no refit).
            float tlasRefitMs     = 0.f;
            // Per-frame dynamic-geometry refit on the frame command buffer:
            // staging (or interop-export) → vertex/normal copies, the
            // vertex→prevVertex motion snapshot, and the batched BLAS refit, for
            // every graduated CPU deformer and every enableVertexInterop mesh.
            // 0 on frames where none of them changed. Skinned / tet / displaced /
            // grass deformers are NOT in here — they record their own dispatches
            // and BLAS work elsewhere, and only the ocean's is bracketed.
            float dynGeomRefitMs  = 0.f;
            // Half-res RT ambient occlusion + bent normals (rtao.comp).
            // 0 unless setDeferredAO is on (the pass is only dispatched then).
            float rtaoMs          = 0.f;
            // World-space probe-GI update (probe_update.comp) plus its
            // prev-store snapshot copy. 0 unless probe GI is enabled. Recorded
            // once per frame for all views — the dispatch is primary-only — so
            // this is the frame's whole probe cost, not a per-view share.
            float probeGiMs       = 0.f;
            // GPU execution SPAN of the whole submitted command buffer — not a sum
            // of the fields above, and not busy time. It covers the passes that
            // have no timestamp bracket at all (skinned/tet/grass deformers,
            // bloom/post, RCAS, cluster build, cloud march, auto-exposure,
            // particle light, ImGui/present transition) and every secondary view, whose
            // timestamps are suppressed. Read against cpuFrameMs to tell "the CPU
            // is the wall" from "the CPU is waiting".
            float gpuTotalMs     = 0.f;
            // The bracketed passes, summed over a DISJOINT set (the three splat
            // sub-stages are excluded — they partition splatMs; the sensor-image
            // pass is included even though it has no field of its own).
            // gpuTotalMs - gpuPassSumMs is GPU work invisible to the brackets.
            float gpuPassSumMs   = 0.f;
            float cpuEnsureSceneMs = 0.f;// ensureSceneBuilt
            float cpuRecordMs      = 0.f;// recordCommandBuffer
            float cpuFrameMs       = 0.f;// total render() wall time
        };
        [[nodiscard]] FrameTimings lastFrameTimings() const;

        // ── Deferred shading knobs ────────────────────────────────────────────

        // Ray-traced env ambient-occlusion / GI. ON by default — soft RT AO/GI
        // (costs occlusion rays; paired with setDenoise, also on by default, to keep
        // the 1-spp gather noise-free). Turn OFF to drop the per-pixel ray cost, or
        // if occlusion-testing the IBL makes a bright HDRI look like it casts shadows.
        void setDeferredAO(bool enabled);
        [[nodiscard]] bool deferredAO() const;

        // World-space irradiance probe grid (DDGI-lite) — multi-bounce diffuse
        // GI for the deferred gather. A 32×16×32 grid of SH-L1 probes is fitted
        // to the scene AABB and refreshed round-robin (2048 probes × 64 rays
        // per frame, ~sub-ms); the stochastic GI gather then adds each hit's
        // probe irradiance, supplying the bounce-2..∞ + through-the-opening
        // sky light a 1-bounce gather cannot.
        // It ALSO switches the deferred ambient model from cosmetic to
        // MEASURED: scene ambient is gated by the gather's real sky visibility,
        // the rough split-sum env specular gets probe-derived specular
        // occlusion, and reflected hits take probe irradiance instead of the
        // env+ambient fill. Enclosed interiors therefore stop being "lit with
        // no light" — they go properly dark and receive only what bounces in
        // through actual openings (e.g. the Sponza ground-floor corridors);
        // pair with setAutoExposure for interior scenes.
        // ON by default (≈ neutral outdoors; ~0.3 ms probe update). Requires
        // setDeferredAO(true) + setDenoise(true) (the probe term rides the
        // denoised GI channel); the cache converges over a few dozen frames
        // after scene load. setProbeGI(false) restores the legacy cosmetic
        // ambient (ungated env/ambient fill, no multi-bounce).
        void setProbeGI(bool enabled);
        [[nodiscard]] bool probeGI() const;

        // Volumetric SPOT-light beams: ray-marched single scattering through a
        // uniform thin haze. `density` is the scattering coefficient σ (1/m; 0 =
        // off, no cost); `anisotropy` is the Henyey-Greenstein g.
        void setDeferredVolumetrics(float density, float anisotropy = 0.55f);
        // {density, anisotropy} as last set.
        [[nodiscard]] std::pair<float, float> deferredVolumetrics() const;

        // DEPRECATED (Phase 2 fog unification) — a no-op. The directional sun
        // shafts + aerial glow are now ALWAYS on when the fog medium is present:
        // set scene.fog (or setHeightFog) and the volumetrics follow automatically
        // — the froxels own the near field [0, 512 m] and the per-pixel march owns
        // the far tail [512 m, ∞]. Kept only so existing callers compile.
        void setVolumetricFog(bool enabled);
        [[nodiscard]] bool volumetricFog() const;

        // ── Volumetric clouds (Nubis/HZD-style far-field cloud layer) ────────
        // A raymarched, wind-driven, procedurally-shaped cloud deck occupying
        // the world-space shell [bottomY, topY], composited over the sky (and,
        // depth-aware, in front of terrain). Density is analytic Perlin-Worley
        // noise (no baked assets), remapped by coverage, shaped by a height
        // gradient and eroded by detail — the classic Decima recipe. Lit by the
        // scene's claimed sun (one-sun policy) with a Beer light-march + powder
        // term + dual-lobe Henyey-Greenstein phase, and by env ambient. nullopt
        // = off (default), and off is free (image-identical to no clouds).
        struct CloudSettings {
            float coverage    = 0.45f;         // 0 = clear sky, 1 = overcast
            float density     = 1.0f;          // density multiplier
            float bottomY     = 600.0f;        // shell base (world Y, m)
            float topY        = 1400.0f;       // shell top (world Y, m)
            Vector3 wind{8.0f, 0.0f, 2.0f};    // m/s xz drift (y ignored)
            float evolveSpeed = 1.0f;          // shape churn rate
        };
        void setClouds(const std::optional<CloudSettings>& settings);
        [[nodiscard]] std::optional<CloudSettings> clouds() const;

        // ── Fog medium PROFILE control (advanced) ────────────────────────────
        // Phase 2 fog unification: there is ONE air-fog medium. `scene.fog` is
        // the primary knob — present, it supplies the medium DENSITY (FogExp2
        // density / linear-Fog span) and COLOUR, and the volumetrics run
        // automatically. setHeightFog is the ADVANCED control of that same
        // medium's PROFILE: an exponential height falloff (baseY / falloff) ×
        // wind-scrolled 3D noise, evaluated inside the 0.25–512 m view froxels.
        //
        // DENSITY PRECEDENCE (the PROFILE — baseY/falloff/noiseAmount — always
        // applies when setHeightFog is set):
        //   • setHeightFog density > 0 → the deliberate ADVANCED OVERRIDE: it WINS
        //     over scene.fog (so an explicit ground-mist density holds even while a
        //     scene sets scene.fog per frame).
        //   • setHeightFog density <= 0 → "profile-only": scene.fog supplies the
        //     density (keeping a live Fog-density slider effective in scenes that
        //     drive scene.fog every frame), setHeightFog only shapes the profile.
        //   • setHeightFog absent → scene.fog's density drives the medium with a
        //     near-uniform default profile (baseY 0, huge falloff ≈ homogeneous).
        //   • neither present → no air medium.
        //
        // The froxels run HETEROGENEOUS whenever a medium exists: per-slice
        // density, a froxel sun in-scatter term (1 RT shadow ray + a short
        // self-shadow march) for [0, 512 m], and the per-pixel march for the far
        // tail. NOTE the froxel medium is the height-fog profile ONLY: the
        // setClouds layer is integrated by the far cloud march over the WHOLE ray,
        // so the two volumes split by phenomenon — no cloud/froxel hand-off. The
        // underwater murk is a SEPARATE medium (setUnderwaterMurk). nullopt = the
        // default near-uniform profile (no explicit height falloff).
        struct HeightFogSettings {
            float density     = 0.02f;// σ_t at baseY; > 0 OVERRIDES scene.fog,
                                      // <= 0 = profile-only (scene.fog's density)
            float baseY       = 0.0f;
            float falloff     = 80.0f;// exponential height scale (m)
            float noiseAmount = 0.6f; // 0 = smooth analytic, 1 = fully noise-modulated
        };
        void setHeightFog(const std::optional<HeightFogSettings>& settings);
        [[nodiscard]] std::optional<HeightFogSettings> heightFog() const;

        // Procedural star field on SKY pixels — hash-based points in direction
        // space, pixel-crisp at any resolution/FOV. 0 disables; ~1.0 = night sky.
        void setDeferredStarfield(float intensity);
        [[nodiscard]] float deferredStarfield() const;

        // HDRI sun extraction. The environment map's dominant compact bright
        // source (the sun) is removed from the PMREM's glossy / rough mips at
        // upload — a ~10⁴:1 disc cannot be Monte-Carlo prefiltered smoothly and
        // shows up as bright blocky "spec blobs" in reflections — and its exact
        // energy can be re-injected as an analytic directional light: a sharp
        // correct sun highlight (the ONLY sun reflection), soft RT shadows
        // (setSunAngularRadius), GI bounce and volumetric shafts. The sky
        // background and true mirror lookups (env mip 0) keep the visible disc.
        //
        // ONE-SUN POLICY — a scene must not end up with the sun twice. Scenes
        // authored for raster renderers add an explicit DirectionalLight as the
        // sun stand-in (raster can't shadow from an env map); an extractor that
        // ALSO injects the env's sun would then light and shadow the scene with
        // two suns (the reported double-shadow). So:
        //   Auto (default) — extract (clamp the glossy mips) always; INJECT the
        //     analytic sun only while the scene has NO visible DirectionalLight.
        //     If the artist provided a sun light, theirs owns direct sun +
        //     shadow (the raster/three.js convention) and the env supplies
        //     sky/ambience only. Exactly one sun in every renderer.
        //   Always — extract AND inject regardless of scene lights (a scene
        //     that genuinely wants the env sun PLUS extra directional lights).
        //   Off — no extraction at all: raw env in every mip (legacy; the HDRI
        //     sun prefilters into blocky spec blobs), nothing injected.
        // Auto↔Always applies next frame; Off toggles force an env re-upload.
        enum class EnvSunPolicy { Auto, Always, Off };
        void setEnvSunPolicy(EnvSunPolicy policy);
        [[nodiscard]] EnvSunPolicy envSunPolicy() const;

        // Back-compat shim: true → Auto, false → Off.
        void setEnvSunExtraction(bool enabled);
        [[nodiscard]] bool envSunExtraction() const;

        // The measured env sun (valid while envSunFound()): unit direction
        // TOWARD the sun and the disc's integrated energy Σ L·dΩ (linear RGB
        // irradiance). Use to ALIGN an explicit DirectionalLight with the HDRI
        // (e.g. so a raster renderer's shadow direction matches the sky).
        [[nodiscard]] bool envSunFound() const;
        [[nodiscard]] Vector3 envSunDirection() const;
        [[nodiscard]] Vector3 envSunColor() const;

        // ── Automatic exposure (eye adaptation) ──────────────────────────────
        // When enabled the renderer samples the log2-luma histogram of the
        // rendered frame each tick and drives toneMappingExposure toward the
        // value that maps the scene's weighted-average luminance to 18% gray,
        // using an asymmetric EMA (fast constriction, slow dilation).
        // toneMappingExposure is IGNORED while auto-exposure is active.
        void setAutoExposure(bool enabled);
        [[nodiscard]] bool autoExposure() const;

        // EV per second for brightness adaptation (default 2.0).
        // Dilation (scene-darkens) is applied at 0.5× this speed.
        void setAutoExposureSpeed(float evPerSecond);
        [[nodiscard]] float autoExposureSpeed() const;

        // Exposure clamp in EV relative to 1.0 (default -3 to +3 EV).
        // E.g. setAutoExposureRange(-2, 4) limits to 0.25× .. 16× exposure.
        void setAutoExposureRange(float minEV, float maxEV);

        // ── Raster G-buffer MSAA (edge/silhouette anti-flicker) ─────────────
        // Rasterizes the material G-buffer at `samples` (1, 2, or 4) per
        // pixel instead of 1, then resolves each pixel to the majority-
        // covering surface (dominant-sample pick, not a box/average blend —
        // averaging normals/ids/depth across a silhouette produces nonsense).
        // Targets the source of the 1-spp jittered-coverage edge flicker
        // (leaf canopies, low-poly rock fields shimmering on a STATIC
        // camera): sample coverage is exact and far more temporally stable
        // than a single jittered point sample. Default 1 = today's path,
        // byte-identical output, zero extra cost. Reallocates render-extent
        // resources (render pass + pipelines + MS images) — same
        // vkDeviceWaitIdle / deferred-apply-mid-frame contract as
        // setRenderScale; safe to call from inside the user's animate
        // lambda. VRAM cost is real (roughly samples× the G-buffer's raster
        // attachments); 2 is the recommended quality step, 4 for maximum
        // stability.
        void setGbufferMsaa(uint32_t samples);
        [[nodiscard]] uint32_t gbufferMsaa() const;

        // ── Gaussian-splat determinism hook (test-only) ─────────────────────
        // Enables order-independent hashes over the splat pass's own buffers:
        // out[0] = the sorted key array, out[1] = the sorted payload array,
        // out[2] = the composited pixels, out[3] = the expanded (splat, tile)
        // pair count. Two frames rendered from the same camera must produce
        // the same four numbers, and if they do not, the tile expansion or the
        // radix sort is order-dependent — which is the failure the prefix-sum
        // expansion exists to prevent, and the reason sensor goldens can be
        // trusted at all.
        //
        // Hashing the pass's OWN buffers rather than the framebuffer is the
        // point: a pixel comparison downstream also has to survive TAA jitter
        // and the denoiser's temporal state, neither of which is on trial.
        //
        // splatDebugChecksum() drains the device. Off by default; both the
        // extra dispatches and the per-pixel atomic cost real time. Returns
        // false when no splat cloud was drawn. (THREEPP_VK_SPLAT_CHECKSUM=1
        // turns the same hashes on from the environment, plus two invariant
        // assertions printed to stderr.)
        void setSplatDebugChecksum(bool enabled);
        [[nodiscard]] bool splatDebugChecksum(std::uint64_t out[4]) const;

        // How many splat clouds hold GPU buffers right now. A test surface for
        // the eviction contract — a deleted cloud's ~1.2 GB (at 5M splats) must
        // actually leave the device once its last referencing frame drains, and
        // VRAM itself is not assertable from a test. 0 when no splat pass exists.
        [[nodiscard]] std::size_t splatResidentClouds() const;

        // The splat pass's shared-scratch high-water, in splats. The other half
        // of the eviction test surface: "deleting the scan released its ~700 MB
        // of sort scratch" is a VRAM claim, and this is the assertable form.
        [[nodiscard]] std::size_t splatScratchSplats() const;

        // ── The splat reflection volume (plans/splat-volume-reflections.md) ──
        // Each resident cloud is voxelized once, at upload, into an rgba16f
        // medium that reflection legs march. These three are its test surface,
        // in the same style as the two above.
        //
        //   splatVolumeBytes()      resident volume VRAM; 0 once the clouds are
        //                           evicted, and 0 under THREEPP_VK_SPLATVOL_OFF
        //   splatVolumeGeneration() bumped when the SET of baked volumes changes
        //   splatVolumeHash(out)    out[0] = FNV-1a of every volume's texels,
        //                           out[1] = texels hashed, out[2] = texels a
        //                           splat actually reached. The bake accumulates
        //                           with integer atomics precisely so this is
        //                           reproducible; DRAINS THE DEVICE and copies
        //                           the whole volume back, so it is a test call.
        [[nodiscard]] std::uint64_t splatVolumeBytes() const;
        [[nodiscard]] std::uint64_t splatVolumeGeneration() const;
        void splatVolumeHash(std::uint64_t out[3]) const;

        // ── GPU per-instance world matrices ───────────────────────────────────
        // A compute pass (instance_expand.comp) that recomputes, per
        // InstancedMesh span, the world matrix the CPU bakes into every
        // instance's scene entry: world = mesh.matrixWorld * instanceMatrix[i].
        //
        // Right now NOTHING consumes it — the draw list, motion vectors,
        // frustum cull and the ray-tracing instance descriptors all still read
        // the CPU values — so this costs an instance-matrix upload and a
        // dispatch and saves nothing yet. It is the verified foundation for
        // moving those consumers onto the GPU; see
        // plans/gpu-driven-instances.md. ON by default so the pass is covered
        // by the same tests as everything else; turn it OFF to A/B its cost or
        // to take it out of the frame entirely.
        void setGpuInstanceExpansion(bool enabled);
        [[nodiscard]] bool gpuInstanceExpansion() const;

        // Test/debug hook: read the compute pass's output back and compare it
        // against the CPU's per-instance matrices, over every instanced span of
        // the frame last rendered. `mismatches` counts entries whose 16 floats
        // are not BITWISE identical — the multiply is spelled out on both sides
        // in the same order precisely so that 0 is the achievable answer, and a
        // non-zero maxAbsDiff with mismatches == 0 is impossible by
        // construction. False when the pass is off or the scene has no
        // instanced geometry.
        //
        // FINALIZES the in-flight frame and then DRAINS THE DEVICE: render()
        // leaves the command buffer open (the present is deferred to the canvas
        // frame-end callback), so reading the output before closing it would
        // compare this frame's CPU matrices against last frame's GPU ones —
        // equal only when nothing moved, i.e. exactly when the check proves
        // nothing.
        struct InstanceExpandCheck {
            std::size_t spans           = 0;// instanced spans on the GPU path
            std::size_t entriesCompared = 0;
            std::size_t mismatches      = 0;// bitwise-unequal matrices
            float       maxAbsDiff      = 0.f;
            std::uint32_t maxUlpDiff    = 0;// worst element, in float steps
        };
        bool instanceExpandCheck(InstanceExpandCheck& out);

    private:
        // The one implementation struct, defined in vulkan/VulkanCoreImpl.hpp
        // and opaque to callers of this header.
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] Impl* core() const { return pimpl_.get(); }
    };

    // Back-compat alias: the renderer briefly had an abstract VulkanRendererCore
    // base while a second Vulkan shading backend existed. There is one backend
    // now, so the base is gone; the name still resolves for existing code
    // (including dynamic_cast, which sees the same type).
    using VulkanRendererCore = VulkanRenderer;

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
