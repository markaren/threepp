// VulkanViewContext — everything a single camera's render owns: its G-buffer
// and every temporal accumulator hanging off it, the per-frame-in-flight
// buffers that describe the camera to the GPU, and the host-side temporal
// bookkeeping that makes a camera CUT distinguishable from a camera MOVE.
// Split out of VulkanCoreImpl.hpp; each struct is aliased back into
// VulkanRenderer::Impl at its original spot so every reference site — Impl's
// own methods and the VulkanCore*.cpp TUs — is unchanged.
//
// ViewContext embeds RasterGbufImages by value, so that one stays first.

#ifndef THREEPP_VULKAN_VIEW_CONTEXT_HPP
#define THREEPP_VULKAN_VIEW_CONTEXT_HPP

#include "VulkanImplCommon.hpp"
#include "VulkanResources.hpp"
#include "VulkanSceneTypes.hpp"// DrawGroup (indirect-skip cache)
#include "BloomPass.hpp"
#include "DeferredShade.hpp"
#include "PostComposite.hpp"
#include "TaaResolve.hpp"

#include "threepp/cameras/Camera.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp::vulkan::impl {

    // ── Hybrid raster G-buffer prepass ──────────────────────────────────
    // Replaces the old PT primary-ray traversal: raster writes
    // depth/normal/motion/IDs into per-frame attachments; the deferred
    // shade reads them and starts its ray-query work at bounce 1.
    // Eliminates moving-object shake from stochastic primary-ray jitter
    // and makes primary visibility deterministic per pixel. AA happens
    // via TAA on top of raster, not as Monte Carlo on the primary ray.
    // Disabled by default until the integration is validated end-to-end
    // (stage 1).
    struct RasterGbufImages {
        Image2D       normal;       // rgba16f — world-space normal in xyz, .w = linear roughness
        Image2D       motion;       // rgba16f — NDC delta in .rg, .ba reserved
        Image2D       ids;          // rgba16ui — instanceCustomIndex/meshID/flags/reserved
        Image2D       uv;           // rgba16f — material UV in .rg
        Image2D       albedo;       // rgba8 unorm — linear base colour in .rgb, metalness in .a (raster-first deferred input)
        Image2D       indirect;     // rgba16f — demodulated diffuse-indirect irradiance (deferred denoiser scratch; STORAGE, not an attachment)
        Image2D       momentsSq;    // rg32f — .r = temporally-accumulated E[L²] of the indirect luminance (SVGF variance: var = E[L²] - lum(indirect)²; r32 because the square overflows fp16 at lum≈256); .g = GI content-change trend (antilag: stale moving-object contact-darkening/bounce fades at content rate); STORAGE+SAMPLED, ping-ponged like indirect
        Image2D       atrousA;      // rgba16f — SVGF multi-pass à-trous ping-pong (rgb=GI, a=variance); STORAGE scratch
        Image2D       atrousB;      // rgba16f — SVGF multi-pass à-trous ping-pong (the other half)
        Image2D       reflect;      // rgba16f — sharp 1-mirror-ray reflection radiance (.rgb), demodulated; roughness-blurred by the reflection denoise. STORAGE
        Image2D       reflAux;      // rgba16f — reflection-denoiser auxiliary (ping-pong, mirrors `reflect`: STORAGE write + SAMPLED prev-frame read)
        Image2D       shadowVis;    // rgba16f — denoised-shadow channel accumulator (.x=visibility ratio, .y=E[R²], .z=histLen, .w=trend); STORAGE + SAMPLED, ping-ponged like indirect
        Image2D       directU;      // rgba16f — unshadowed analytic direct (dir/point/spot) for the denoise recombine (U × R̃); STORAGE, current frame only
        Image2D       shadowAtrousA;// rg16f — shadow-ratio à-trous ping-pong (x=R, y=variance); STORAGE scratch
        Image2D       shadowAtrousB;// rg16f — shadow-ratio à-trous ping-pong (the other half)
        Image2D       froxelScatter;// rgba16f 3D (128×72×64, FIXED size) — froxel in-scatter accumulator (.a=histLen); STORAGE + SAMPLED, ping-ponged like indirect
        Image2D       froxelLut;    // rgba16f 3D — front-to-back-integrated volumetric LUT; STORAGE (integrate) + SAMPLED (shade, trilinear)
        Image2D       cloudColor;   // rgba16f HALF-res — cloud march result (rgb=in-scatter, a=transmittance); STORAGE (march) + SAMPLED (shade upsample + prev-fif reproject), ping-ponged
        Image2D       cloudAux;     // rg16f HALF-res — cloud mean-depth (.r) + temporal histLen (.g); STORAGE (march) + SAMPLED (prev-fif history), ping-ponged
        Image2D       cloudShadow;  // r8 512² (FIXED) — top-down cloud transmittance over an 8 km camera-centred square; STORAGE (shadow pass) + SAMPLED (surface/froxel/water sun); regenerated per frame
        Image2D       rtao;         // rgba16f HALF-res — RT ambient occlusion (rgb=bentN*0.5+0.5, a=ao); STORAGE (rtao pass) + SAMPLED (shade upsample + prev-fif reproject), ping-ponged
        Image2D       rtaoAux;      // rgba16f HALF-res — RTAO temporal aux (.r=histLen, .g=E[ao^2], .b=near-field ao, .a=skyVis); STORAGE (rtao pass) + SAMPLED (prev-fif history), ping-ponged
        Image2D       splatDepth;   // r32f — Gaussian-splat expected view distance in world units, 0 where no cloud owns the pixel (SplatPass); FULL-RES only under setSplatDepthAov, 1x1 otherwise. STORAGE + TRANSFER (clear + AOV copy), GENERAL for its whole life
        Image2D       depth;        // d32_sfloat — JITTERED projection (matches color attachments above; consumed by chit + TAA)
        // Hybrid raster overlay's UNJITTERED depth attachment. Filled by
        // an extra depth-only prepass (overlay_depth.vert) right after
        // the main G-buffer pass. The wireframe overlay reads it as a
        // depth attachment so its depth test compares unjittered z
        // against unjittered z and doesn't shimmer between frames.
        Image2D       unjitDepth;   // d32_sfloat — UNJITTERED projection
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        uint32_t      width = 0;
        uint32_t      height = 0;

        // ── MSAA raster targets (only allocated when gbufMsaaSamples_ > 1) ──
        // True multisampled siblings of normal/motion/ids/uv/albedo/depth,
        // rasterized at VK_SAMPLE_COUNT_{2,4}_BIT. The single-sample images
        // above stay allocated unchanged and become the RESOLVE TARGETS: every
        // existing consumer (DeferredShade, TaaResolve, the shade's hybrid
        // set, debug blit) keeps reading them, unaware MSAA is involved. gbuf_resolve
        // (GbufResolve.{hpp,cpp}) picks the per-pixel dominant sample (majority
        // instance id, reversed-Z-nearest tie-break) and writes it into the
        // single-sample images; a tiny depth-only fullscreen pass resolves depth
        // (compute can't write a depth-aspect image). See setGbufferMsaa.
        Image2D       normalMS;
        Image2D       motionMS;
        Image2D       idsMS;
        Image2D       uvMS;
        Image2D       albedoMS;
        Image2D       depthMS;
        VkFramebuffer framebufferMS = VK_NULL_HANDLE;// MS render target (rasterGbufRenderPassMS)
    };

    // ── ViewContext: everything a single camera's render owns ───────────
    // >>> VIEWCTX_BEGIN (the rename pass that introduced view() skips this
    //     block — these are the DECLARATIONS, not uses.)
    //
    // The deferred pipeline used to be hard-wired to exactly one view: one
    // G-buffer set, one temporal-history chain, one camera-UBO chain, all
    // sitting as plain Impl members. Everything below is what turned out to
    // be genuinely PER-VIEW once that assumption was written down, split
    // into three groups:
    //
    //   1. the G-buffer + every temporal accumulator hanging off it
    //      (rasterGbufs — GI, moments, reflections, shadow ratio, froxels,
    //       clouds, depth: all screen-space, all history-bearing),
    //   2. the per-frame-in-flight buffers/descriptors that describe THIS
    //      camera to the GPU (camera UBOs, raster camera UBO, the culled
    //      draw list and its indirect commands, the raster descriptor sets),
    //   3. the host-side temporal bookkeeping the shaders can't hold
    //      (previous VP, previous jitter, sky reprojection, depth
    //       linearization, previous camera pose).
    //
    // Group 3 is the subtle one: it is exactly the state that makes a
    // camera CUT distinguishable from a camera MOVE. Sharing it between
    // views is what would make view A's cut ghost into view B.
    //
    // What deliberately stays on Impl: anything world-space or scene-wide —
    // TLAS/BLAS, geometry/material descriptor tables, light UBOs, the env
    // map and its PMREM, probe GI (world-space by construction), and
    // `sampleIndex` (a clock, not a history).
    struct ViewContext {
        // Group 0 — identity and geometry of the view itself.
        //
        // `secondary` is the one branch that separates this view from the
        // swapchain-presenting primary. False for views_[0], and every
        // extent/target accessor keys off it so the primary path can be
        // read as untouched.
        bool     secondary = false;
        uint32_t id        = 0;// stable handle handed to the public API
        // addView / removeView are routinely called from inside the
        // animate callback, i.e. with a frame already open and its command
        // buffer recording. Allocating (or freeing) GPU resources there is
        // exactly the hazard the rest of this class defers around — see
        // pendingRenderScaleRealloc_ / pendingAccumulationReset_. So a view
        // is registered immediately (the caller gets its handle straight
        // away) and its resources are created, or released, at the next
        // frame boundary: post-fence, pre-record, device drained.
        bool pendingCreate  = false;
        bool pendingDestroy = false;
        // Only meaningful when `secondary`: the primary derives both from
        // the swapchain on every call so it can never go stale on resize.
        // Native-res by scope fence, so these two are equal.
        VkExtent2D renderExt{};
        VkExtent2D outExt{};
        // The camera this view renders. Held as a raw pointer with a uuid
        // alongside, NEVER as a bare pointer used as a cache key — pointers
        // get recycled in this codebase and a recycled Camera* silently
        // inherits the previous camera's temporal history.
        Camera*     camera = nullptr;
        std::string cameraUuid;
        // Where a secondary's finished frame lands: this view's own colour
        // image, in the swapchain's format so the whole TAA/post tail
        // writes it byte-identically to how it writes the swapchain. It is
        // the single entry of the "swapchain of one" that this view's
        // TaaResolve was built against (imageCount = 1, imageIndex = 0).
        // Read back from HERE, never through the swapchain path — a
        // MAILBOX-mode swapchain read is allowed to be stale.
        Image2D colorTarget{};
        // Host-visible landing buffer for readViewRGBPixels. Allocated
        // eagerly with the rest of the view so a readback never allocates
        // mid-frame.
        Buffer  readbackBuf{};
        // Everything this view cost, in bytes, summed as it was allocated.
        // Reported at addView time — a robot rig with eight cameras should
        // find out what it just asked for, not discover it in a VRAM graph.
        VkDeviceSize allocatedBytes = 0;

        // Optional on-screen destination: where this view's finished image
        // is copied into the primary's swapchain image, in top-left pixel
        // coordinates. `displayed` off means the view is a measurement
        // camera only (the default) — it renders and can be read back, but
        // never touches the frame the user sees.
        //
        // This exists so a tool can show a second camera WITHOUT a CPU
        // round trip: the pixels are already on the device, in the
        // swapchain's own format, so putting them on screen is one copy in
        // the frame's existing command buffer rather than a readback, an
        // upload and a texture.
        bool    displayed = false;
        int32_t dispX = 0, dispY = 0;
        // May this view rasterize sensor-only surfaces (meshes on
        // VulkanRenderer::kSensorOnlyLayer)? A DEPTH sensor's view wants them;
        // an RGB camera preview and an editor viewport pane must never show
        // them — they are untextured bake shells standing in front of the
        // splat cloud they approximate. Both are secondary views, so
        // `secondary` cannot be the gate and this flag is. Default off: a view
        // sees them only if it asks AND the scene opted in
        // (Impl::sensorOnlySurfaces_).
        bool    sensorSurfaces = false;
        // May this view rasterize SplatClouds (VulkanRenderer::setViewSplats)?
        // Off by default because the splat sort scales with splat count rather
        // than view size, so a second view is a second full sort — an RGB
        // camera sensor pointed at a scan wants that and an editor pane may
        // not. `splatTarget` is the slot this view claimed in SplatPass's
        // target table (kNoTarget = none yet, or the table was full); it is
        // claimed on the first frame the view renders with the flag set and
        // returned when the view is destroyed.
        bool     splats      = false;
        uint32_t splatTarget = 0xFFFFFFFFu;// == SplatPass::kNoTarget
        // Zero means "this view's own size" — the copy path is a straight
        // vkCmdCopyImage then, with no filtering to argue about.
        int32_t dispW = 0, dispH = 0;

        // This view's frustum-cull result, one byte per entry of
        // lastVisibleEntries_, written by cullEntriesAgainstFrustum.
        //
        // This used to be a bool ON the shared MeshEntry, which worked
        // exactly as long as there was one camera. With N views the last
        // one to cull would win, and the primary's overlay depth prepass —
        // the other reader — would silently draw the WRONG occluder set: no
        // crash, no validation error, just geometry intermittently missing
        // from the overlay depending on where in the frame the secondaries
        // ran. Correct frame ordering happens to hide it today, which is
        // precisely why it should not be left as an ordering obligation.
        //
        // Indexed in lockstep with lastVisibleEntries_; sized (and
        // default-included) by the cull itself, so a view that has never
        // culled draws everything rather than nothing.
        std::vector<uint8_t> inFrustum;
        // Previous frame's inFrustum + a version counter bumped whenever the
        // cull results change. Feeds the buildIndirectDrawData skip signature:
        // DrawInfo/cmd contents depend on the camera ONLY through these bits,
        // so a static scene under a static (or fully-containing) camera can
        // reuse the per-FIF device buffers verbatim.
        std::vector<uint8_t> prevInFrustum;
        uint32_t cullVersion = 0;
        // Cull-recompute gate: inFrustum is still valid when the VP matrix
        // and the scene draw inputs are unchanged since it was last written
        // (static scene + static camera skips the whole cull walk).
        uint32_t cullValidVersion = 0;// drawInputsVersion_ at last cull; 0 = never
        float prevCullVp[16] = {};

        // buildIndirectDrawData skip cache: the input signature each FIF slot's
        // DrawInfo/cmd buffers were last built from ({0,0} = never), plus the
        // CPU-side outputs (bucket groups / total / occl flag) the record path
        // reads — restored on a signature match instead of rebuilt.
        std::array<std::array<uint64_t, 2>, kFramesInFlight> indirectBuiltSig{};
        std::array<vulkan::impl::DrawGroup, 4> cachedIndirectGroups{};
        uint32_t cachedIndirectTotal = 0;
        bool     cachedOcclActive = false;

        // Per-view raster descriptor POOL. The layout is shared (one set
        // shape for every view); the pool is not, because it is sized for
        // exactly kFramesInFlight sets and a second view would exhaust it.
        // A pool per view also means removeView frees its sets by
        // destroying one object instead of returning them individually.
        VkDescriptorPool rasterDescPool = VK_NULL_HANDLE;

        // Group 1 — G-buffer + temporal accumulators.
        std::array<RasterGbufImages, kFramesInFlight> rasterGbufs{};

        // Group 2 — per-frame-in-flight description of this camera.
        //
        // Camera UBO (viewInverse + projInverse), 2 mat4 back-to-back,
        // std140.
        std::array<Buffer, kFramesInFlight> cameraUbos{};
        // Raster camera UBO (RasterCameraData: jittered/unjittered VP,
        // prev VP, jitter).
        std::array<Buffer, kFramesInFlight> rasterCameraUbos{};
        std::array<VkDescriptorSet, kFramesInFlight> rasterDescSets{};
        // Per-frame-slot gate for THIS view's raster binding 3 — the
        // 2048-entry bindless material-texture array (1 = current,
        // 0 = needs (re)write; value-inits to 0 so a view's first
        // uploadRasterCameraUbo always writes it). Lives on the view
        // because the SETS are per-view: as a single Impl-wide array the
        // primary consumed the invalidation first and every secondary's
        // binding 3 stayed stale — a view added after the first frames
        // never had it written at all (no textures), and one that
        // outlived a table rebuild kept image views the rebuild had
        // freed (intermittent device-lost, not a visual glitch).
        std::array<int8_t, kFramesInFlight> rasterMatTexValid_{};
        // Per-frame draw info ring. Each entry mirrors the GLSL DrawInfo
        // struct in gbuffer_indirect.vert: model matrix + buffer device
        // addresses + flags. Sized lazily; grows on demand. Per-view
        // because the contents are the FRUSTUM-CULLED draw list, and each
        // camera culls differently.
        std::array<Buffer, kFramesInFlight> drawInfoBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> drawInfoBufferCapacity{};
        // Per-frame indirect command ring. Holds a contiguous array of
        // VkDrawIndirectCommand structs partitioned by cull mode (Front,
        // then Back, then Double). Counts + offsets per group recorded in
        // indirectGroupRanges_ during recordRasterGbufPass.
        std::array<Buffer, kFramesInFlight> indirectCmdBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> indirectCmdBufferCapacity{};
        // ReSTIR DI reservoir ping-pong — [2] PHYSICAL images (not
        // per-frame-in-flight). Screen-space, so per-view. At frame N the
        // shade writes slot (N & 1) and reads slot ((N+1) & 1).
        std::array<Image2D, 2> reservoirPosImagesPP{};
        std::array<Image2D, 2> reservoirWImagesPP{};

        // Group 3 — host-side temporal bookkeeping.
        //
        // Which projection this view shades through. Read by the uploads
        // (parallel-ray packing, jitter placement) and by DoF, which has no
        // meaning without a lens.
        bool orthoFrame_ = false;
        // Prev-frame camera packed as four vec4s (matches PrevCameraUbo):
        //   [0..3]  = vec4(pos.xyz,  projScaleX)   → prevCamPosX
        //   [4..7]  = vec4(fwd.xyz,  projScaleY)   → prevCamFwdY
        //   [8..11] = vec4(rgt.xyz,  0)             → prevCamRgt
        //   [12..15]= vec4(up.xyz,   0)             → prevCamUp
        std::array<float, 16> prevCamBufData_{};
        bool prevCameraValid = false;
        // Cached unjittered view-projection matrix (column-major,
        // row-of-element-4 layout). Computed once per frame in
        // uploadRasterCameraUbo and read by recordOverlayPass to build
        // the per-draw mvp = vpUnjit · model push constant.
        std::array<float, 16> currVPunjit_{};
        // Cached unjittered view and reverse-Z projection matrices,
        // mirrored alongside currVPunjit_ each frame. The particle
        // billboard pass needs them SEPARATELY (not the combined VP): the
        // distance-attenuated billboard scale uses view-space depth and
        // proj[1][1] individually, so it pushes
        // modelView = currViewUnjit_ · meshWorld and currProjUnjit_.
        std::array<float, 16> currViewUnjit_{};
        std::array<float, 16> currProjUnjit_{};
        bool  rasterPrevVPValid_ = false;
        float rasterPrevVP_[16]{};
        // prevVPunjit · currVPunjit⁻¹ for the TAA's sky-motion
        // reconstruction (sky rasterizes nothing → zero motion → wrong
        // reproject under camera rotation). Computed in
        // uploadRasterCameraUbo while both VPs are in hand; identity until
        // the first real frame.
        std::array<float, 16> taaSkyReproj_{1.f, 0.f, 0.f, 0.f,
                                            0.f, 1.f, 0.f, 0.f,
                                            0.f, 0.f, 1.f, 0.f,
                                            0.f, 0.f, 0.f, 1.f};
        // Reverse-Z view-depth linearization (A,B,C,D) for the TAA depth
        // disocclusion gate: viewZ = (A·d + B)/(C·d + D). Set each frame in
        // uploadRasterCameraUbo from the inverse reverse-Z projection. Zero
        // until the first real frame ⇒ shader leaves the depth gate off.
        std::array<float, 4> taaDepthLin_{};
        // This frame's Halton jitter in RENDER TEXELS for the TAA resolve's
        // current-sample jitter cancellation (taa_resolve re-anchors every
        // current-frame read at the unjittered pixel center — without it
        // the composed output translates with the 8-phase jitter: the
        // systemic "everything shakes"). {0, 0} whenever the raster renders
        // unjittered (MSAA mode, event camera) — the resolve then runs
        // bit-identical to its pre-cancellation arithmetic.
        float taaJitterTexels_[2]{};
        float rasterPrevJitter_[2]{};
        bool  rasterPrevJitterValid_ = false;
        // Camera WORLD motion this frame (translation m, forward-rotation
        // rad) for the deferred reflection history policy: a chase-cam
        // surface (car sunroof with a following camera) is
        // screen-STATIONARY — its motion vectors are ~0 — while its
        // view-dependent reflection content slides with every meter the
        // camera travels. Screen-space motion alone cannot see
        // camera+object co-motion; these can.
        float deferredCamPrevPos_[3]{};
        float deferredCamPrevFwd_[3]{};
        bool  deferredCamPrevValid_ = false;
        float deferredCamDeltaLen_ = 0.f;

        // Group 4 — the passes whose DESCRIPTOR SETS or IMAGES are this
        // view's. Each owns a pipeline it could in principle share with a
        // sibling view; they are per-view anyway because the alternative is
        // rewriting a descriptor set that may still be in flight, and this
        // codebase has already paid for that lesson once (VUID-03047, fixed
        // by going per-frame-in-flight — this is the same fix extended by
        // one dimension).
        //
        //   taa_    owns the temporal history ping-pong. The most
        //           view-specific object in the renderer: sharing it is
        //           exactly how view A's camera cut would ghost view B.
        //   bloom_  owns sceneHdr at the render extent (the deferred
        //           shade's output target) plus the pyramid above it.
        //   post_   owns hdrOut at the display extent.
        //   deferredShade_  owns per-frame-in-flight sets binding this
        //           view's G-buffer, reservoirs and sceneHdr.
        //
        // Deliberately NOT here, and primary-only by scope: occlusion
        // culling + its Hi-Z, the MSAA G-buffer resolve, and the thin-lens
        // DoF. They read per-view images too, but no secondary view is
        // allowed to use them, so they stay single instances on Impl.
        std::unique_ptr<vulkan::TaaResolve>    taa_;
        std::unique_ptr<vulkan::BloomPass>     bloom_;
        std::unique_ptr<vulkan::PostComposite> post_;
        std::unique_ptr<vulkan::DeferredShade> deferredShade_;
    };

    // Per-frame raster camera data. currVPjittered drives gl_Position;
    // currVPunjittered + prevVP drive the motion-vector computation
    // (which must be jitter-free or motion vectors include the jitter
    // and pollute reproject).
    struct RasterCameraData {
        float currVPjittered[16];
        float currVPunjittered[16];
        float prevVP[16];
        float jitter[4];          // .xy = clip-space sub-texel offset, .zw = 1/resolution
        float prevJitter[4];      // .xy = previous frame's jitter. NOTE: gbuffer.frag's
                                  // motion vec is JITTER-FREE (clean prevNDC − currNDC
                                  // from the unjittered VPs — a (prev−curr) jitter delta
                                  // was tested and rejected there); .xy is kept for the
                                  // deferred shade's hybrid reproject tap correction.
                                  // .z smuggles the normal-map Toksvig toggle.
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_VIEW_CONTEXT_HPP
