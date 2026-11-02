// DlssUpscaler — NVIDIA DLSS Super Resolution (NGX) temporal upscaler.
//
// A compile-time replacement for the built-in temporal upsampler (TaaResolve /
// taa_resolve.comp) on Windows + Vulkan + RTX, sitting beside (and outranking)
// the FSR 3.1 path: same seam, same inputs, same output slot. Only DLSS
// Super Resolution is used — Frame Generation is deliberately never touched
// (interpolated frames are not valid sensor data for this platform).
//
// DLSS Ray Reconstruction (DLSS-D, nvngx_dlssd.dll) is a second NGX feature on
// the same seam, opt-in through VulkanRenderer::setDlssRayReconstruction. It
// replaces the SR evaluate on the frames it runs (the RR guide: "when DLSS-RR
// is enabled, DLSS-SR is no longer being executed") and takes SR's inputs plus
// guide images: diffuse and specular albedo, and the normals with roughness in
// .w (the packed roughness mode). The SR feature stays alive beside it, so the
// fallback after a sticky RR failure is SR, not the TAA. The guides are
// placeholders for now (diffuse 1, specular 0, the G-buffer normal image),
// which proves the feature's lifecycle; the shade writes real ones later
// (plans/dlss-ray-reconstruction.md).
//
// This class encapsulates ALL NGX usage so the rest of the renderer never sees
// an NGX type. Unlike the FFX runtime there is no dllexport trap: the NGX entry
// points are in the static import lib (nvsdk_ngx_d.lib), and the lib locates
// the signed nvngx_dlss.dll at init time — we pass the module directory (the
// .exe, or threepp_py.pyd under the Python bindings) as an explicit NGX search
// path, since a .pyd's directory is not searched by default.
//
// Lifetime mirrors FsrUpscaler: constructed once, its DLSS feature (re)created
// for the current *display* (swapchain) extent. The feature is created with
// maxRenderSize == display extent, so any renderScale in (0,1] is a valid
// per-dispatch render subrect with no recreation — only a swapchain/display
// resize recreates. NGX feature creation records GPU work, so create() takes a
// one-shot command buffer the caller submits (and waits) afterwards.
//
// Layout contract (recordDispatch): NGX does not transition resources itself.
// recordDispatch borrows the inputs into SHADER_READ_ONLY_OPTIMAL and the
// output into GENERAL, then restores the caller-supplied original input
// layouts — the caller hands over images in their steady-state layouts and
// gets them back unchanged. The output is left in GENERAL with the upscale's
// writes (same hand-off as FsrUpscaler / TaaResolve).

#ifndef THREEPP_VULKAN_DLSS_UPSCALER_HPP
#define THREEPP_VULKAN_DLSS_UPSCALER_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// NGX forward declarations (kept out of the renderer's headers).
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Resource_VK;

namespace threepp::vulkan {

    class VulkanContext;

    class DlssUpscaler {

    public:
        DlssUpscaler(VulkanContext& ctx, uint32_t framesInFlight);
        ~DlssUpscaler();
        DlssUpscaler(const DlssUpscaler&) = delete;
        DlssUpscaler& operator=(const DlssUpscaler&) = delete;

        // Vulkan device extensions NGX/DLSS needs, queried from the static lib
        // BEFORE device creation (no NGX init required). Returns the raw NGX
        // list; the caller filters against what the device actually supports —
        // on non-NVIDIA hardware most are absent and DLSS init later fails
        // gracefully into the FSR/TAA fallback. Instance extensions NGX asks
        // for are core in Vulkan 1.3 (the renderer's instance version).
        static std::vector<const char*> requiredDeviceExtensions();

        // (Re)create the DLSS feature for the given display (swapchain) extent.
        // Idempotent — releases any existing feature first. NGX itself is
        // initialised lazily on the first call (needs the live device).
        // renderW/H select the quality preset (qualityFor); the feature itself
        // is created with maxRenderSize == display, so a renderScale change
        // inside that preset's range needs no recreation, only a display resize
        // or a change of preset does (see quality()).
        // Feature creation records initialisation GPU work into initCb — the
        // caller submits and waits it before the first recordDispatch. Returns
        // true on success; on failure the feature stays null (valid() == false)
        // and the caller keeps the FSR/TAA path.
        bool create(VkCommandBuffer initCb,
                    uint32_t displayWidth, uint32_t displayHeight,
                    uint32_t renderWidth, uint32_t renderHeight);
        void destroy();
        [[nodiscard]] bool valid() const { return feature_ != nullptr; }
        [[nodiscard]] uint32_t displayWidth() const { return displayW_; }
        [[nodiscard]] uint32_t displayHeight() const { return displayH_; }

        // The NGX quality preset a render/display ratio selects (DLAA at
        // scale 1, then MaxQuality down to UltraPerformance), as the enum's
        // integer, and the preset the live feature was created with. NGX
        // evaluates a render subrect only inside the created preset's range —
        // DLAA takes the display size and nothing smaller — so a renderScale
        // whose preset differs from the feature's needs a recreate; the
        // display-extent funnel compares the two. -1 until create() succeeds.
        static int qualityFor(uint32_t renderWidth, uint32_t displayWidth);
        [[nodiscard]] int quality() const { return quality_; }

        // All render-extent inputs unless noted. NGX binds by IMAGE VIEW (it
        // uses VK_NVX_image_view_handle), so views are required alongside the
        // images — unlike the FFX backend which creates its own views. Formats
        // are the images' real VkFormats (color rgba16f linear-HDR, depth D32
        // reversed-Z, motion rg16f-in-rgba16f NDC-delta, output rgba16f display
        // extent). Layouts are the images' CURRENT (steady-state) layouts;
        // recordDispatch transitions them into the NGX read/UAV layouts and
        // restores these exact layouts before returning.
        struct DispatchInputs {
            VkCommandBuffer cmd = VK_NULL_HANDLE;

            VkImage       colorImage  = VK_NULL_HANDLE;
            VkImageView   colorView   = VK_NULL_HANDLE;
            VkFormat      colorFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout colorLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkImage       depthImage  = VK_NULL_HANDLE;
            VkImageView   depthView   = VK_NULL_HANDLE;
            VkFormat      depthFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkImage       motionImage  = VK_NULL_HANDLE;
            VkImageView   motionView   = VK_NULL_HANDLE;
            VkFormat      motionFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            // Output at DISPLAY extent, storage image (UAV → GENERAL).
            VkImage       outputImage  = VK_NULL_HANDLE;
            VkImageView   outputView   = VK_NULL_HANDLE;
            VkFormat      outputFormat = VK_FORMAT_UNDEFINED;

            uint32_t renderWidth = 0, renderHeight = 0;
            uint32_t displayWidth = 0, displayHeight = 0;

            // Sub-pixel jitter [-0.5,0.5] applied to the projection this frame
            // (raw Halton texel units, as fed to m02/m12 += 2j/extent). The
            // dispatch converts PER AXIS to (−jitterX, +jitterY) — the image-
            // content shift in y-down pixel space, same convention as FSR; a
            // scalar sign can't express the GL-y-up clip → y-down pixel
            // conversion (see jitterSignX/Y in the .cpp for the measured 2×2
            // matrix; THREEPP_DLSS_JITTER_SIGN_X/_Y=neg|pos override per axis).
            float jitterX = 0.f, jitterY = 0.f;
            // NDC-delta → render-pixel conversion: {0.5*renderW, -0.5*renderH}.
            float motionScaleX = 0.f, motionScaleY = 0.f;

            float frameTimeDeltaMs = 16.6f;
            float preExposure = 1.f;// > 0; the pre-exposure baked into colour
            bool  reset = false;    // camera cut / history invalidation

            // Bias-current-color mask (opt-in) — DLSS's equivalent of FSR's
            // reactive mask, and it matters for the same content: deformer /
            // wind-swept surfaces (grass!) whose motion vectors can't describe
            // the shader displacement, so accumulated history GHOSTS at their
            // edges. When `reactive` is true, recordDispatch generates an R8
            // mask from the G-buffer IDs flags (fsr_reactive.comp — shared
            // shader) and feeds it as pInBiasCurrentColorMask so those pixels
            // favor the current frame. Same inputs as the FSR path: `frame`
            // picks the per-fif mask image + set, `idsView` is this frame's
            // raster G-buffer IDs view (SHADER_READ_ONLY, usampler2D).
            uint32_t      frame         = 0;
            VkImageView   idsView       = VK_NULL_HANDLE;
            bool          reactive      = false;
            float         reactiveValue = 0.6f;

            // Ray Reconstruction. When set and rayReconstructionValid(), the
            // DLSSD evaluate runs instead of SR's. normal* is the G-buffer
            // normal (rgba16f, linear roughness in .w: the packed roughness
            // mode reads it there), left in its layout when that is already
            // SHADER_READ_ONLY_OPTIMAL. The matrices are the camera's view and
            // projection: NGX wants them row-major for row vectors (v * M),
            // which is the transpose of threepp's column-vector matrices and
            // so exactly their column-major element arrays. NGX reads them
            // only with a specular hit distance, which nothing feeds yet.
            bool          rayReconstruction = false;
            VkImage       normalImage  = VK_NULL_HANDLE;
            VkImageView   normalView   = VK_NULL_HANDLE;
            VkFormat      normalFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout normalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            float worldToView[16]{};
            float viewToClip[16]{};
        };

        // Record the DLSS evaluate into in.cmd. No-op if !valid(). See the
        // layout contract in the class header comment.
        void recordDispatch(const DispatchInputs& in);

        // ── Ray Reconstruction (DLSS-D) ───────────────────────────────────
        // The DLSSD feature lives beside SR's and is created and released by
        // the frame loop, which calls rayReconstructionNeedsSync() once a
        // frame with the knob's state and the render extent. The RR guide
        // states that RR has no dynamic resolution, so unlike SR the feature
        // is created AT the render extent and recreated when it changes. A
        // failed creation, or an evaluate failure the self-heal could not
        // clear (giveUpRayReconstruction), keeps RR off until the next
        // create(), i.e. the next display resize or preset change; SR runs
        // in the meantime.
        [[nodiscard]] bool rayReconstructionAvailable() const { return rrAvailable_; }
        [[nodiscard]] bool rayReconstructionValid() const { return featureRR_ != nullptr; }
        [[nodiscard]] bool rayReconstructionFailing() const { return rrEvalFails_ >= 3; }
        [[nodiscard]] bool rayReconstructionNeedsSync(bool wanted,
                                                      uint32_t renderWidth, uint32_t renderHeight) const;
        // Creates the DLSSD feature (releasing any old one first) and its
        // placeholder guide images, recording the NGX init work and the guide
        // clears into initCb, which the caller submits and waits. Needs a
        // live SR feature (create() succeeded). The caller idles the device
        // before replacing a feature that may still be in flight.
        bool createRayReconstruction(VkCommandBuffer initCb,
                                     uint32_t renderWidth, uint32_t renderHeight);
        void destroyRayReconstruction();
        void giveUpRayReconstruction();

        // NGX evaluate has entered a sticky failure state (≥3 consecutive
        // failures, e.g. 0xBAD00005 InvalidParameter after an extent transition
        // the resize funnel didn't see). The record path falls back to FSR/TAA
        // while this is true; the frame loop self-heals by recreating the
        // feature (create() resets the counter).
        [[nodiscard]] bool failing() const { return evalFails_ >= 3; }

    private:
        VulkanContext& ctx_;
        uint32_t framesInFlight_ = 0;

        // ngxTried_: ensureNgx() has run, so don't retry (success or failure).
        // ngxInited_: NGX is initialised and owes a matching Shutdown1. Kept
        // separate so a failed init is never shut down, and a partially-failed
        // init that already shut down inline is not shut down twice.
        bool                 ngxTried_  = false;
        bool                 ngxInited_ = false;
        NVSDK_NGX_Parameter* params_    = nullptr;// capability/eval parameter map
        NVSDK_NGX_Handle*    feature_   = nullptr;// the DLSS SR feature
        uint32_t displayW_ = 0, displayH_ = 0;
        int      quality_  = -1;// the created feature's preset (see quality())
        uint32_t evalFails_ = 0;// consecutive evaluate failures (see failing())

        // Ray Reconstruction state (see the RR block above). rrAvailable_ is
        // NGX's SuperSamplingDenoising capability (driver + nvngx_dlssd.dll),
        // read once in ensureNgx(). rrBlocked_ holds RR off after a failed
        // create or an exhausted self-heal, cleared by create().
        NVSDK_NGX_Handle*    featureRR_ = nullptr;
        NVSDK_NGX_Parameter* paramsRR_  = nullptr;// RR's own map (see createRayReconstruction)
        bool     rrAvailable_  = false;
        bool     rrBlocked_    = false;
        uint32_t rrRenderW_ = 0, rrRenderH_ = 0;
        uint32_t rrEvalFails_ = 0;
        // Placeholder guides at the RR render extent: diffuse albedo 1 and
        // specular albedo 0 (rgba8 unorm, cleared once at creation, then
        // SHADER_READ_ONLY_OPTIMAL for good).
        Image2D rrDiffuseAlbedo_;
        Image2D rrSpecularAlbedo_;
        void createGuideImage(Image2D& img, uint32_t width, uint32_t height, const char* name);
        void recordRayReconstruction(const DispatchInputs& in,
                                     NVSDK_NGX_Resource_VK& color, NVSDK_NGX_Resource_VK& depth,
                                     NVSDK_NGX_Resource_VK& motion, NVSDK_NGX_Resource_VK& output,
                                     NVSDK_NGX_Resource_VK* mask);
        void restoreInputs(const DispatchInputs& in);

        // Module directory (wide) kept alive for NGX's PathListInfo.
        std::wstring modulePathW_;
        const wchar_t* modulePathPtrs_[1] = {nullptr};

        bool ensureNgx();// lazy NGX init + capability check
        void shutdownNgx();

        // ── Bias-mask generator (fsr_reactive.comp, shared with FSR) ─────────
        // R8 mask per frame-in-flight at DISPLAY (== maxRenderSize) extent; the
        // compute writes only the render-extent sub-region each frame. Pipeline
        // built once in the ctor; images (re)allocated in create(). Same shape
        // as FsrUpscaler's reactive generator.
        std::vector<Image2D>  reactive_;
        VkSampler             idsSampler_        = VK_NULL_HANDLE;// NEAREST (uint texelFetch)
        VkDescriptorSetLayout reactiveDsLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      reactivePipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            reactivePipe_      = VK_NULL_HANDLE;
        VkDescriptorPool      reactivePool_      = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> reactiveSets_;// [framesInFlight]

        void createReactivePipeline();
        void createReactiveImages(uint32_t width, uint32_t height);
        void destroyReactiveImages();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DLSS_UPSCALER_HPP
