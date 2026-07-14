// DlssUpscaler — NVIDIA DLSS Super Resolution (NGX) temporal upscaler.
//
// A compile-time replacement for the built-in temporal upsampler (TaaResolve /
// taa_resolve.comp) on Windows + Vulkan + RTX, sitting beside (and outranking)
// the FSR 3.1 path: same seam, same inputs, same output slot. Only DLSS
// Super Resolution is used — Frame Generation is deliberately never touched
// (interpolated frames are not valid sensor data for this platform). The same
// NGX plumbing is the doorway to DLSS Ray Reconstruction (nvngx_dlssd) later.
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

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// NGX forward declarations (kept out of the renderer's headers).
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

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
        // renderW/H hint the quality-mode selection (DLAA at scale 1, MaxQuality
        // below); the feature itself is created with maxRenderSize == display so
        // renderScale changes need no recreation, only display resizes do.
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

            // Sub-pixel jitter [-0.5,0.5] applied to the projection this frame.
            // Passed to DLSS AS APPLIED (un-negated) — the OPPOSITE of the FSR
            // dispatch convention on this seam; measured 2.3× more stable
            // (see jitterSign() in the .cpp; THREEPP_DLSS_JITTER_SIGN=neg flips).
            float jitterX = 0.f, jitterY = 0.f;
            // NDC-delta → render-pixel conversion: {0.5*renderW, -0.5*renderH}.
            float motionScaleX = 0.f, motionScaleY = 0.f;

            float frameTimeDeltaMs = 16.6f;
            float preExposure = 1.f;// > 0; the pre-exposure baked into colour
            bool  reset = false;    // camera cut / history invalidation
        };

        // Record the DLSS evaluate into in.cmd. No-op if !valid(). See the
        // layout contract in the class header comment.
        void recordDispatch(const DispatchInputs& in);

    private:
        VulkanContext& ctx_;
        uint32_t framesInFlight_ = 0;

        bool                 ngxInited_ = false;
        NVSDK_NGX_Parameter* params_    = nullptr;// capability/eval parameter map
        NVSDK_NGX_Handle*    feature_   = nullptr;// the DLSS SR feature
        uint32_t displayW_ = 0, displayH_ = 0;

        // Module directory (wide) kept alive for NGX's PathListInfo.
        std::wstring modulePathW_;
        const wchar_t* modulePathPtrs_[1] = {nullptr};

        bool ensureNgx();// lazy NGX init + capability check
        void shutdownNgx();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DLSS_UPSCALER_HPP
