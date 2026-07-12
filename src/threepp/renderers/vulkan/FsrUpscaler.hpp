// FsrUpscaler — AMD FidelityFX FSR 3.1 temporal upscaler (super-resolution).
//
// A compile-time replacement for the built-in temporal upsampler (TaaResolve /
// taa_resolve.comp) on Windows + Vulkan. Only the FSR *upscaler* is used —
// frame generation is deliberately never touched (interpolated frames are not
// valid sensor data for this platform).
//
// This class encapsulates ALL FidelityFX ffx-api usage so the rest of the
// renderer never sees an ffx type. It owns one ffxContext (the upscaler), built
// against the shared VulkanContext device/physical-device via the ffx-api
// Vulkan backend. The prebuilt, signed amd_fidelityfx_vk.dll ships every backend
// + precompiled shader permutation, so there is no shader-compile or
// SDK-from-source build step — CMake fetches the SDK zip and links the import
// lib (see cmake/FetchFidelityFX.cmake).
//
// Lifetime mirrors TaaResolve: constructed once, its ffxContext (re)created for
// the current *display* (swapchain) extent — FSR stores the display extent at
// create time but takes the render extent per-dispatch, so a renderScale change
// needs no recreation, only a swapchain/display resize does.
//
// Layout contract (recordDispatch): the ffx-api Vulkan backend transitions each
// imported resource *from the VkImageLayout implied by the FfxApiResource.state
// we declare* and restores it to that layout afterwards (read states map to
// SHADER_READ_ONLY_OPTIMAL, the UAV output to GENERAL). recordDispatch therefore
// borrows the inputs into SHADER_READ_ONLY_OPTIMAL and restores them to the
// caller-supplied original layouts itself — the caller hands over images in
// their steady-state layouts and gets them back unchanged.

#ifndef THREEPP_VULKAN_FSR_UPSCALER_HPP
#define THREEPP_VULKAN_FSR_UPSCALER_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class FsrUpscaler {

    public:
        explicit FsrUpscaler(VulkanContext& ctx);
        ~FsrUpscaler();
        FsrUpscaler(const FsrUpscaler&) = delete;
        FsrUpscaler& operator=(const FsrUpscaler&) = delete;

        // (Re)create the upscaler context for the given display (swapchain)
        // extent. Idempotent — destroys any existing context first. Returns
        // true on success; on failure the context stays null (valid() == false)
        // and the caller keeps the built-in TAA path. maxRenderSize is set to
        // the display extent so any renderScale in (0,1] is a valid per-dispatch
        // renderSize with no recreation.
        bool create(uint32_t displayWidth, uint32_t displayHeight);
        void destroy();
        [[nodiscard]] bool valid() const { return context_ != nullptr; }
        [[nodiscard]] uint32_t displayWidth() const { return displayW_; }
        [[nodiscard]] uint32_t displayHeight() const { return displayH_; }

        // Jitter sequence, straight from FSR so the projection jitter and the
        // dispatch jitterOffset come from one source. jitterPhaseCount tracks
        // the render/display ratio (FSR's Halton period); jitterOffset returns
        // the sub-pixel offset in [-0.5, +0.5] for a sequence index. The SAME
        // (x,y) must be applied to the projection (as the renderer already does
        // for Halton) and passed back in DispatchInputs (recordDispatch negates
        // it, matching the FSR reference integration).
        [[nodiscard]] int jitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth) const;
        void jitterOffset(int index, int phaseCount, float& outX, float& outY) const;

        // All render-extent inputs unless noted. Formats are the images' real
        // VkFormats (color rgba16f linear-HDR, depth D32 reversed-Z, motion
        // rg16f-in-rgba16f NDC-delta, output rgba16f display extent). Layouts
        // are the images' CURRENT (steady-state) layouts; recordDispatch
        // transitions them into/out of the FFX read/UAV states and restores
        // these exact layouts before returning. Motion whose current layout is
        // already SHADER_READ_ONLY_OPTIMAL costs no extra barrier.
        struct DispatchInputs {
            VkCommandBuffer cmd = VK_NULL_HANDLE;

            VkImage      colorImage  = VK_NULL_HANDLE;
            VkFormat     colorFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout colorLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkImage      depthImage  = VK_NULL_HANDLE;
            VkFormat     depthFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkImage      motionImage  = VK_NULL_HANDLE;
            VkFormat     motionFormat = VK_FORMAT_UNDEFINED;
            VkImageLayout motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            // Output at DISPLAY extent, storage image (UAV → GENERAL).
            VkImage      outputImage  = VK_NULL_HANDLE;
            VkFormat     outputFormat = VK_FORMAT_UNDEFINED;

            uint32_t renderWidth = 0, renderHeight = 0;
            uint32_t displayWidth = 0, displayHeight = 0;

            // Sub-pixel jitter [-0.5,0.5] applied to the projection this frame
            // (recordDispatch negates it for the dispatch, per FSR convention).
            float jitterX = 0.f, jitterY = 0.f;
            // NDC-delta → render-pixel conversion: {0.5*renderW, -0.5*renderH}.
            float motionScaleX = 0.f, motionScaleY = 0.f;

            float nearPlane = 0.1f, farPlane = 1000.f, fovYRadians = 1.0f;
            float frameTimeDeltaMs = 16.6f;
            float preExposure = 1.f;// > 0; the pre-exposure baked into colour
            bool  reset = false;    // camera cut / history invalidation

            bool  sharpen = false;  // FSR's built-in RCAS (left off — the
            float sharpness = 0.f;  // renderer keeps its own display RCAS)
        };

        // Record the FSR upscale into in.cmd. No-op if !valid(). See the layout
        // contract in the class header comment.
        void recordDispatch(const DispatchInputs& in);

    private:
        VulkanContext& ctx_;
        void*    context_ = nullptr;// ffxContext (opaque; kept ffx-type-free here)
        uint32_t displayW_ = 0, displayH_ = 0;
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_FSR_UPSCALER_HPP
