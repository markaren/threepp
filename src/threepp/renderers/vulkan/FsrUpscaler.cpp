#include "threepp/renderers/vulkan/FsrUpscaler.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

// The ffx-api entry points MUST be resolved via GetProcAddress, not the import
// library: ffx_api.h hard-codes FFX_API_ENTRY = __declspec(dllexport) even for
// consumers (no #ifndef guard), so linking + calling them directly makes the
// linker emit a null forwarder and the call executes at address 0. ffxLoadFunctions
// (ffx_api_loader.h) does LoadLibrary + GetProcAddress, which is the SDK's
// intended consumption path. NOMINMAX/WIN32_LEAN_AND_MEAN keep <windows.h> from
// polluting the TU (it also pulls in vulkan headers).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <ffx_api/ffx_api_loader.h>

#include <cstdio>

namespace threepp::vulkan {

    namespace {

        // Process-wide ffx-api entry points (the DLL is a singleton; one renderer
        // + one FSR context). Loaded once, lazily, on the first create().
        ffxFunctions g_ffx{};
        bool         g_ffxTriedLoad = false;

        bool ensureFfxLoaded() {
            if (g_ffxTriedLoad) return g_ffx.CreateContext != nullptr;
            g_ffxTriedLoad = true;
            // The DLL is copied next to the executable (CMake POST_BUILD), so the
            // default search order (exe dir first) finds it.
            HMODULE m = LoadLibraryA("amd_fidelityfx_vk.dll");
            if (!m) {
                std::fprintf(stderr,
                             "[threepp] FSR: LoadLibrary(amd_fidelityfx_vk.dll) failed "
                             "(err %lu) — falling back to the built-in TAA upscaler.\n",
                             static_cast<unsigned long>(GetLastError()));
                return false;
            }
            ffxLoadFunctions(&g_ffx, m);
            if (!g_ffx.CreateContext) {
                std::fprintf(stderr, "[threepp] FSR: amd_fidelityfx_vk.dll missing ffx entry points.\n");
                return false;
            }
            return true;
        }

        // Diagnostic message sink for the ffx runtime. MUST be non-null: some
        // ffx create/validation paths invoke the message callback unconditionally,
        // so leaving it null crashes (a null call) instead of reporting the issue.
        void ffxMessageSink(uint32_t type, const wchar_t* message) {
            std::fprintf(stderr, "[threepp] FSR runtime %s: %ls\n",
                         type == FFX_API_MESSAGE_TYPE_ERROR ? "ERROR" : "WARNING",
                         message ? message : L"(null)");
            std::fflush(stderr);
        }

        // Wrap a threepp VkImage as an FfxApiResource. The ffx-api Vulkan backend
        // creates its own image views, so it only needs the VkImage handle plus a
        // description synthesized from the format/extent/usage (mirrors what
        // ffxApiGetImageResourceDescriptionVK reads out of a VkImageCreateInfo).
        // `storage` marks the output (UAV) so the description carries the UAV
        // usage flag; inputs pass false (sampled read).
        FfxApiResource wrapImage(VkImage image, VkFormat format,
                                 uint32_t width, uint32_t height,
                                 bool storage, uint32_t state) {
            VkImageCreateInfo ci{};
            ci.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ci.imageType   = VK_IMAGE_TYPE_2D;
            ci.format      = format;
            ci.extent      = {width, height, 1};
            ci.mipLevels   = 1;
            ci.arrayLayers = 1;
            ci.samples     = VK_SAMPLE_COUNT_1_BIT;
            ci.usage       = storage ? VK_IMAGE_USAGE_STORAGE_BIT
                                     : VK_IMAGE_USAGE_SAMPLED_BIT;
            const FfxApiResourceDescription desc =
                    ffxApiGetImageResourceDescriptionVK(image, ci, 0);
            return ffxApiGetResourceVK(reinterpret_cast<void*>(image), desc, state);
        }

        // Once-per-frame image layout transition for the FSR borrow/restore. Broad
        // stage/access masks (correctness over micro-optimisation — this is a
        // single barrier per resource per frame). Restores use UNDEFINED as the
        // old layout: the input contents are not needed after the upscale (colour
        // and depth are regenerated next frame), so we don't depend on whatever
        // working layout the ffx backend left them in.
        void transition(VkCommandBuffer cb, VkImage image,
                        VkImageLayout oldLayout, VkImageLayout newLayout,
                        VkImageAspectFlags aspect) {
            VkImageMemoryBarrier2 b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.oldLayout     = oldLayout;
            b.newLayout     = newLayout;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image         = image;
            b.subresourceRange.aspectMask = aspect;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cb, &dep);
        }

    }// namespace

    FsrUpscaler::FsrUpscaler(VulkanContext& ctx) : ctx_(ctx) {}

    FsrUpscaler::~FsrUpscaler() { destroy(); }

    bool FsrUpscaler::create(uint32_t displayWidth, uint32_t displayHeight) {
        destroy();
        if (displayWidth == 0 || displayHeight == 0) return false;
        if (!ensureFfxLoaded()) return false;

        // Vulkan backend: hand the ffx-api the shared device/physical-device and
        // the standard device-proc-addr loader (the renderer already resolves
        // extension entrypoints through the same global symbol).
        ffxCreateBackendVKDesc backendDesc{};
        backendDesc.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backendDesc.vkDevice         = ctx_.device();
        backendDesc.vkPhysicalDevice = ctx_.physicalDevice();
        backendDesc.vkDeviceProcAddr = vkGetDeviceProcAddr;

        ffxCreateContextDescUpscale createDesc{};
        createDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        // Linear-HDR colour, reversed-Z depth, auto-exposure (the renderer only
        // has a host-side exposure scalar — pre-exposure is passed per dispatch),
        // and dynamic render size (renderScale can change at runtime; maxRenderSize
        // == display extent covers any ratio in (0,1]). Motion is render-resolution
        // and jitter-free, so neither DISPLAY_RESOLUTION_MOTION_VECTORS nor
        // MOTION_VECTORS_JITTER_CANCELLATION is set.
        createDesc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                           FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                           FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
                           FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
        createDesc.maxRenderSize  = {displayWidth, displayHeight};
        createDesc.maxUpscaleSize = {displayWidth, displayHeight};
        createDesc.fpMessage      = ffxMessageSink;// non-null — see ffxMessageSink
        createDesc.header.pNext   = &backendDesc.header;

        ffxContext context = nullptr;
        const ffxReturnCode_t rc = g_ffx.CreateContext(&context, &createDesc.header, nullptr);
        if (rc != FFX_API_RETURN_OK || context == nullptr) {
            std::fprintf(stderr,
                         "[threepp] FSR: ffxCreateContext failed (code %u) — "
                         "falling back to the built-in TAA upscaler.\n",
                         static_cast<unsigned>(rc));
            context_ = nullptr;
            return false;
        }
        context_  = context;
        displayW_ = displayWidth;
        displayH_ = displayHeight;
        std::fprintf(stderr, "[threepp] FSR 3.1 upscaler active (%ux%u display).\n",
                     displayWidth, displayHeight);
        return true;
    }

    void FsrUpscaler::destroy() {
        if (context_) {
            ffxContext c = context_;
            if (g_ffx.DestroyContext) g_ffx.DestroyContext(&c, nullptr);
            context_ = nullptr;
        }
        displayW_ = displayH_ = 0;
    }

    int FsrUpscaler::jitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth) const {
        if (!context_) return 8;// sane Halton fallback when FSR is unavailable
        int32_t phaseCount = 0;
        ffxQueryDescUpscaleGetJitterPhaseCount q{};
        q.header.type     = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
        q.renderWidth     = renderWidth;
        q.displayWidth    = displayWidth;
        q.pOutPhaseCount  = &phaseCount;
        ffxContext c = context_;
        if (!g_ffx.Query || g_ffx.Query(&c, &q.header) != FFX_API_RETURN_OK || phaseCount <= 0) return 8;
        return phaseCount;
    }

    void FsrUpscaler::jitterOffset(int index, int phaseCount, float& outX, float& outY) const {
        outX = outY = 0.f;
        if (!context_ || phaseCount <= 0) return;
        ffxQueryDescUpscaleGetJitterOffset q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
        q.index       = index;
        q.phaseCount  = phaseCount;
        q.pOutX       = &outX;
        q.pOutY       = &outY;
        ffxContext c = context_;
        if (g_ffx.Query) g_ffx.Query(&c, &q.header);
    }

    void FsrUpscaler::recordDispatch(const DispatchInputs& in) {
        if (!context_) return;

        // Borrow the inputs into the layout the ffx backend expects for a
        // read state (SHADER_READ_ONLY_OPTIMAL). Motion is already there.
        transition(in.cmd, in.colorImage, in.colorLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(in.cmd, in.depthImage, in.depthLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (in.motionLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            transition(in.cmd, in.motionImage, in.motionLayout,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        ffxDispatchDescUpscale d{};
        d.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList   = reinterpret_cast<void*>(in.cmd);
        d.color         = wrapImage(in.colorImage,  in.colorFormat,
                                    in.renderWidth,  in.renderHeight, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth         = wrapImage(in.depthImage,  in.depthFormat,
                                    in.renderWidth,  in.renderHeight, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = wrapImage(in.motionImage, in.motionFormat,
                                    in.renderWidth,  in.renderHeight, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        // Auto-exposure: no exposure texture. No reactive / T&C masks (deferred).
        d.exposure      = wrapImage(VK_NULL_HANDLE, VK_FORMAT_UNDEFINED, 0, 0, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.reactive      = wrapImage(VK_NULL_HANDLE, VK_FORMAT_UNDEFINED, 0, 0, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.transparencyAndComposition = wrapImage(VK_NULL_HANDLE, VK_FORMAT_UNDEFINED, 0, 0, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output        = wrapImage(in.outputImage, in.outputFormat,
                                    in.displayWidth, in.displayHeight, true,
                                    FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        // Jitter offset is the negated sub-pixel projection jitter (FSR reference
        // convention). Motion is NDC-delta → render-pixel via {0.5W, -0.5H}.
        d.jitterOffset       = {-in.jitterX, -in.jitterY};
        d.motionVectorScale  = {in.motionScaleX, in.motionScaleY};
        d.renderSize         = {in.renderWidth,  in.renderHeight};
        d.upscaleSize        = {in.displayWidth, in.displayHeight};
        d.enableSharpening   = in.sharpen;
        d.sharpness          = in.sharpness;
        d.frameTimeDelta     = in.frameTimeDeltaMs > 0.f ? in.frameTimeDeltaMs : 16.6f;
        d.preExposure        = in.preExposure > 0.f ? in.preExposure : 1.f;
        d.reset              = in.reset;
        d.cameraNear         = in.nearPlane;
        d.cameraFar          = in.farPlane;
        d.cameraFovAngleVertical = in.fovYRadians;
        d.viewSpaceToMetersFactor = 1.f;
        d.flags              = 0;

        if (g_ffx.Dispatch) g_ffx.Dispatch(&context_, &d.header);

        // Restore the borrowed inputs to their steady-state layouts. UNDEFINED old
        // layout: their contents aren't needed after the upscale (see transition()).
        transition(in.cmd, in.colorImage, VK_IMAGE_LAYOUT_UNDEFINED,
                   in.colorLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(in.cmd, in.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
                   in.depthLayout, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (in.motionLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            transition(in.cmd, in.motionImage, VK_IMAGE_LAYOUT_UNDEFINED,
                       in.motionLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        // The output is left in GENERAL with the upscale's writes; the caller
        // barriers it before PostComposite reads it (threepp's existing pattern
        // for the resolve → post-composite hand-off).
    }

}// namespace threepp::vulkan
