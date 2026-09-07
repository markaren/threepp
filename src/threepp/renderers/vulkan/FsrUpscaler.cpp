#include "threepp/renderers/vulkan/FsrUpscaler.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/fsr_reactive.comp.spv.h"

#include <vector>

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
#include <string>

namespace threepp::vulkan {

    namespace {

        // Process-wide ffx-api entry points (the DLL is a singleton; one renderer
        // + one FSR context). Loaded once, lazily, on the first create().
        ffxFunctions g_ffx{};
        bool         g_ffxTriedLoad = false;

        bool ensureFfxLoaded() {
            if (g_ffxTriedLoad) return g_ffx.CreateContext != nullptr;
            g_ffxTriedLoad = true;
            // Load the DLL from the directory of the module that contains THIS code
            // — the .exe for a C++ build, or threepp_py.pyd under the Python
            // bindings. A .pyd's runtime LoadLibrary does NOT search its own
            // directory (Python 3.8+ DLL rules), so a bare name fails there even
            // with the DLL sitting next to the module; resolving the full path
            // works in both cases. Falls back to the default search order.
            HMODULE m = nullptr;
            {
                HMODULE self = nullptr;
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       reinterpret_cast<LPCSTR>(&ensureFfxLoaded), &self) && self) {
                    char path[MAX_PATH];
                    const DWORD n = GetModuleFileNameA(self, path, MAX_PATH);
                    if (n > 0 && n < MAX_PATH) {
                        std::string p(path, n);
                        const auto slash = p.find_last_of("\\/");
                        if (slash != std::string::npos)
                            m = LoadLibraryA((p.substr(0, slash + 1) + "amd_fidelityfx_vk.dll").c_str());
                    }
                }
            }
            if (!m) m = LoadLibraryA("amd_fidelityfx_vk.dll");// default search order
            if (!m) {
                std::fprintf(stderr,
                             "[threepp] FSR: LoadLibrary(amd_fidelityfx_vk.dll) failed "
                             "(err %lu) - falling back to the built-in TAA upscaler.\n",
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

    FsrUpscaler::FsrUpscaler(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        reactive_.resize(framesInFlight_);
        reactiveSets_.resize(framesInFlight_);
        createReactivePipeline();
    }

    FsrUpscaler::~FsrUpscaler() {
        destroy();// frees the ffx context + reactive images (device still alive)
        VkDevice d = ctx_.device();
        if (reactivePipe_)       vkDestroyPipeline(d, reactivePipe_, nullptr);
        if (reactivePipeLayout_) vkDestroyPipelineLayout(d, reactivePipeLayout_, nullptr);
        if (reactiveDsLayout_)   vkDestroyDescriptorSetLayout(d, reactiveDsLayout_, nullptr);
        if (reactivePool_)       vkDestroyDescriptorPool(d, reactivePool_, nullptr);
        if (idsSampler_)         vkDestroySampler(d, idsSampler_, nullptr);
    }

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
                         "[threepp] FSR: ffxCreateContext failed (code %u) - "
                         "falling back to the built-in TAA upscaler.\n",
                         static_cast<unsigned>(rc));
            context_ = nullptr;
            return false;
        }
        context_  = context;
        displayW_ = displayWidth;
        displayH_ = displayHeight;
        // Reactive mask scratch at the DISPLAY (== maxRenderSize) extent — the
        // compute writes only the render sub-region each frame.
        createReactiveImages(displayWidth, displayHeight);
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
        destroyReactiveImages();
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

        // ── Reactive mask (opt-in) ───────────────────────────────────────────
        // Generate an R8 reactive mask from the G-buffer IDs flags and hand it to
        // FSR so deformer/animated surfaces trust history less (less ghosting).
        // Per-frame-in-flight image + set; the set is safe to update here because
        // this fif slot's fence was already waited this frame (never in-flight).
        VkImage reactiveImg = VK_NULL_HANDLE;
        if (in.reactive && reactivePipe_ && in.idsView != VK_NULL_HANDLE &&
            in.frame < reactive_.size() && reactive_[in.frame].view != VK_NULL_HANDLE) {
            reactiveImg = reactive_[in.frame].image;

            VkDescriptorImageInfo idsI{};
            idsI.sampler     = idsSampler_;
            idsI.imageView   = in.idsView;
            idsI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo reI{};
            reI.imageView    = reactive_[in.frame].view;
            reI.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w[2]{};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = reactiveSets_[in.frame]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &idsI;
            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = reactiveSets_[in.frame]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &reI;
            vkUpdateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);

            // Regenerated in full every frame → UNDEFINED old layout (discard).
            transition(in.cmd, reactiveImg, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
            vkCmdBindPipeline(in.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reactivePipe_);
            vkCmdBindDescriptorSets(in.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    reactivePipeLayout_, 0, 1, &reactiveSets_[in.frame], 0, nullptr);
            struct { uint32_t w, h; float v; } pc{in.renderWidth, in.renderHeight, in.reactiveValue};
            vkCmdPushConstants(in.cmd, reactivePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDispatch(in.cmd, (in.renderWidth + 7u) / 8u, (in.renderHeight + 7u) / 8u, 1u);
            // Compute write (GENERAL) → FSR read (SHADER_READ; declared COMPUTE_READ).
            transition(in.cmd, reactiveImg, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        }

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
        // Auto-exposure: no exposure texture. Reactive mask fed when generated
        // above (else a null resource = "no reactive"); no T&C mask (transparents
        // composite after FSR, so there's nothing to feed here).
        d.exposure      = wrapImage(VK_NULL_HANDLE, VK_FORMAT_UNDEFINED, 0, 0, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.reactive      = wrapImage(reactiveImg,
                                    reactiveImg ? VK_FORMAT_R8_UNORM : VK_FORMAT_UNDEFINED,
                                    in.renderWidth, in.renderHeight, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.transparencyAndComposition = wrapImage(VK_NULL_HANDLE, VK_FORMAT_UNDEFINED, 0, 0, false,
                                    FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output        = wrapImage(in.outputImage, in.outputFormat,
                                    in.displayWidth, in.displayHeight, true,
                                    FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        // Jitter offset — FSR's convention is the IMAGE-CONTENT shift in y-DOWN
        // pixel space: its reference application is a clip-space translation of
        // (+2jx/W, −2jy/H), which moves content by (+jx, +jy) pixels. threepp
        // applies the same raw (jx, jy) as m02/m12 += 2j/extent in GL y-UP clip
        // space, which moves content by (−jx, +jy) pixels — so the conversion
        // is per-axis, NOT a plain negation. The old {-jx, -jy} had X right and
        // Y flipped: FSR mis-anchored every frame's samples by 2·jy vertically,
        // which surfaced as the 8-phase vertical tremble ("shaking") on the FSR
        // path. Same seam as the motion vectors, which already do the y-flip
        // via motionScaleY = -0.5H.
        d.jitterOffset       = {-in.jitterX, +in.jitterY};
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

    void FsrUpscaler::createReactivePipeline() {
        VkDevice d = ctx_.device();

        // NEAREST sampler for the uint IDs (texelFetch ignores filtering, but a
        // combined-image-sampler binding still needs a sampler object).
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &idsSampler_), "vkCreateSampler(fsr.reactive.ids)");

        // 0: gbuf IDs (usampler2D), 1: reactive R8 (storage).
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2; dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &reactiveDsLayout_),
              "vkCreateDescriptorSetLayout(fsr.reactive)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 12;// w,h,value
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &reactiveDsLayout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &reactivePipeLayout_),
              "vkCreatePipelineLayout(fsr.reactive)");

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kFsrReactiveCompSpv); smci.pCode = kFsrReactiveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(fsr.reactive)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = mod; stage.pName = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage; cpci.layout = reactivePipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &reactivePipe_),
              "vkCreateComputePipelines(fsr.reactive)");
        vkDestroyShaderModule(d, mod, nullptr);

        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = framesInFlight_;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = framesInFlight_; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &reactivePool_),
              "vkCreateDescriptorPool(fsr.reactive)");
        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, reactiveDsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = reactivePool_; ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(d, &ai, reactiveSets_.data()),
              "vkAllocateDescriptorSets(fsr.reactive)");
    }

    void FsrUpscaler::createReactiveImages(uint32_t width, uint32_t height) {
        destroyReactiveImages();
        VkDevice d = ctx_.device();
        for (auto& img : reactive_) {
            img.width = width; img.height = height; img.format = VK_FORMAT_R8_UNORM;
            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = VK_FORMAT_R8_UNORM;
            ici.extent        = {width, height, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;// recordDispatch does UNDEFINED→GENERAL
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(fsr.reactive)");
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = VK_FORMAT_R8_UNORM;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(d, &vci, nullptr, &img.view), "vkCreateImageView(fsr.reactive)");
            ctx_.setObjectName(img.image, "fsr.reactiveMask");
            ctx_.setObjectName(img.view, "fsr.reactiveMask");
        }
    }

    void FsrUpscaler::destroyReactiveImages() {
        VkDevice d = ctx_.device();
        for (auto& img : reactive_) destroyImage2D(ctx_.allocator(), d, img);
    }

}// namespace threepp::vulkan
