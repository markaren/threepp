
#include "threepp/renderers/vulkan/DlssUpscaler.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/fsr_reactive.comp.spv.h"// shared bias-mask generator

#include <cstdio>
#include <cstdlib>
#include <cstring>

// NOMINMAX/WIN32_LEAN_AND_MEAN keep <windows.h> (needed for the module-dir
// lookup) from polluting the TU.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

namespace threepp::vulkan {

    // Free-function alias for VulkanContext.cpp (which can't include this
    // class's header — see the declaration there).
    std::vector<const char*> dlssRequiredDeviceExtensions() {
        return DlssUpscaler::requiredDeviceExtensions();
    }

    namespace {

        // Stable project identity for NGX (NVSDK_NGX_ENGINE_TYPE_CUSTOM). Any
        // consistent UUID works for an unregistered application; NGX uses it to
        // key driver-side per-app data (model overrides, logs).
        constexpr const char* kNgxProjectId    = "6e7b4a92-31c5-4f8e-9a0d-8f2b1c6d5e43";
        constexpr const char* kNgxEngineVersion = "1.0";

        // Directory of the module containing this code — the .exe for a C++
        // build, or threepp_py.pyd under the Python bindings. NGX searches the
        // process executable's directory for nvngx_dlss.dll by default, which
        // is wrong for the .pyd case; passing this directory as an explicit
        // search path covers both (same trick as FsrUpscaler's DLL load).
        std::wstring moduleDirW() {
            HMODULE self = nullptr;
            if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                    reinterpret_cast<LPCWSTR>(&moduleDirW), &self) ||
                !self) return L".";
            wchar_t path[MAX_PATH];
            const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
            if (n == 0 || n >= MAX_PATH) return L".";
            std::wstring p(path, n);
            const auto slash = p.find_last_of(L"\\/");
            return slash == std::wstring::npos ? L"." : p.substr(0, slash);
        }

        // Once-per-frame image layout transition for the NGX borrow/restore.
        // Broad stage/access masks (one barrier per resource per frame) — same
        // shape as FsrUpscaler's. Restores use UNDEFINED as the old layout: the
        // input contents are not needed after the upscale.
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

        NVSDK_NGX_Resource_VK wrapView(VkImageView view, VkImage image, VkFormat format,
                                       uint32_t width, uint32_t height,
                                       VkImageAspectFlags aspect, bool readWrite) {
            VkImageSubresourceRange range{};
            range.aspectMask     = aspect;
            range.baseMipLevel   = 0;
            range.levelCount     = 1;
            range.baseArrayLayer = 0;
            range.layerCount     = 1;
            return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, range, format,
                                                          width, height, readWrite);
        }

        // Jitter sign: DLSS takes the projection jitter AS APPLIED (un-negated) —
        // the opposite of the FSR dispatch convention on this same seam. Measured
        // on the ABeautifulGame static 16-frame sequence: un-negated is 2.3× more
        // temporally stable on the model crop (0.159 vs 0.368 mean |Δ|); negated
        // mis-reprojects every frame. THREEPP_DLSS_JITTER_SIGN=neg flips it back
        // for debugging.
        float jitterSign() {
            static const float s = [] {
                const char* v = std::getenv("THREEPP_DLSS_JITTER_SIGN");
                return (v && std::strcmp(v, "neg") == 0) ? -1.f : 1.f;
            }();
            return s;
        }

    }// namespace

    DlssUpscaler::DlssUpscaler(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        reactive_.resize(framesInFlight_);
        reactiveSets_.resize(framesInFlight_);
        createReactivePipeline();
    }

    DlssUpscaler::~DlssUpscaler() {
        destroy();
        shutdownNgx();
        VkDevice d = ctx_.device();
        if (reactivePipe_)       vkDestroyPipeline(d, reactivePipe_, nullptr);
        if (reactivePipeLayout_) vkDestroyPipelineLayout(d, reactivePipeLayout_, nullptr);
        if (reactiveDsLayout_)   vkDestroyDescriptorSetLayout(d, reactiveDsLayout_, nullptr);
        if (reactivePool_)       vkDestroyDescriptorPool(d, reactivePool_, nullptr);
        if (idsSampler_)         vkDestroySampler(d, idsSampler_, nullptr);
    }

    std::vector<const char*> DlssUpscaler::requiredDeviceExtensions() {
        unsigned int instCount = 0, devCount = 0;
        const char** instExts = nullptr;
        const char** devExts  = nullptr;
        const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(
                &instCount, const_cast<const char***>(&instExts),
                &devCount, const_cast<const char***>(&devExts));
        std::vector<const char*> out;
        if (NVSDK_NGX_FAILED(r) || !devExts) return out;
        out.reserve(devCount);
        for (unsigned int i = 0; i < devCount; ++i) out.push_back(devExts[i]);
        return out;// pointers into the NGX lib's static tables — stable
    }

    bool DlssUpscaler::ensureNgx() {
        if (ngxInited_) return params_ != nullptr;
        ngxInited_ = true;

        modulePathW_       = moduleDirW();
        modulePathPtrs_[0] = modulePathW_.c_str();
        NVSDK_NGX_FeatureCommonInfo fci{};
        fci.PathListInfo.Path   = modulePathPtrs_;
        fci.PathListInfo.Length = 1;

        NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
                kNgxProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kNgxEngineVersion,
                modulePathW_.c_str(),// application data path (NGX logs)
                ctx_.instance(), ctx_.physicalDevice(), ctx_.device(),
                vkGetInstanceProcAddr, vkGetDeviceProcAddr,
                &fci, NVSDK_NGX_Version_API);
        if (NVSDK_NGX_FAILED(r)) {
            std::fprintf(stderr,
                         "[threepp] DLSS: NGX init failed (0x%08x) — "
                         "falling back to FSR/TAA.\n", static_cast<unsigned>(r));
            return false;
        }

        // Capability check: driver present + DLSS supported on this GPU.
        NVSDK_NGX_Parameter* caps = nullptr;
        r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps);
        int available = 0;
        if (NVSDK_NGX_SUCCEED(r) && caps) {
            NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
            NVSDK_NGX_VULKAN_DestroyParameters(caps);
        }
        if (!available) {
            std::fprintf(stderr,
                         "[threepp] DLSS: not available on this GPU/driver — "
                         "falling back to FSR/TAA.\n");
            NVSDK_NGX_VULKAN_Shutdown1(ctx_.device());
            return false;
        }

        r = NVSDK_NGX_VULKAN_AllocateParameters(&params_);
        if (NVSDK_NGX_FAILED(r) || !params_) {
            std::fprintf(stderr, "[threepp] DLSS: parameter allocation failed (0x%08x).\n",
                         static_cast<unsigned>(r));
            params_ = nullptr;
            NVSDK_NGX_VULKAN_Shutdown1(ctx_.device());
            return false;
        }
        return true;
    }

    void DlssUpscaler::shutdownNgx() {
        if (params_) {
            NVSDK_NGX_VULKAN_DestroyParameters(params_);
            params_ = nullptr;
        }
        if (ngxInited_) {
            NVSDK_NGX_VULKAN_Shutdown1(ctx_.device());
            ngxInited_ = false;
        }
    }

    bool DlssUpscaler::create(VkCommandBuffer initCb,
                              uint32_t displayWidth, uint32_t displayHeight,
                              uint32_t renderWidth, uint32_t renderHeight) {
        destroy();
        if (displayWidth == 0 || displayHeight == 0) return false;
        if (!ensureNgx()) return false;

        // Quality mode is a model-selection hint keyed to the CURRENT upscale
        // ratio; the feature itself is created with maxRenderSize == display
        // (dynamic render subrects), so later renderScale changes stay valid
        // without recreation — only the hint goes stale, which is benign.
        const float ratio = renderWidth > 0
                                    ? static_cast<float>(renderWidth) / static_cast<float>(displayWidth)
                                    : 1.f;
        NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
        if (ratio >= 0.99f)      quality = NVSDK_NGX_PerfQuality_Value_DLAA;
        else if (ratio >= 0.62f) quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
        else if (ratio >= 0.55f) quality = NVSDK_NGX_PerfQuality_Value_Balanced;
        else if (ratio >= 0.45f) quality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
        else                     quality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;

        NVSDK_NGX_DLSS_Create_Params cp{};
        cp.Feature.InWidth            = displayWidth; // max render size (dynamic subrect)
        cp.Feature.InHeight           = displayHeight;
        cp.Feature.InTargetWidth      = displayWidth;
        cp.Feature.InTargetHeight     = displayHeight;
        cp.Feature.InPerfQualityValue = quality;
        // Linear-HDR colour, render-res unjittered motion, reversed-Z depth,
        // auto-exposure (pre-exposure scalar passed per dispatch) — the exact
        // input contract the FSR path already validated on this seam.
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                  NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                  NVSDK_NGX_DLSS_Feature_Flags_DepthInverted |
                                  NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

        const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(
                ctx_.device(), initCb, 1u, 1u, &feature_, params_, &cp);
        if (NVSDK_NGX_FAILED(r) || !feature_) {
            std::fprintf(stderr,
                         "[threepp] DLSS: feature creation failed (0x%08x) — "
                         "falling back to FSR/TAA.\n", static_cast<unsigned>(r));
            feature_ = nullptr;
            return false;
        }
        displayW_ = displayWidth;
        displayH_ = displayHeight;
        evalFails_ = 0;// fresh feature — clear any sticky-failure state
        // Bias-mask scratch at the DISPLAY (== maxRenderSize) extent — the
        // compute writes only the render sub-region each frame.
        createReactiveImages(displayWidth, displayHeight);
        std::fprintf(stderr, "[threepp] DLSS upscaler active (%ux%u display, quality mode %d).\n",
                     displayWidth, displayHeight, static_cast<int>(quality));
        return true;
    }

    void DlssUpscaler::destroy() {
        if (feature_) {
            NVSDK_NGX_VULKAN_ReleaseFeature(feature_);
            feature_ = nullptr;
        }
        destroyReactiveImages();
        displayW_ = displayH_ = 0;
    }

    void DlssUpscaler::recordDispatch(const DispatchInputs& in) {
        if (!feature_ || !params_) return;

        // ── Bias-current-color mask (opt-in) ─────────────────────────────────
        // Generate an R8 mask from the G-buffer IDs flags and hand it to DLSS so
        // deformer/wind-animated surfaces (grass) favor the current frame —
        // their shader displacement isn't in the motion vectors, so history
        // GHOSTS at their edges without this. Mirrors FsrUpscaler's reactive
        // mask (same shader, same per-fif set update safety: this fif's fence
        // was already waited this frame).
        VkImage     maskImg  = VK_NULL_HANDLE;
        VkImageView maskView = VK_NULL_HANDLE;
        if (in.reactive && reactivePipe_ && in.idsView != VK_NULL_HANDLE &&
            in.frame < reactive_.size() && reactive_[in.frame].view != VK_NULL_HANDLE) {
            maskImg  = reactive_[in.frame].image;
            maskView = reactive_[in.frame].view;

            VkDescriptorImageInfo idsI{};
            idsI.sampler     = idsSampler_;
            idsI.imageView   = in.idsView;
            idsI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo reI{};
            reI.imageView    = maskView;
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
            transition(in.cmd, maskImg, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
            vkCmdBindPipeline(in.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reactivePipe_);
            vkCmdBindDescriptorSets(in.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    reactivePipeLayout_, 0, 1, &reactiveSets_[in.frame], 0, nullptr);
            struct { uint32_t w, h; float v; } pcm{in.renderWidth, in.renderHeight, in.reactiveValue};
            vkCmdPushConstants(in.cmd, reactivePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pcm), &pcm);
            vkCmdDispatch(in.cmd, (in.renderWidth + 7u) / 8u, (in.renderHeight + 7u) / 8u, 1u);
            // Compute write (GENERAL) → DLSS sampled read.
            transition(in.cmd, maskImg, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // Borrow the inputs into NGX's expected sampled-read layout. The output
        // (TAA history slot) already lives in GENERAL — NGX's UAV layout.
        transition(in.cmd, in.colorImage, in.colorLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(in.cmd, in.depthImage, in.depthLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (in.motionLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            transition(in.cmd, in.motionImage, in.motionLayout,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        NVSDK_NGX_Resource_VK color = wrapView(in.colorView, in.colorImage, in.colorFormat,
                                               in.renderWidth, in.renderHeight,
                                               VK_IMAGE_ASPECT_COLOR_BIT, false);
        NVSDK_NGX_Resource_VK depth = wrapView(in.depthView, in.depthImage, in.depthFormat,
                                               in.renderWidth, in.renderHeight,
                                               VK_IMAGE_ASPECT_DEPTH_BIT, false);
        NVSDK_NGX_Resource_VK motion = wrapView(in.motionView, in.motionImage, in.motionFormat,
                                                in.renderWidth, in.renderHeight,
                                                VK_IMAGE_ASPECT_COLOR_BIT, false);
        NVSDK_NGX_Resource_VK output = wrapView(in.outputView, in.outputImage, in.outputFormat,
                                                in.displayWidth, in.displayHeight,
                                                VK_IMAGE_ASPECT_COLOR_BIT, true);

        NVSDK_NGX_VK_DLSS_Eval_Params ev{};
        ev.Feature.pInColor  = &color;
        ev.Feature.pInOutput = &output;
        ev.Feature.InSharpness = 0.f;// the renderer keeps its own display RCAS
        ev.pInDepth          = &depth;
        ev.pInMotionVectors  = &motion;
        const float s        = jitterSign();
        ev.InJitterOffsetX   = s * in.jitterX;
        ev.InJitterOffsetY   = s * in.jitterY;
        ev.InRenderSubrectDimensions = {in.renderWidth, in.renderHeight};
        ev.InReset           = in.reset ? 1 : 0;
        ev.InMVScaleX        = in.motionScaleX;
        ev.InMVScaleY        = in.motionScaleY;
        ev.InPreExposure     = in.preExposure > 0.f ? in.preExposure : 1.f;
        ev.InFrameTimeDeltaInMsec = in.frameTimeDeltaMs > 0.f ? in.frameTimeDeltaMs : 16.6f;
        NVSDK_NGX_Resource_VK mask;
        if (maskImg != VK_NULL_HANDLE) {
            mask = wrapView(maskView, maskImg, VK_FORMAT_R8_UNORM,
                            in.renderWidth, in.renderHeight,
                            VK_IMAGE_ASPECT_COLOR_BIT, false);
            ev.pInBiasCurrentColorMask = &mask;
        }

        const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(in.cmd, feature_, params_, &ev);
        if (NVSDK_NGX_FAILED(r)) {
            // Log the full dimension state on the FIRST failure of a streak —
            // 0xBAD00005 (InvalidParameter) is almost always a size mismatch
            // between the eval inputs and the created feature, and this line is
            // the diagnosis. failing() trips after 3 consecutive failures; the
            // frame loop then falls back to FSR/TAA and recreates the feature.
            if (evalFails_ == 0) {
                std::fprintf(stderr,
                             "[threepp] DLSS: evaluate failed (0x%08x) — render %ux%u, "
                             "display %ux%u, feature %ux%u; will self-heal after 3 failures.\n",
                             static_cast<unsigned>(r),
                             in.renderWidth, in.renderHeight,
                             in.displayWidth, in.displayHeight,
                             displayW_, displayH_);
            }
            ++evalFails_;
        } else {
            evalFails_ = 0;
        }

        // Restore the borrowed inputs to their steady-state layouts. UNDEFINED
        // old layout: their contents aren't needed after the upscale.
        transition(in.cmd, in.colorImage, VK_IMAGE_LAYOUT_UNDEFINED,
                   in.colorLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(in.cmd, in.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
                   in.depthLayout, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (in.motionLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            transition(in.cmd, in.motionImage, VK_IMAGE_LAYOUT_UNDEFINED,
                       in.motionLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        // The output is left in GENERAL with the upscale's writes; the caller
        // barriers it before PostComposite reads it.
    }

    void DlssUpscaler::createReactivePipeline() {
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
        check(vkCreateSampler(d, &sci, nullptr, &idsSampler_), "vkCreateSampler(dlss.mask.ids)");

        // 0: gbuf IDs (usampler2D), 1: mask R8 (storage).
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2; dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &reactiveDsLayout_),
              "vkCreateDescriptorSetLayout(dlss.mask)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 12;// w,h,value
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &reactiveDsLayout_;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &reactivePipeLayout_),
              "vkCreatePipelineLayout(dlss.mask)");

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kFsrReactiveCompSpv); smci.pCode = kFsrReactiveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(dlss.mask)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = mod; stage.pName = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage; cpci.layout = reactivePipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &reactivePipe_),
              "vkCreateComputePipelines(dlss.mask)");
        vkDestroyShaderModule(d, mod, nullptr);

        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = framesInFlight_;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = framesInFlight_; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &reactivePool_),
              "vkCreateDescriptorPool(dlss.mask)");
        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, reactiveDsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = reactivePool_; ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(d, &ai, reactiveSets_.data()),
              "vkAllocateDescriptorSets(dlss.mask)");
    }

    void DlssUpscaler::createReactiveImages(uint32_t width, uint32_t height) {
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
                  "vmaCreateImage(dlss.mask)");
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = VK_FORMAT_R8_UNORM;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(d, &vci, nullptr, &img.view), "vkCreateImageView(dlss.mask)");
        }
    }

    void DlssUpscaler::destroyReactiveImages() {
        VkDevice d = ctx_.device();
        for (auto& img : reactive_) destroyImage2D(ctx_.allocator(), d, img);
    }

}// namespace threepp::vulkan
