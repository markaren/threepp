#include "threepp/renderers/vulkan/Denoiser.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/denoise.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/denoise_atrous.comp.spv.h"

namespace threepp::vulkan {

    Denoiser::Denoiser(VulkanContext& ctx,
                       VkDescriptorSetLayout sharedRtDsLayout,
                       VkCommandPool cmdPool)
        : ctx_(ctx), sharedDsLayout_(sharedRtDsLayout), cmdPool_(cmdPool) {
        createPipelines();
    }

    Denoiser::~Denoiser() {
        VkDevice d = ctx_.device();
        if (finalizePipeline_) vkDestroyPipeline(d, finalizePipeline_, nullptr);
        if (atrousPipeline_)   vkDestroyPipeline(d, atrousPipeline_, nullptr);
        if (pipelineLayout_)   vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        destroyImages();
    }

    void Denoiser::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : filteredImagesPP_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : momentsImagesPP_)  destroyImage2D(ctx_.allocator(), d, img);
    }

    Image2D Denoiser::createStorageImage2D(uint32_t w, uint32_t h,
                                           VkFormat format,
                                           const char* label) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = format;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci,
                             &out.image, &out.alloc, nullptr),
              label);

        transitionFreshImage(out.image);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vkCreateImageView(denoise)");
        return out;
    }

    void Denoiser::transitionFreshImage(VkImage img) {
        // One-shot UNDEFINED → GENERAL. The renderer's clearGbufImages pass
        // also touches these slots (it expects GENERAL on entry), so the
        // transition has to happen before clearGbufImages runs.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb),
              "alloc one-shot cb(denoise)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(denoise)");

        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(denoise)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE),
              "submit one-shot(denoise)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(denoise)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void Denoiser::createImages(uint32_t width, uint32_t height) {
        destroyImages();
        // filt[0] is the pass-0 / pass-2 destination, filt[1] is the pass-1
        // destination. After the final atrous pass filt[0] holds the filtered
        // radiance that denoise.comp (finalize) mixes with raw accumImage by
        // per-pixel FC.
        for (auto& img : filteredImagesPP_)
            img = createStorageImage2D(width, height,
                                       VK_FORMAT_R32G32B32A32_SFLOAT,
                                       "vmaCreateImage(denoise.filtered)");
        // Temporal moments ping-pong (R32F): each slot stores per-pixel
        // mean(luminance²) integrated alongside accumImage with the same FC
        // weight and reproject taps. denoise_atrous reads variance =
        // max(M2 − lum(mean)², 0) to drive σ_lum per pixel.
        for (auto& img : momentsImagesPP_)
            img = createStorageImage2D(width, height,
                                       VK_FORMAT_R32_SFLOAT,
                                       "vmaCreateImage(denoise.moments)");
    }

    void Denoiser::createPipelines() {
        // Push constants: 4 × u32 = 16 bytes (toneMapping, exposureBits,
        // denoiseEnabled, _pad for finalize; stride, readFromAccum, inputIdx,
        // outputIdx for atrous). COMPUTE-only, separate from rtPipelineLayout's
        // RGEN/CHIT/MISS push constant range.
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 16;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &sharedDsLayout_;// reuse the RT descriptor set layout
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(denoise)");

        // Finalize pipeline (denoise.comp): tonemap + sRGB + FC fade → outImage.
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDenoiseCompSpv);
        smci.pCode    = kDenoiseCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(denoise)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE,
                                       1, &cpci, nullptr, &finalizePipeline_),
              "vkCreateComputePipelines(denoise)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Atrous pipeline (denoise_atrous.comp) shares the same layout —
        // same descriptor set, same 16-byte COMPUTE push constant range
        // (interpretation differs).
        VkShaderModuleCreateInfo asmci{};
        asmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        asmci.codeSize = sizeof(kDenoiseAtrousCompSpv);
        asmci.pCode    = kDenoiseAtrousCompSpv;
        VkShaderModule amod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &asmci, nullptr, &amod),
              "vkCreateShaderModule(denoise_atrous)");

        VkPipelineShaderStageCreateInfo astage{};
        astage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        astage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        astage.module = amod;
        astage.pName  = "main";

        VkComputePipelineCreateInfo acpci{};
        acpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        acpci.stage  = astage;
        acpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE,
                                       1, &acpci, nullptr, &atrousPipeline_),
              "vkCreateComputePipelines(denoise_atrous)");
        vkDestroyShaderModule(ctx_.device(), amod, nullptr);
    }

    void Denoiser::recordDispatch(VkCommandBuffer cb,
                                  VkDescriptorSet frameDescriptorSet,
                                  VkExtent2D      extent,
                                  bool            denoiseEnabled,
                                  uint32_t        toneMapping,
                                  uint32_t        exposureBits) {
        const uint32_t gx = (extent.width  + 7u) / 8u;
        const uint32_t gy = (extent.height + 7u) / 8u;

        // Bind once — both pipelines share pipelineLayout_.
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1,
                                &frameDescriptorSet, 0, nullptr);

        // RT_SHADER write → COMPUTE_SHADER read+write (or COMPUTE → COMPUTE
        // between atrous passes / atrous → finalize).
        auto barrierMem = [&](VkPipelineStageFlags2 srcStage) {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = srcStage;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo bd{};
            bd.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            bd.memoryBarrierCount = 1;
            bd.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &bd);
        };

        barrierMem(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

        if (denoiseEnabled) {
            // 2 atrous passes: stride 1 → 2. Pass 0 sources from accumImage
            // and writes filt[1]; pass 1 reads filt[1] and writes filt[0].
            // Finalize then reads filt[0].
            //
            // Initial multi-pass land used 3 passes (strides 1/2/4) with a
            // 21×21 effective radius. On this fast-converging PT (60-200
            // FPS, 2 spp → settled in <1s) the third pass overblurred clean
            // input. 2 passes give a 9×9 radius which keeps detail without
            // leaving residual noise.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, atrousPipeline_);

            struct AtrousPc {
                uint32_t stride;
                uint32_t readFromAccum;
                uint32_t inputIdx;
                uint32_t outputIdx;
            };
            // Schedule lands the final filtered output in filt[0] so
            // finalize.comp's hard-coded filteredArray[0] read is correct.
            const AtrousPc passes[2] = {
                    {1u, 1u, 0u, 1u}, // accum → filt[1]
                    {2u, 0u, 1u, 0u}, // filt[1] → filt[0]
            };
            for (int i = 0; i < 2; ++i) {
                if (i > 0) barrierMem(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                vkCmdPushConstants(cb, pipelineLayout_,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(passes[i]), &passes[i]);
                vkCmdDispatch(cb, gx, gy, 1);
            }

            // Atrous → finalize barrier (filt[0] write → finalize read).
            barrierMem(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        }

        // Finalize: tonemap + sRGB → outImage. When `denoiseEnabled` is
        // false, this is the only compute dispatch in the block; the
        // initial RT → COMPUTE barrier above covers it.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, finalizePipeline_);
        const uint32_t finalizePc[4] = {
                toneMapping,
                exposureBits,
                denoiseEnabled ? 1u : 0u,
                0u,
        };
        vkCmdPushConstants(cb, pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(finalizePc), finalizePc);
        vkCmdDispatch(cb, gx, gy, 1);
    }

}// namespace threepp::vulkan
