#include "threepp/renderers/vulkan/TaaResolve.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/taa_resolve.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/rcas.comp.spv.h"

#include <array>
#include <cstring>

namespace threepp::vulkan {

    TaaResolve::TaaResolve(VulkanContext& ctx,
                           VkCommandPool cmdPool,
                           uint32_t imageCount,
                           uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool),
          imageCount_(imageCount), framesInFlight_(framesInFlight) {
        inputImagesPP_.resize(framesInFlight_);
        createPipeline();
        createDescriptorPool();
    }

    TaaResolve::~TaaResolve() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (rcasPipe_)       vkDestroyPipeline(d, rcasPipe_, nullptr);
        if (rcasPipeLayout_) vkDestroyPipelineLayout(d, rcasPipeLayout_, nullptr);
        if (rcasDsLayout_)   vkDestroyDescriptorSetLayout(d, rcasDsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        destroyImages();
    }

    void TaaResolve::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : inputImagesPP_)   destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : historyImagesPP_) destroyImage2D(ctx_.allocator(), d, img);
        historyValid_ = false;
    }

    Image2D TaaResolve::createStorageSampledImage(uint32_t w, uint32_t h,
                                                  VkFormat format,
                                                  const char* label) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = format;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
              label);

        transitionFreshImage(out.image);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vkCreateImageView(taa)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view,  label);
        return out;
    }

    void TaaResolve::transitionFreshImage(VkImage img) {
        // One-shot UNDEFINED → GENERAL so the first frame's storage + sampled
        // accesses work without further layout management.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb),
              "alloc one-shot cb(taa)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(taa)");

        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(taa)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE),
              "submit one-shot(taa)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(taa)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void TaaResolve::createImages(uint32_t inWidth, uint32_t inHeight,
                                  uint32_t outWidth, uint32_t outHeight) {
        destroyImages();
        // Input: BGRA8_UNORM at the render extent — matches denoise.comp's
        // rgba8 output and the swapchain channel order.
        for (auto& img : inputImagesPP_)
            img = createStorageSampledImage(inWidth, inHeight,
                                            VK_FORMAT_B8G8R8A8_UNORM,
                                            "vmaCreateImage(taa.input)");
        // History: RGBA16F at the output extent — the running mix() stays
        // sub-quantum precise and the reconstructed full-res image
        // accumulates here when the input is lower-resolution.
        for (auto& img : historyImagesPP_)
            img = createStorageSampledImage(outWidth, outHeight,
                                            VK_FORMAT_R16G16B16A16_SFLOAT,
                                            "vmaCreateImage(taa.history)");
    }

    void TaaResolve::createPipeline() {
        if (sampler_ == VK_NULL_HANDLE) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            // LINEAR filter — required by the Catmull-Rom 5-tap history
            // reconstruction in taa_resolve.comp (each tap fuses 4 texels
            // via the bilinear sampler). A naive single bilinear sample
            // would compound a half-pixel blur every frame on translating
            // close objects — the long-standing "everything smears" bug.
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod       = 0.f;
            check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_),
                  "vkCreateSampler(taa)");
        }
        // Descriptor set layout — 7 bindings:
        //   0..2: combined image samplers — taaInput, historyRead, gbufMotion
        //   3..4: storage images          — swapOut, historyWrite
        //   5..6: combined image samplers — gbufIds (curr + prev)
        VkDescriptorSetLayoutBinding bindings[7]{};
        for (int i = 0; i < 3; ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].binding         = 4;
        bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].binding         = 5;
        bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[6].binding         = 6;
        bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 7;
        dlci.pBindings    = bindings;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(taa)");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 24;// blendAlpha + out w/h + in w/h + pad

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(taa)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kTaaResolveCompSpv);
        smci.pCode    = kTaaResolveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(taa)");

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
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(taa)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // ── RCAS sharpen pipeline: sampled resolved @0, storage swapchain @1;
        //    16-byte PC (width, height, amount, pad). ──────────────────────
        {
            VkDescriptorSetLayoutBinding rb[2]{};
            rb[0].binding = 0;
            rb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            rb[0].descriptorCount = 1;
            rb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            rb[1].binding = 1;
            rb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            rb[1].descriptorCount = 1;
            rb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo rdlci{};
            rdlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            rdlci.bindingCount = 2;
            rdlci.pBindings    = rb;
            check(vkCreateDescriptorSetLayout(ctx_.device(), &rdlci, nullptr, &rcasDsLayout_),
                  "vkCreateDescriptorSetLayout(rcas)");

            VkPushConstantRange rpc{};
            rpc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            rpc.offset     = 0;
            rpc.size       = 16;
            VkPipelineLayoutCreateInfo rplci{};
            rplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            rplci.setLayoutCount         = 1;
            rplci.pSetLayouts            = &rcasDsLayout_;
            rplci.pushConstantRangeCount = 1;
            rplci.pPushConstantRanges    = &rpc;
            check(vkCreatePipelineLayout(ctx_.device(), &rplci, nullptr, &rcasPipeLayout_),
                  "vkCreatePipelineLayout(rcas)");

            VkShaderModuleCreateInfo rsmci{};
            rsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            rsmci.codeSize = sizeof(kRcasCompSpv);
            rsmci.pCode    = kRcasCompSpv;
            VkShaderModule rmod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &rsmci, nullptr, &rmod),
                  "vkCreateShaderModule(rcas)");
            VkPipelineShaderStageCreateInfo rstage{};
            rstage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            rstage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            rstage.module = rmod;
            rstage.pName  = "main";
            VkComputePipelineCreateInfo rcpci{};
            rcpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            rcpci.stage  = rstage;
            rcpci.layout = rcasPipeLayout_;
            check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &rcpci,
                                           nullptr, &rcasPipe_),
                  "vkCreateComputePipelines(rcas)");
            vkDestroyShaderModule(ctx_.device(), rmod, nullptr);
        }
    }

    void TaaResolve::createDescriptorPool() {
        const uint32_t totalSets = imageCount_ * framesInFlight_;
        VkDescriptorPoolSize sizes[2]{};
        // Main resolve set: 5 sampled + 2 storage. RCAS set: 1 sampled + 1
        // storage. Both families × totalSets, sharing this pool.
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = totalSets * (5 + 1);
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = totalSets * (2 + 1);

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = totalSets * 2;// main + rcas
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(taa)");

        std::vector<VkDescriptorSetLayout> layouts(totalSets, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = totalSets;
        ai.pSetLayouts        = layouts.data();
        descSets_.resize(totalSets);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, descSets_.data()),
              "vkAllocateDescriptorSets(taa)");

        std::vector<VkDescriptorSetLayout> rlayouts(totalSets, rcasDsLayout_);
        VkDescriptorSetAllocateInfo rai{};
        rai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        rai.descriptorPool     = descPool_;
        rai.descriptorSetCount = totalSets;
        rai.pSetLayouts        = rlayouts.data();
        rcasSets_.resize(totalSets);
        check(vkAllocateDescriptorSets(ctx_.device(), &rai, rcasSets_.data()),
              "vkAllocateDescriptorSets(rcas)");
    }

    void TaaResolve::rewriteDescriptors(const DescriptorWriteInputs& inputs) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            for (uint32_t i = 0; i < imageCount_; ++i) {
                const uint32_t idx = f * imageCount_ + i;
                const uint32_t readSlot  = 1u - (f & 1u);
                const uint32_t writeSlot = (f & 1u);

                VkDescriptorImageInfo inputI{};
                inputI.sampler     = sampler_;
                inputI.imageView   = inputImagesPP_[f].view;
                inputI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo histReadI{};
                histReadI.sampler     = sampler_;
                histReadI.imageView   = historyImagesPP_[readSlot].view;
                histReadI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo motionI{};
                motionI.sampler     = inputs.gbufSampler;
                motionI.imageView   = inputs.gbufMotionPerFrame[f];
                motionI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkDescriptorImageInfo swapI{};
                swapI.imageView   = inputs.swapchainViews[i];
                swapI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo histWriteI{};
                histWriteI.imageView   = historyImagesPP_[writeSlot].view;
                histWriteI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                // Curr / prev gbuffer IDs for mesh-ID rejection + skinned
                // detection. Prev gbuffer is the OTHER frame-in-flight slot.
                const uint32_t prevFrame = (f + (framesInFlight_ - 1u)) % framesInFlight_;
                VkDescriptorImageInfo idsCurrI{};
                idsCurrI.sampler     = inputs.gbufSampler;
                idsCurrI.imageView   = inputs.gbufIdsPerFrame[f];
                idsCurrI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkDescriptorImageInfo idsPrevI{};
                idsPrevI.sampler     = inputs.gbufSampler;
                idsPrevI.imageView   = inputs.gbufIdsPerFrame[prevFrame];
                idsPrevI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet w[7]{};
                for (int b = 0; b < 7; ++b) {
                    w[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[b].dstSet          = descSets_[idx];
                    w[b].dstBinding      = uint32_t(b);
                    w[b].descriptorCount = 1;
                }
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[0].pImageInfo = &inputI;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[1].pImageInfo = &histReadI;
                w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[2].pImageInfo = &motionI;
                w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[3].pImageInfo = &swapI;
                w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[4].pImageInfo = &histWriteI;
                w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[5].pImageInfo = &idsCurrI;
                w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[6].pImageInfo = &idsPrevI;
                vkUpdateDescriptorSets(ctx_.device(), 7, w, 0, nullptr);

                // RCAS set: this frame's resolved output lives in the history
                // WRITE slot (writeSlot) → sample it, sharpen, write swapchain.
                VkDescriptorImageInfo rcasIn{};
                rcasIn.sampler     = sampler_;
                rcasIn.imageView   = historyImagesPP_[writeSlot].view;
                rcasIn.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo rcasOut{};
                rcasOut.imageView   = inputs.swapchainViews[i];
                rcasOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet rw[2]{};
                rw[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                rw[0].dstSet          = rcasSets_[idx];
                rw[0].dstBinding      = 0;
                rw[0].descriptorCount = 1;
                rw[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                rw[0].pImageInfo      = &rcasIn;
                rw[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                rw[1].dstSet          = rcasSets_[idx];
                rw[1].dstBinding      = 1;
                rw[1].descriptorCount = 1;
                rw[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                rw[1].pImageInfo      = &rcasOut;
                vkUpdateDescriptorSets(ctx_.device(), 2, rw, 0, nullptr);
            }
        }
    }

    void TaaResolve::recordResolve(VkCommandBuffer cb,
                                   uint32_t frame,
                                   uint32_t imageIndex,
                                   uint32_t inWidth,
                                   uint32_t inHeight,
                                   uint32_t outWidth,
                                   uint32_t outHeight,
                                   float blendAlpha,
                                   bool sharpen,
                                   float sharpenAmount) {
        // Barrier: taaInput write → read; both history slots covered (RAW
        // hazard on the read slot, WAW on the write slot we're about to
        // overwrite this frame).
        std::array<VkImageMemoryBarrier2, 3> pre{};
        for (auto& b : pre) {
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                              VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
        }
        pre[0].image = inputImagesPP_[frame].image;
        pre[1].image = historyImagesPP_[0].image;
        pre[2].image = historyImagesPP_[1].image;
        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount  = static_cast<uint32_t>(pre.size());
        dep.pImageMemoryBarriers     = pre.data();
        vkCmdPipelineBarrier2(cb, &dep);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const uint32_t descIdx = frame * imageCount_ + imageIndex;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1,
                                &descSets_[descIdx], 0, nullptr);
        // First-frame history is undefined → force alpha=1 so the resolved
        // output is purely the current frame and garbage doesn't bleed into
        // a permanent history. Subsequent frames use the caller's alpha.
        const float alpha = historyValid_ ? blendAlpha : 1.0f;
        uint32_t alphaBits;
        std::memcpy(&alphaBits, &alpha, sizeof(alphaBits));
        // Layout: blendAlpha, output w/h (history + dispatch + writes),
        // input w/h (the render extent the samples were traced at).
        const uint32_t pc[6] = {alphaBits, outWidth, outHeight,
                                inWidth, inHeight, sharpen ? 0u : 1u};
        vkCmdPushConstants(cb, pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), pc);
        // Dispatch covers the output extent — one thread per full-res pixel.
        const uint32_t gx = (outWidth  + 7u) / 8u;
        const uint32_t gy = (outHeight + 7u) / 8u;
        vkCmdDispatch(cb, gx, gy, 1);

        if (sharpen) {
            // The resolve wrote the resolved frame into the history slot and
            // skipped the swapchain. Make it visible, then RCAS-sharpen it
            // into the swapchain.
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rcasPipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rcasPipeLayout_, 0, 1, &rcasSets_[descIdx], 0, nullptr);
            uint32_t amountBits;
            std::memcpy(&amountBits, &sharpenAmount, sizeof(amountBits));
            const uint32_t rpc[4] = {outWidth, outHeight, amountBits, 0u};
            vkCmdPushConstants(cb, rcasPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(rpc), rpc);
            vkCmdDispatch(cb, gx, gy, 1);
        }

        historyValid_ = true;
    }

}// namespace threepp::vulkan
