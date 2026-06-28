#include "threepp/renderers/vulkan/AutoExposure.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/lum_histogram.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace threepp::vulkan {

    AutoExposure::AutoExposure(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        createPipeline();

        // Per-frame host-visible SSBOs (128 uint = 512 bytes).
        // VMA_MEMORY_USAGE_AUTO + HOST_ACCESS flags → GPU-writeable + CPU-readable
        // without an explicit staging copy; HOST_COHERENT ensures no cache flush.
        constexpr VkDeviceSize sz = kBins * sizeof(uint32_t);
        histBufs_.resize(framesInFlight_);
        for (auto& h : histBufs_) {
            h.buf = createBuffer(
                    ctx_.allocator(), ctx_.device(),
                    sz,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    // HOST_ACCESS_RANDOM forces HOST_VISIBLE allocation so pMappedData
                    // is always non-null. Do NOT add ALLOW_TRANSFER_INSTEAD_BIT — that
                    // lets VMA pick DEVICE_LOCAL memory and leaves the pointer null.
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), h.buf.alloc, &info);
            h.ptr = static_cast<uint32_t*>(info.pMappedData);
            if (!h.ptr)
                throw std::runtime_error("[AutoExposure] histogram buffer is not mapped");
            std::memset(h.ptr, 0, sz);
        }
    }

    AutoExposure::~AutoExposure() {
        VkDevice d = ctx_.device();
        if (descPool_)    vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipe_)        vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_)  vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)    vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (sampler_)     vkDestroySampler(d, sampler_, nullptr);
        for (auto& h : histBufs_) destroyBuffer(ctx_.allocator(), h.buf);
    }

    void AutoExposure::createPipeline() {
        VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.minFilter    = sci.magFilter = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(autoexp)");

        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding            = 0;
        bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount    = 1;
        bindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].pImmutableSamplers = &sampler_;
        bindings[1].binding            = 1;
        bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount    = 1;
        bindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 2;
        dslci.pBindings    = bindings;
        check(vkCreateDescriptorSetLayout(d, &dslci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(autoexp)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = 2 * sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(autoexp)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kLumHistogramCompSpv);
        smci.pCode    = kLumHistogramCompSpv;
        VkShaderModule sm = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &sm),
              "vkCreateShaderModule(lum_histogram)");

        VkComputePipelineCreateInfo cpci{};
        cpci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module       = sm;
        cpci.stage.pName        = "main";
        cpci.layout             = pipeLayout_;
        check(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(autoexp)");
        vkDestroyShaderModule(d, sm, nullptr);
    }

    void AutoExposure::rewriteDescriptors(const VkImageView* sceneHdrViews) {
        VkDevice d = ctx_.device();

        if (descPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(d, descPool_, nullptr);
            descPool_ = VK_NULL_HANDLE;
            descSets_.clear();
        }

        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = framesInFlight_;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(autoexp)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = framesInFlight_;
        dsai.pSetLayouts        = layouts.data();
        descSets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(d, &dsai, descSets_.data()),
              "vkAllocateDescriptorSets(autoexp)");

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorImageInfo imgInfo{};
            imgInfo.imageView   = sceneHdrViews[f];
            imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = histBufs_[f].buf.handle;
            bufInfo.offset = 0;
            bufInfo.range  = kBins * sizeof(uint32_t);

            VkWriteDescriptorSet w[2]{};
            w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet          = descSets_[f];
            w[0].dstBinding      = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo      = &imgInfo;
            w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet          = descSets_[f];
            w[1].dstBinding      = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[1].pBufferInfo     = &bufInfo;
            vkUpdateDescriptorSets(d, 2, w, 0, nullptr);
        }
    }

    void AutoExposure::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                      uint32_t width, uint32_t height) {
        // Clear this frame's bins on the GPU (avoids a CPU→GPU sync point).
        vkCmdFillBuffer(cb, histBufs_[frame].buf.handle, 0,
                        kBins * sizeof(uint32_t), 0u);

        // Transfer write → SSBO read/write in compute.
        VkMemoryBarrier2 fillBar{};
        fillBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fillBar.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        fillBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fillBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fillBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo fillDep{};
        fillDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        fillDep.memoryBarrierCount = 1;
        fillDep.pMemoryBarriers    = &fillBar;
        vkCmdPipelineBarrier2(cb, &fillDep);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &descSets_[frame], 0, nullptr);
        const uint32_t pc[2] = {width, height};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), pc);
        vkCmdDispatch(cb, (width + 15u) / 16u, (height + 15u) / 16u, 1u);
    }

    void AutoExposure::tick(uint32_t currentFrame, float dt) {
        // The previous frame's GPU slot is retired (framesInFlight fence logic).
        const uint32_t prev = (currentFrame + framesInFlight_ - 1) % framesInFlight_;
        // Invalidate CPU cache (required for non-HOST_COHERENT; harmless otherwise).
        vmaInvalidateAllocation(ctx_.allocator(), histBufs_[prev].buf.alloc,
                                0, kBins * sizeof(uint32_t));

        const uint32_t* bins = histBufs_[prev].ptr;
        if (!bins) return;

        uint64_t total = 0;
        for (uint32_t i = 0; i < kBins; ++i) total += bins[i];
        if (total == 0) return;

        // Weighted-percentile: average EV in [lowPercent, highPercent] of pixels.
        const uint64_t lo = static_cast<uint64_t>(static_cast<double>(total) * lowPercent);
        const uint64_t hi = static_cast<uint64_t>(static_cast<double>(total) * highPercent);

        double   sumEV   = 0.0;
        uint64_t count   = 0;
        uint64_t cumul   = 0;
        constexpr float evRange = kEvMax - kEvMin;
        for (uint32_t i = 0; i < kBins; ++i) {
            const uint64_t b    = bins[i];
            const uint64_t bEnd = cumul + b;
            if (b > 0) {
                const uint64_t oLo = std::max(cumul, lo);
                const uint64_t oHi = std::min(bEnd, hi);
                if (oLo < oHi) {
                    const float ev = kEvMin + (static_cast<float>(i) + 0.5f)
                                     / static_cast<float>(kBins) * evRange;
                    sumEV += ev * static_cast<double>(oHi - oLo);
                    count += (oHi - oLo);
                }
            }
            cumul = bEnd;
        }
        if (count == 0) return;

        const float meanEV = static_cast<float>(sumEV / static_cast<double>(count));

        // Target exposure: map scene mean to 18% gray (EV ≈ -2.47 = log2(0.18)).
        const float keyEV      = std::log2f(0.18f);
        float       targetEV   = std::clamp(keyEV - meanEV, minEV, maxEV);

        // Asymmetric EMA: eye constricts faster toward bright than it dilates toward dark.
        const float speed  = (targetEV < currentEV_) ? adaptSpeed * 0.5f : adaptSpeed;
        const float alpha  = std::min(1.0f, dt * speed);
        currentEV_         = currentEV_ + (targetEV - currentEV_) * alpha;
        exposure_          = std::exp2f(currentEV_);
    }

}// namespace threepp::vulkan
