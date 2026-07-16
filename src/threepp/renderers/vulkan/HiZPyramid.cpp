#include "threepp/renderers/vulkan/HiZPyramid.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/hiz_build.comp.spv.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace threepp::vulkan {

    namespace {
        struct HiZPc {
            int32_t dstW, dstH;
            int32_t srcW, srcH;
            uint32_t fromDepth;
            uint32_t msSamples;// >1 = mip 0 reduces the raw MS depth attachment
        };
        static_assert(sizeof(HiZPc) == 24, "hiz_build push-constant drift");

        uint32_t mipDim(uint32_t base, uint32_t level) {
            return std::max(base >> level, 1u);
        }
    }// namespace

    HiZPyramid::HiZPyramid(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        assert(framesInFlight_ <= 8);// cachedDepthViews_ bound
        // NEAREST + unclamped LOD: the occlusion cull texelFetches explicit mips.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(hiz)");
        createPipeline();
    }

    HiZPyramid::~HiZPyramid() {
        VkDevice d = ctx_.device();
        destroyImage();
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
    }

    void HiZPyramid::destroyImage() {
        VkDevice d = ctx_.device();
        for (auto v : mipViews_) vkDestroyImageView(d, v, nullptr);
        mipViews_.clear();
        if (fullView_) vkDestroyImageView(d, fullView_, nullptr);
        fullView_ = VK_NULL_HANDLE;
        if (image_) vmaDestroyImage(ctx_.allocator(), image_, alloc_);
        image_ = VK_NULL_HANDLE;
        alloc_ = VK_NULL_HANDLE;
    }

    void HiZPyramid::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[4]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding = i;
            b[i].descriptorType = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// 1× depth (mip-0 pass)
        set(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);         // src mip (reduce pass)
        set(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);         // dst mip
        set(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// MS depth (or 1×1 dummy)

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 4;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(hiz)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(HiZPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(hiz)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kHizBuildCompSpv);
        smci.pCode    = kHizBuildCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(hiz_build)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(hiz_build)");
        vkDestroyShaderModule(d, mod, nullptr);
    }

    void HiZPyramid::resize(VkExtent2D extent, const VkImageView* depthViews,
                            const VkImageView* msDepthViews, uint32_t msSamples) {
        bool sameDepth = msSamples == msSamples_;
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            sameDepth = sameDepth && cachedDepthViews_[f] == depthViews[f] &&
                        cachedMsDepthViews_[f] == msDepthViews[f];
        if (image_ && extent.width == extent_.width && extent.height == extent_.height && sameDepth)
            return;

        VkDevice d = ctx_.device();
        destroyImage();
        if (descPool_) vkDestroyDescriptorPool(d, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        sets_.clear();

        extent_ = extent;
        mipCount_ = 1;
        while (mipDim(extent.width, mipCount_) > 1 || mipDim(extent.height, mipCount_) > 1)
            ++mipCount_;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            cachedDepthViews_[f]   = depthViews[f];
            cachedMsDepthViews_[f] = msDepthViews[f];
        }
        msSamples_ = msSamples;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_R32_SFLOAT;
        ici.extent        = {extent.width, extent.height, 1};
        ici.mipLevels     = mipCount_;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &image_, &alloc_, nullptr),
              "vmaCreateImage(hiz)");
        ctx_.setObjectName(image_, "hizPyramid");
        needsLayoutInit_ = true;

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = image_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = VK_FORMAT_R32_SFLOAT;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = mipCount_;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(d, &vci, nullptr, &fullView_), "vkCreateImageView(hiz full)");
        mipViews_.resize(mipCount_);
        for (uint32_t m = 0; m < mipCount_; ++m) {
            vci.subresourceRange.baseMipLevel = m;
            vci.subresourceRange.levelCount   = 1;
            check(vkCreateImageView(d, &vci, nullptr, &mipViews_[m]), "vkCreateImageView(hiz mip)");
        }

        // Descriptor sets: one per (frame, mip). Mip 0 samples the frame's
        // depth; reduces load mip m-1 → store mip m (mip 0 binds itself as
        // the unused src — never loaded on the fromDepth path).
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * mipCount_ * 2;// 1× + MS depth
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_ * mipCount_ * 2;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_ * mipCount_;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_), "vkCreateDescriptorPool(hiz)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_ * mipCount_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        ai.pSetLayouts        = layouts.data();
        sets_.resize(layouts.size());
        check(vkAllocateDescriptorSets(d, &ai, sets_.data()), "vkAllocateDescriptorSets(hiz)");

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            for (uint32_t m = 0; m < mipCount_; ++m) {
                VkDescriptorImageInfo depthInfo{};
                depthInfo.sampler     = sampler_;
                depthInfo.imageView   = depthViews[f];
                depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                VkDescriptorImageInfo srcInfo{};
                srcInfo.imageView   = mipViews_[m == 0 ? 0 : m - 1];
                srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo dstInfo{};
                dstInfo.imageView   = mipViews_[m];
                dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo msInfo{};
                msInfo.sampler     = sampler_;
                msInfo.imageView   = msDepthViews[f];
                msInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet w[4]{};
                auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                                const VkDescriptorImageInfo* img) {
                    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[n].dstSet = sets_[f * mipCount_ + m];
                    w[n].dstBinding = bind;
                    w[n].descriptorCount = 1;
                    w[n].descriptorType = t;
                    w[n].pImageInfo = img;
                };
                setw(0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo);
                setw(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &srcInfo);
                setw(2, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &dstInfo);
                setw(3, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &msInfo);
                vkUpdateDescriptorSets(d, 4, w, 0, nullptr);
            }
        }
    }

    void HiZPyramid::record(VkCommandBuffer cb, uint32_t frame) {
        if (!image_) return;

        // Leading barrier: fresh image → GENERAL; otherwise fence last
        // frame's sampled reads against this frame's writes (WAR).
        {
            VkImageMemoryBarrier2 ib{};
            ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            ib.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            ib.srcAccessMask = needsLayoutInit_ ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            ib.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            ib.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            ib.oldLayout     = needsLayoutInit_ ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_GENERAL;
            ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = image_;
            ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ib.subresourceRange.levelCount = mipCount_;
            ib.subresourceRange.layerCount = 1;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &ib;
            vkCmdPipelineBarrier2(cb, &dep);
            needsLayoutInit_ = false;
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);

        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo mdep{};
        mdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        mdep.memoryBarrierCount = 1;
        mdep.pMemoryBarriers = &mb;

        for (uint32_t m = 0; m < mipCount_; ++m) {
            if (m > 0) vkCmdPipelineBarrier2(cb, &mdep);// mip m-1 writes → reads
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                    0, 1, &sets_[frame * mipCount_ + m], 0, nullptr);
            HiZPc pc{};
            pc.dstW = static_cast<int32_t>(mipDim(extent_.width, m));
            pc.dstH = static_cast<int32_t>(mipDim(extent_.height, m));
            pc.srcW = static_cast<int32_t>(m == 0 ? extent_.width  : mipDim(extent_.width, m - 1));
            pc.srcH = static_cast<int32_t>(m == 0 ? extent_.height : mipDim(extent_.height, m - 1));
            pc.fromDepth = m == 0 ? 1u : 0u;
            pc.msSamples = msSamples_;
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (static_cast<uint32_t>(pc.dstW) + 7u) / 8u,
                          (static_cast<uint32_t>(pc.dstH) + 7u) / 8u, 1);
        }

        // Trailing: pyramid writes → the shade dispatch's sampled reads.
        VkMemoryBarrier2 tb{};
        tb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        tb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        tb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        tb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        tb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo tdep{};
        tdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        tdep.memoryBarrierCount = 1;
        tdep.pMemoryBarriers = &tb;
        vkCmdPipelineBarrier2(cb, &tdep);
    }

}// namespace threepp::vulkan
