#include "threepp/renderers/vulkan/BloomPyramid.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/bloom_up.comp.spv.h"

#include <algorithm>

namespace threepp::vulkan {

    namespace {

        // bloom_down.comp / bloom_up.comp's push block (the upsample reads the
        // four extents only).
        struct BloomPc {
            uint32_t srcW, srcH, dstW, dstH;
            float    threshold, clampMax;
            uint32_t firstLevel;
        };
        static_assert(sizeof(BloomPc) == 28, "bloom_down.comp's push block is 28 bytes");

    }// namespace

    BloomPyramid::BloomPyramid(VulkanContext& ctx, uint32_t framesInFlight, uint32_t maxLevels, const char* label)
        : ctx_(ctx), framesInFlight_(framesInFlight), maxLevels_(maxLevels), label_(label) {
        const VkDevice d = ctx_.device();
        const auto what = [&](const char* call) { return std::string(call) + "(" + label_ + ")"; };

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), what("vkCreateSampler").c_str());

        // Down and up share one layout: combined sampler @0, storage image @1.
        VkDescriptorSetLayoutBinding bnd[2]{};
        bnd[0].binding         = 0;
        bnd[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[0].descriptorCount = 1;
        bnd[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bnd[1].binding         = 1;
        bnd[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[1].descriptorCount = 1;
        bnd[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings    = bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &setLayout_),
              what("vkCreateDescriptorSetLayout").c_str());

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(BloomPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &setLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_), what("vkCreatePipelineLayout").c_str());

        downPipe_ = createComputePipeline(d, ctx_.pipelineCache(), pipeLayout_, kBloomDownCompSpv,
                                          sizeof(kBloomDownCompSpv), (label_ + "_down").c_str());
        upPipe_   = createComputePipeline(d, ctx_.pipelineCache(), pipeLayout_, kBloomUpCompSpv,
                                          sizeof(kBloomUpCompSpv), (label_ + "_up").c_str());

        const uint32_t perChain = framesInFlight_ * maxLevels_;
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = 2 * perChain;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = 2 * perChain;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 2 * perChain;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_), what("vkCreateDescriptorPool").c_str());

        const std::vector<VkDescriptorSetLayout> layouts(perChain, setLayout_);
        for (auto* sets : {&downSets_, &upSets_}) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = perChain;
            ai.pSetLayouts        = layouts.data();
            sets->resize(perChain);
            check(vkAllocateDescriptorSets(d, &ai, sets->data()), what("vkAllocateDescriptorSets").c_str());
        }
    }

    BloomPyramid::~BloomPyramid() {
        const VkDevice d = ctx_.device();
        if (downPipe_)   vkDestroyPipeline(d, downPipe_, nullptr);
        if (upPipe_)     vkDestroyPipeline(d, upPipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (setLayout_)  vkDestroyDescriptorSetLayout(d, setLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
    }

    uint32_t BloomPyramid::levelCount(uint32_t w, uint32_t h, uint32_t maxLevels) {
        uint32_t levels = 0;
        while (levels < maxLevels && std::min(w >> (levels + 1u), h >> (levels + 1u)) >= 8u) ++levels;
        return levels == 0 ? 1u : levels;// a degenerate tiny extent keeps the half-extent level
    }

    VkExtent2D BloomPyramid::levelExtent(uint32_t w, uint32_t h, uint32_t l) {
        return {std::max(w >> (l + 1u), 1u), std::max(h >> (l + 1u), 1u)};
    }

    void BloomPyramid::writeSets(uint32_t frame, VkImageView input, const Image2D* level0, uint32_t levels) {
        const auto sampled = [&](VkImageView v) {
            VkDescriptorImageInfo i{};
            i.sampler     = sampler_;
            i.imageView   = v;
            i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            return i;
        };
        const auto storage = [&](VkImageView v) {
            VkDescriptorImageInfo i{};
            i.imageView   = v;
            i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            return i;
        };
        const auto writePair = [&](VkDescriptorSet ds, const VkDescriptorImageInfo* src,
                                   const VkDescriptorImageInfo* dst) {
            VkWriteDescriptorSet w[2]{};
            for (int n = 0; n < 2; ++n) {
                w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet          = ds;
                w[n].dstBinding      = static_cast<uint32_t>(n);
                w[n].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo     = src;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = dst;
            vulkan::updateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
        };

        // Down chain: input -> level 0, then level l-1 -> level l.
        // Up chain: level l+1 (tent-sampled) accumulates into level l.
        for (uint32_t l = 0; l < levels; ++l) {
            const VkDescriptorImageInfo dIn  = sampled(l == 0 ? input : level0[l - 1].view);
            const VkDescriptorImageInfo dOut = storage(level0[l].view);
            writePair(downSets_[frame * maxLevels_ + l], &dIn, &dOut);
            if (l + 1 < levels) {
                const VkDescriptorImageInfo uIn  = sampled(level0[l + 1].view);
                const VkDescriptorImageInfo uOut = storage(level0[l].view);
                writePair(upSets_[frame * maxLevels_ + l], &uIn, &uOut);
            }
        }
    }

    void BloomPyramid::record(VkCommandBuffer cb, uint32_t frame, const Chain& chain) const {
        if (chain.levels == 0) return;

        // Between levels: bloom_down and bloom_up read the level written just
        // before through a sampler2D, which synchronization2 counts as
        // SHADER_SAMPLED_READ, not SHADER_STORAGE_READ, so both are in the
        // destination scope (bloom_up also reads and writes its own level as
        // a storage image).
        const auto barrier = [&] {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        };
        const auto level = [&](uint32_t l) { return levelExtent(chain.base.width, chain.base.height, l); };

        // Progressive downsample: input -> level 0 (the first level's filter,
        // per chain.firstLevel and clampMax) -> level 1 -> ... (plain 13-tap).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, downPipe_);
        for (uint32_t l = 0; l < chain.levels; ++l) {
            const VkExtent2D src = l == 0 ? chain.input : level(l - 1);
            const VkExtent2D dst = level(l);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                    &downSets_[frame * maxLevels_ + l], 0, nullptr);
            const BloomPc pc{src.width, src.height, dst.width, dst.height, chain.threshold,
                             l == 0 ? chain.clampMax : 0.f, l == 0 ? chain.firstLevel : 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (dst.width + 7u) / 8u, (dst.height + 7u) / 8u, 1);
            // Level l feeds level l+1; the last level feeds the upsample
            // (when there is one).
            if (chain.levels > 1) barrier();
        }

        // Progressive upsample walk-back: tent-filter each coarser level and
        // ADD it into the next finer one, accumulating the whole chain into
        // level 0.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, upPipe_);
        for (uint32_t l = chain.levels - 1; l > 0; --l) {
            const uint32_t   d   = l - 1;
            const VkExtent2D src = level(l);
            const VkExtent2D dst = level(d);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                    &upSets_[frame * maxLevels_ + d], 0, nullptr);
            const BloomPc pc{src.width, src.height, dst.width, dst.height, 0.f, 0.f, 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (dst.width + 7u) / 8u, (dst.height + 7u) / 8u, 1);
            // Level d feeds the next upsample step; level 0 feeds the caller's
            // consumer, which orders its own read.
            if (d > 0) barrier();
        }
    }

}// namespace threepp::vulkan
