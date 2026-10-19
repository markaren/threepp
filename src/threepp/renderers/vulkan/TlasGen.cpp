#include "threepp/renderers/vulkan/TlasGen.hpp"

#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/SpirvReflect.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/tlas_gen.comp.spv.h"

#include <algorithm>

namespace threepp::vulkan {

    namespace {
        constexpr uint32_t kBindings = 3;
        constexpr uint32_t kMinSpans = 64;
    }// namespace

    TlasGen::TlasGen(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        spanBufs_.resize(framesInFlight_);
        spanPtrs_.resize(framesInFlight_, nullptr);
        createPipeline();
    }

    TlasGen::~TlasGen() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        for (auto& b : spanBufs_) destroyBuffer(ctx_.allocator(), b);
    }

    void TlasGen::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[kBindings]{};
        for (uint32_t i = 0; i < kBindings; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = kBindings;
        dlci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_), "vkCreateDescriptorSetLayout(tlas_gen)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(Pc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_), "vkCreatePipelineLayout(tlas_gen)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kTlasGenCompSpv);
        smci.pCode    = kTlasGenCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(tlas_gen)");
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
              "vkCreateComputePipelines(tlas_gen)");
        vkDestroyShaderModule(d, mod, nullptr);

        VkDescriptorPoolSize size{};
        size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = framesInFlight_ * kBindings;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &size;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_), "vkCreateDescriptorPool(tlas_gen)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        dsai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(d, &dsai, sets_.data()), "vkAllocateDescriptorSets(tlas_gen)");
    }

    void TlasGen::prepareFrame(uint32_t frame, uint32_t spanCount, VkBuffer world, VkBuffer instances) {
        // Per-frame span pool: host-mapped, rewritten whole each frame it is
        // used (O(spans)), so a plain destroy on growth is safe - this slot's
        // fence has signaled.
        const VkDeviceSize spanBytes = VkDeviceSize(std::max(spanCount, 1u)) * sizeof(SpanDesc);
        if (spanBufs_[frame].size < spanBytes) {
            destroyBuffer(ctx_.allocator(), spanBufs_[frame]);
            const VkDeviceSize cap = std::max<VkDeviceSize>(spanBytes * 2, kMinSpans * sizeof(SpanDesc));
            spanBufs_[frame] = createBuffer(ctx_.allocator(), ctx_.device(), cap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VMA_MEMORY_USAGE_AUTO,
                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), spanBufs_[frame].alloc, &info);
            spanPtrs_[frame] = static_cast<SpanDesc*>(info.pMappedData);
        }
        // Unconditionally, as InstanceExpand does: a reallocated buffer can
        // come back with the handle value it had.
        const VkDescriptorBufferInfo bi[kBindings]{{spanBufs_[frame].handle, 0, VK_WHOLE_SIZE},
                                                   {world, 0, VK_WHOLE_SIZE},
                                                   {instances, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[kBindings]{};
        for (uint32_t i = 0; i < kBindings; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = sets_[frame];
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &bi[i];
        }
        vulkan::updateDescriptorSets(ctx_.device(), kBindings, w, 0, nullptr);
    }

    void TlasGen::flushSpans(uint32_t frame, uint32_t spanCount) {
        if (spanCount == 0u || spanBufs_[frame].alloc == VK_NULL_HANDLE) return;
        flushHostWrites(ctx_.allocator(), spanBufs_[frame].alloc, 0, VkDeviceSize(spanCount) * sizeof(SpanDesc));
    }

    void TlasGen::record(VkCommandBuffer cb, uint32_t frame, const Pc& pc) {
        if (pc.spanCount == 0u || pc.totalWork == 0u || pipe_ == VK_NULL_HANDLE) return;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (pc.totalWork + 63u) / 64u, 1, 1);
    }

    void TlasGen::declare(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame) const {
        static const auto refl = reflectSpirvBindings(kTlasGenCompSpv, sizeof(kTlasGenCompSpv) / sizeof(uint32_t));
        declareDescriptorSet(g, pass, sets_[frame], 0, refl, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

}// namespace threepp::vulkan
