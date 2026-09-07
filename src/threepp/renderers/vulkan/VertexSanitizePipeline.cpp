#include "threepp/renderers/vulkan/VertexSanitizePipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/vertex_sanitize.comp.spv.h"

namespace threepp::vulkan {

    VertexSanitizePipeline::VertexSanitizePipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    VertexSanitizePipeline::~VertexSanitizePipeline() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
    }

    void VertexSanitizePipeline::createPipeline() {
        // Set-0 layout: one storage buffer (the exported positions).
        VkDescriptorSetLayoutBinding sb{};
        sb.binding         = 0;
        sb.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sb.descriptorCount = 1;
        sb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = 1;
        slci.pBindings    = &sb;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &slci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(vertexSanitize)");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(uint32_t);// vertexCount

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(vertexSanitize)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kVertexSanitizeCompSpv);
        smci.pCode    = kVertexSanitizeCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(vertexSanitize)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(vertexSanitize)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // FREE_DESCRIPTOR_SET_BIT so disableVertexInterop can return a set;
        // without it a repeated enable/disable cycle exhausts the pool.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kMaxRecords;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = kMaxRecords;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(vertexSanitize)");
    }

    VkDescriptorSet VertexSanitizePipeline::allocateRecordDescriptorSet(VkBuffer posBuf) {
        if (posBuf == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &dsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        // Pool exhaustion is NOT fatal here, unlike the tet path: the caller's
        // fallback is "interop without the finiteness guard", which is what the
        // producer asked for when it set validate=false anyway. Throwing would
        // turn a 65th interop mesh into a crash.
        if (vkAllocateDescriptorSets(ctx_.device(), &ai, &ds) != VK_SUCCESS)
            return VK_NULL_HANDLE;

        VkDescriptorBufferInfo bi{};
        bi.buffer = posBuf;
        bi.offset = 0;
        bi.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet wr{};
        wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet          = ds;
        wr.dstBinding      = 0;
        wr.descriptorCount = 1;
        wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(ctx_.device(), 1, &wr, 0, nullptr);
        return ds;
    }

    void VertexSanitizePipeline::freeRecordDescriptorSet(VkDescriptorSet ds) {
        if (ds == VK_NULL_HANDLE) return;
        vkFreeDescriptorSets(ctx_.device(), descPool_, 1, &ds);
    }

    void VertexSanitizePipeline::recordDispatch(VkCommandBuffer cb, VkDescriptorSet ds,
                                                uint32_t vertexCount) {
        if (ds == VK_NULL_HANDLE || vertexCount == 0u) return;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(vertexCount), &vertexCount);
        vkCmdDispatch(cb, (vertexCount + 63u) / 64u, 1u, 1u);
    }

}// namespace threepp::vulkan
