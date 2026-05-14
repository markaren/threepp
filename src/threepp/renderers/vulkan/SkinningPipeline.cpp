#include "threepp/renderers/vulkan/SkinningPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/skinning.comp.spv.h"

#include <array>
#include <stdexcept>
#include <string>

namespace threepp::vulkan {

    SkinningPipeline::SkinningPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    SkinningPipeline::~SkinningPipeline() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
    }

    void SkinningPipeline::createPipeline() {
        // Set-0 layout: 7 storage buffers.
        std::array<VkDescriptorSetLayoutBinding, 7> sb{};
        for (uint32_t i = 0; i < sb.size(); ++i) {
            sb[i].binding         = i;
            sb[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sb[i].descriptorCount = 1;
            sb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = static_cast<uint32_t>(sb.size());
        slci.pBindings    = sb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &slci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(skinning)");

        // Pipeline layout: set 0 + push constant (vertexCount as u32).
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(skinning)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kSkinningCompSpv);
        smci.pCode    = kSkinningCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(skinning)");

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
              "vkCreateComputePipelines(skinning)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Descriptor pool sized for up to kMaxSkinnedMeshes meshes — each
        // set holds 7 storage buffers. FREE_DESCRIPTOR_SET_BIT lets the
        // per-frame liveCheck-expired erase loop call vkFreeDescriptorSets
        // when a SkinnedMeshState is destroyed; without it the slot leaks
        // until the pool exhausts on repeated remove/re-add.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kMaxSkinnedMeshes * 7;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = kMaxSkinnedMeshes;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(skinning)");
    }

    VkDescriptorSet SkinningPipeline::allocateMeshDescriptorSet() {
        VkDescriptorSetAllocateInfo dsAi{};
        dsAi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAi.descriptorPool     = descPool_;
        dsAi.descriptorSetCount = 1;
        dsAi.pSetLayouts        = &dsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(ctx_.device(), &dsAi, &ds);
        if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
            throw std::runtime_error(
                    "SkinningPipeline: descriptor pool exhausted — "
                    "scene has more than kMaxSkinnedMeshes=" +
                    std::to_string(kMaxSkinnedMeshes) + " SkinnedMesh instances "
                    "(live count " + std::to_string(liveSetCount_) + "). "
                    "Bump kMaxSkinnedMeshes in SkinningPipeline.hpp.");
        }
        check(r, "vkAllocateDescriptorSets(skinning)");
        ++liveSetCount_;
        return ds;
    }

    void SkinningPipeline::freeMeshDescriptorSet(VkDescriptorSet ds) {
        if (ds == VK_NULL_HANDLE) return;
        vkFreeDescriptorSets(ctx_.device(), descPool_, 1, &ds);
        if (liveSetCount_ > 0) --liveSetCount_;
    }

    void SkinningPipeline::bindPipeline(VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    }

    void SkinningPipeline::recordDispatch(VkCommandBuffer cb,
                                          VkDescriptorSet ds,
                                          uint32_t        vertexCount) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(vertexCount), &vertexCount);
        const uint32_t gx = (vertexCount + 63u) / 64u;
        vkCmdDispatch(cb, gx, 1u, 1u);
    }

}// namespace threepp::vulkan
