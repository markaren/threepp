#include "threepp/renderers/vulkan/TetSkinningPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/tet_skinning.comp.spv.h"

#include <array>
#include <stdexcept>
#include <string>

namespace threepp::vulkan {

    TetSkinningPipeline::TetSkinningPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    TetSkinningPipeline::~TetSkinningPipeline() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
    }

    void TetSkinningPipeline::createPipeline() {
        // Set-0 layout: kBindingsPerSet storage buffers (see tet_skinning.comp).
        std::array<VkDescriptorSetLayoutBinding, kBindingsPerSet> sb{};
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
              "vkCreateDescriptorSetLayout(tetSkinning)");

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
              "vkCreatePipelineLayout(tetSkinning)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kTetSkinningCompSpv);
        smci.pCode    = kTetSkinningCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(tetSkinning)");

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
              "vkCreateComputePipelines(tetSkinning)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Descriptor pool sized for up to kMaxTetMeshes meshes. FREE_DESCRIPTOR_SET_BIT
        // lets a TetMeshState free its set when the soft body is removed.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kMaxTetMeshes * kBindingsPerSet;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = kMaxTetMeshes;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(tetSkinning)");
    }

    VkDescriptorSet TetSkinningPipeline::allocateMeshDescriptorSet() {
        VkDescriptorSetAllocateInfo dsAi{};
        dsAi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAi.descriptorPool     = descPool_;
        dsAi.descriptorSetCount = 1;
        dsAi.pSetLayouts        = &dsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(ctx_.device(), &dsAi, &ds);
        if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
            throw std::runtime_error(
                    "TetSkinningPipeline: descriptor pool exhausted — scene has more "
                    "than kMaxTetMeshes=" +
                    std::to_string(kMaxTetMeshes) + " tet-skinned meshes (live count " +
                    std::to_string(liveSetCount_) + "). Bump kMaxTetMeshes in "
                    "TetSkinningPipeline.hpp.");
        }
        check(r, "vkAllocateDescriptorSets(tetSkinning)");
        ++liveSetCount_;
        return ds;
    }

    void TetSkinningPipeline::freeMeshDescriptorSet(VkDescriptorSet ds) {
        if (ds == VK_NULL_HANDLE) return;
        vkFreeDescriptorSets(ctx_.device(), descPool_, 1, &ds);
        if (liveSetCount_ > 0) --liveSetCount_;
    }

    void TetSkinningPipeline::bindPipeline(VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    }

    void TetSkinningPipeline::recordDispatch(VkCommandBuffer cb,
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
