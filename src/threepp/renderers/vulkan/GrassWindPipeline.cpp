#include "threepp/renderers/vulkan/GrassWindPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/grass_wind.comp.spv.h"

namespace threepp::vulkan {

    static_assert(sizeof(GrassWindPipeline::PushConstants) == 48,
                  "GrassWindPipeline::PushConstants must be 48 bytes (44 used + 4 tail pad)");

    GrassWindPipeline::GrassWindPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    GrassWindPipeline::~GrassWindPipeline() {
        VkDevice d = ctx_.device();
        if (pipeline_) vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
    }

    void GrassWindPipeline::createPipeline() {
        // No descriptor sets — everything is passed by device address.
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 0;
        plci.pSetLayouts = nullptr;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(grass_wind)");

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kGrassWindCompSpv);
        smci.pCode = kGrassWindCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(grass_wind)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(grass_wind)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);
    }

    void GrassWindPipeline::recordDispatch(VkCommandBuffer cb, const PushConstants& pc) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        const uint32_t groups = (pc.vertexCount + 63u) / 64u;
        vkCmdDispatch(cb, groups, 1, 1);
    }

}// namespace threepp::vulkan
