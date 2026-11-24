#include "threepp/renderers/vulkan/GrassWindPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/grass_wind.comp.spv.h"

#include <cstddef>

namespace threepp::vulkan {

    static_assert(sizeof(GrassWindPipeline::PushConstants) == 104,
                  "GrassWindPipeline::PushConstants drifted from grass_wind.comp's Pc block");
    static_assert(offsetof(GrassWindPipeline::PushConstants, downwashAddr) == 48,
                  "GrassWindPipeline::PushConstants::downwashAddr must land at 48");

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

        pipeline_ = createComputePipeline(ctx_.device(), ctx_.pipelineCache(), pipelineLayout_, kGrassWindCompSpv,
                                          sizeof(kGrassWindCompSpv), "grass_wind");
    }

    void GrassWindPipeline::recordDispatch(VkCommandBuffer cb, const PushConstants& pc) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        const uint32_t groups = (pc.vertexCount + 63u) / 64u;
        vkCmdDispatch(cb, groups, 1, 1);
    }

}// namespace threepp::vulkan
