#include "threepp/renderers/vulkan/WaterDisplacePipeline.hpp"
#include "threepp/renderers/vulkan/OceanExtended.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/water_displace.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/water_displace.comp.chop.spv.h"

#include "threepp/renderers/vulkan/shaders/water_displace_cdlod.comp.spv.h"
#include <array>
#include <cstddef>
#include "threepp/renderers/vulkan/shaders/water_displace_cdlod.comp.chop.spv.h"

namespace threepp::vulkan {

    static_assert(sizeof(WaterDisplacePipeline::PushConstants) == 160,
                  "WaterDisplacePipeline host parameters must contain a 128-byte Pc and 32-byte surface tail");
    static_assert(offsetof(WaterDisplacePipeline::PushConstants, skirtRings) == WaterDisplacePipeline::kPushConstantBytes);

    WaterDisplacePipeline::WaterDisplacePipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    WaterDisplacePipeline::~WaterDisplacePipeline() {
        VkDevice d = ctx_.device();
        if (lodChopPipeline_) vkDestroyPipeline(d, lodChopPipeline_, nullptr);
        if (lodPipeline_) vkDestroyPipeline(d, lodPipeline_, nullptr);
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        if (chopPipeline_)       vkDestroyPipeline(d, chopPipeline_, nullptr);
        if (chopPipelineLayout_) vkDestroyPipelineLayout(d, chopPipelineLayout_, nullptr);
        if (chopDsLayout_)       vkDestroyDescriptorSetLayout(d, chopDsLayout_, nullptr);
        if (chopDescPool_)       vkDestroyDescriptorPool(d, chopDescPool_, nullptr);
    }

    void WaterDisplacePipeline::createPipeline() {
        // 8 combined-image-sampler bindings — three (height, displacement)
        // cascade pairs at 0-5, then the wake field's ripples (an array, a
        // layer a patch) at kRippleBinding and the wave climate's map at
        // kClimateBinding.
        std::array<VkDescriptorSetLayoutBinding, 8> bb{};
        for (uint32_t i = 0; i < bb.size(); ++i) {
            bb[i].binding         = i < 6u ? i : (i == 6u ? kRippleBinding : kClimateBinding);
            bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bb[i].descriptorCount = 1;
            bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bb.size());
        dlci.pBindings    = bb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(displace)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = kPushConstantBytes;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(displace)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kWaterDisplaceCompSpv);
        smci.pCode    = kWaterDisplaceCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(displace)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        const OceanSpec oceanSpec;// ocean_extended.glsl: none with the extended ocean off
        stage.pSpecializationInfo = oceanSpec.info();

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(displace)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);


        // Pool sized for up to kMaxOceans DisplacedMesh instances at once.
        // No FREE_DESCRIPTOR_SET_BIT: sets are released when the pool is
        // destroyed; bump kMaxOceans if scenes ever exceed it.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = static_cast<uint32_t>(bb.size()) * kMaxOceans;// every binding of the layout, a set an ocean
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kMaxOceans;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(displace)");

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_),
              "vkCreateSampler(displace)");
    }

    VkDescriptorSet WaterDisplacePipeline::allocateMeshDescriptorSet() {
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = descPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds),
              "vkAllocateDescriptorSets(displace)");
        return ds;
    }

    void WaterDisplacePipeline::createChopPipeline() {
        // The plain six bindings, then the chop normals' (slope, jacDiag)
        // pairs of cascades 0 and 1 (water_displace.comp bindings 6-9), then
        // the wake field's ripples at kRippleBinding and the wave climate's
        // map at kClimateBinding.
        std::array<VkDescriptorSetLayoutBinding, 12> bb{};
        for (uint32_t i = 0; i < bb.size(); ++i) {
            bb[i].binding         = i < 10u ? i : (i == 10u ? kRippleBinding : kClimateBinding);
            bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bb[i].descriptorCount = 1;
            bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bb.size());
        dlci.pBindings    = bb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &chopDsLayout_),
              "vkCreateDescriptorSetLayout(displace chop)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = kPushConstantBytes;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &chopDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &chopPipelineLayout_),
              "vkCreatePipelineLayout(displace chop)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kWaterDisplaceChopCompSpv);
        smci.pCode    = kWaterDisplaceChopCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(displace chop)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        const OceanSpec oceanSpec;// ocean_extended.glsl: none with the extended ocean off
        stage.pSpecializationInfo = oceanSpec.info();
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = chopPipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &chopPipeline_),
              "vkCreateComputePipelines(displace chop)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = static_cast<uint32_t>(bb.size()) * kMaxOceans;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kMaxOceans;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &chopDescPool_),
              "vkCreateDescriptorPool(displace chop)");
    }

    VkDescriptorSet WaterDisplacePipeline::allocateChopDescriptorSet() {
        if (chopPipeline_ == VK_NULL_HANDLE) createChopPipeline();
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = chopDescPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &chopDsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds),
              "vkAllocateDescriptorSets(displace chop)");
        return ds;
    }

    void WaterDisplacePipeline::recordDispatch(VkCommandBuffer cb,
                                               VkDescriptorSet ds,
                                               const PushConstants& pc,
                                               bool chop) {
        chop = chop && chopPipeline_ != VK_NULL_HANDLE;
        const VkPipelineLayout layout = chop ? chopPipelineLayout_ : pipelineLayout_;
        VkPipeline& lod = chop ? lodChopPipeline_ : lodPipeline_;
        if (pc.lodPatches && lod == VK_NULL_HANDLE) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = chop ? sizeof(kWaterDisplaceCdlodChopCompSpv) : sizeof(kWaterDisplaceCdlodCompSpv);
            smci.pCode = chop ? kWaterDisplaceCdlodChopCompSpv : kWaterDisplaceCdlodCompSpv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod), "vkCreateShaderModule(cdlod)");
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.layout = layout;
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = mod;
            cpci.stage.pName = "main";
            const OceanSpec oceanSpec;
            cpci.stage.pSpecializationInfo = oceanSpec.info();
            const VkResult result = vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &lod);
            vkDestroyShaderModule(ctx_.device(), mod, nullptr);
            check(result, "vkCreateComputePipelines(cdlod)");
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pc.lodPatches ? lod : (chop ? chopPipeline_ : pipeline_));
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, kPushConstantBytes, &pc);
        const uint32_t groups = (pc.vertexCount + 63u) / 64u;
        vkCmdDispatch(cb, groups, 1, 1);
    }

}// namespace threepp::vulkan
