#include "threepp/renderers/vulkan/FoamWorldPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/foam_world.comp.spv.h"

#include <array>

namespace threepp::vulkan {

    static_assert(sizeof(FoamWorldPipeline::PushConstants) == 80,
                  "FoamWorldPipeline::PushConstants must match foam_world.comp's Pc layout (80 bytes)");

    FoamWorldPipeline::FoamWorldPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    FoamWorldPipeline::~FoamWorldPipeline() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
    }

    void FoamWorldPipeline::createPipeline() {
        // Bindings: 0..5 = 6 cascade combined image-samplers (h0/d0/h1/d1/h2/d2);
        //           6   = foam storage image (read-via-imageLoad + write-via-imageStore).
        std::array<VkDescriptorSetLayoutBinding, 7> bb{};
        for (uint32_t i = 0; i < 6; ++i) {
            bb[i].binding         = i;
            bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bb[i].descriptorCount = 1;
            bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bb[6].binding         = 6;
        bb[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bb[6].descriptorCount = 1;
        bb[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = uint32_t(bb.size());
        dlci.pBindings    = bb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(foamWorld)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(foamWorld)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kFoamWorldCompSpv);
        smci.pCode    = kFoamWorldCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(foamWorld)");

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
              "vkCreateComputePipelines(foamWorld)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Pool: 6 image-samplers + 1 storage image per descriptor set, up to
        // kMaxOceans sets (one per DisplacedMesh).
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 6 * kMaxOceans;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 1 * kMaxOceans;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kMaxOceans;
        dpci.poolSizeCount = uint32_t(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(foamWorld)");

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // REPEAT — the foam tile wraps with the FFT pattern; sampling at any
        // world XZ outside the tile folds back into the same texture.
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_),
              "vkCreateSampler(foamWorld)");
    }

    VkDescriptorSet FoamWorldPipeline::allocateMeshDescriptorSet() {
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = descPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsLayout_;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds),
              "vkAllocateDescriptorSets(foamWorld)");
        return ds;
    }

    void FoamWorldPipeline::recordDispatch(VkCommandBuffer cb,
                                           VkDescriptorSet ds,
                                           const PushConstants& pc) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        const uint32_t groups = (pc.foamRes + 7u) / 8u;
        vkCmdDispatch(cb, groups, groups, 1);
    }

}// namespace threepp::vulkan
