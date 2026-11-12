#include "threepp/renderers/vulkan/FoamWorldPipeline.hpp"
#include "threepp/renderers/vulkan/OceanExtended.hpp"
#include <array>
#include <cstring>

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/foam_world.comp.spv.h"

#include <array>
#include "threepp/objects/UniformCurrent.hpp"
#include "threepp/renderers/vulkan/shaders/foam_world.comp.current.spv.h"
#include "threepp/renderers/vulkan/shaders/foam_world.comp.audit.spv.h"
#include "threepp/renderers/vulkan/shaders/foam_world.comp.audit_current.spv.h"
#include <cstdlib>

namespace threepp::vulkan {

    static_assert(sizeof(FoamWorldPipeline::PushConstants) == 128,
                  "FoamWorldPipeline::PushConstants must match the diagnostic Pc layout");
    static_assert(offsetof(FoamWorldPipeline::PushConstants, auditAddr) == 120);

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
        //           7   = the previous frame's foam (read-only, the advection's source).
        //           8   = the swell's displacement as drawn (the CDLOD filter's output, or binding 1's).
        std::array<VkDescriptorSetLayoutBinding, 9> bb{};
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
        bb[7] = bb[6];
        bb[7].binding         = 7;
        bb[8] = bb[0];
        bb[8].binding         = 8;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bb.size());
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
        // With the extended ocean master switch off: upstream's one SPIR-V, no specialization.
        const bool oceanExt = oceanExtended();
        const auto current = oceanExt ? UniformCurrent::environment() : UniformCurrent{};
        if(current.enabled) { smci.codeSize=sizeof(kFoamWorldCurrentCompSpv); smci.pCode=kFoamWorldCurrentCompSpv; }
        // Separate SPIR-V keeps the unset rendering path exactly unchanged.
        if (oceanExt && std::getenv("THREEPP_OCEAN_FOAM_DIAGNOSTICS")) {
            smci.codeSize = current.enabled ? sizeof(kFoamWorldAuditCurrentCompSpv) : sizeof(kFoamWorldAuditCompSpv);
            smci.pCode = current.enabled ? kFoamWorldAuditCurrentCompSpv : kFoamWorldAuditCompSpv;
        }
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(foamWorld)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        // the uniform current (19, 20) and ocean_extended.glsl's constants (21, 22)
        const OceanSpec oceanSpec;
        std::array<uint32_t, 4> specData{};
        std::memcpy(&specData[0], &current.x, sizeof(float));
        std::memcpy(&specData[1], &current.z, sizeof(float));
        std::array<VkSpecializationMapEntry, 4> specEntries{};
        uint32_t specCount = 0, specWords = 0;
        if (current.enabled) {
            specEntries[specCount++] = {19, 0, 4};
            specEntries[specCount++] = {20, 4, 4};
            specWords = 2;
        }
        oceanSpec.appendTo(specData, 2u, specEntries, specCount, specWords);
        const VkSpecializationInfo spec{specCount, specEntries.data(), specWords * sizeof(uint32_t), specData.data()};
        if (specCount) stage.pSpecializationInfo = &spec;

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(foamWorld)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Pool: 6 image-samplers + 1 storage image per descriptor set, up to
        // kMaxOceans sets (one per DisplacedMesh).
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 7 * kMaxOceans;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 2 * kMaxOceans;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kMaxOceans;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
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
