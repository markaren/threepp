#include "threepp/renderers/vulkan/DeferredShade.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/renderers/vulkan/shaders/deferred_shade.comp.spv.h"

#include <array>
#include <cstring>

namespace threepp::vulkan {

    DeferredShade::DeferredShade(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        createPipeline();
        createDescriptorPool();
    }

    DeferredShade::~DeferredShade() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (gbufSampler_) vkDestroySampler(d, gbufSampler_, nullptr);
    }

    void DeferredShade::createPipeline() {
        VkDevice d = ctx_.device();

        // Nearest sampler for the G-buffer combined-image-sampler bindings. The
        // shader uses texelFetch (sampler ignored), but a valid sampler handle
        // is still required by the descriptor.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(d, &sci, nullptr, &gbufSampler_), "vkCreateSampler(deferred)");

        VkDescriptorSetLayoutBinding b[8]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding = i;
            b[i].descriptorType = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);          // camera
        set(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);          // lights
        set(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // env (PMREM)
        set(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf normal+rough
        set(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf depth
        set(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf ids
        set(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf albedo+metal
        set(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);           // out sceneHdr

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 8;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(deferred)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 16;// 4×u32 (envMipCount, width, height, flags)
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(deferred)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDeferredShadeCompSpv);
        smci.pCode    = kDeferredShadeCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(deferred_shade)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(deferred_shade)");
        vkDestroyShaderModule(d, mod, nullptr);
    }

    void DeferredShade::createDescriptorPool() {
        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 2;// camera + lights
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * 5;// env + 4 gbuf
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[2].descriptorCount = framesInFlight_ * 1;// out

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(deferred)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets_.data()),
              "vkAllocateDescriptorSets(deferred)");
    }

    void DeferredShade::rewriteDescriptors(const DescriptorWriteInputs& in) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorBufferInfo camInfo{};
            camInfo.buffer = in.cameraUbo[f];
            camInfo.offset = 0;
            camInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo lightInfo{};
            lightInfo.buffer = in.lightsUbo[f];
            lightInfo.offset = 0;
            lightInfo.range  = VK_WHOLE_SIZE;

            auto sampled = [&](VkImageView v, VkSampler s) {
                VkDescriptorImageInfo i{};
                i.sampler     = s;
                i.imageView   = v;
                i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                return i;
            };
            VkDescriptorImageInfo envInfo    = sampled(in.envView, in.envSampler);
            VkDescriptorImageInfo normalInfo = sampled(in.gbufNormal[f], gbufSampler_);
            VkDescriptorImageInfo idsInfo    = sampled(in.gbufIds[f], gbufSampler_);
            VkDescriptorImageInfo albInfo    = sampled(in.gbufAlbedo[f], gbufSampler_);
            // Depth rests in DEPTH_STENCIL_READ_ONLY_OPTIMAL (the G-buffer render
            // pass's finalLayout for the depth attachment), not SHADER_READ_ONLY.
            VkDescriptorImageInfo depthInfo{};
            depthInfo.sampler     = gbufSampler_;
            depthInfo.imageView   = in.gbufDepth[f];
            depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo outInfo{};
            outInfo.imageView   = in.sceneHdr[f];
            outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet w[8]{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet = sets_[f];
                w[n].dstBinding = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType = t;
                w[n].pImageInfo = img;
                w[n].pBufferInfo = buf;
            };
            setw(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &camInfo);
            setw(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &lightInfo);
            setw(2, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo,    nullptr);
            setw(3, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo, nullptr);
            setw(4, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo,  nullptr);
            setw(5, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &idsInfo,    nullptr);
            setw(6, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albInfo,    nullptr);
            setw(7, 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &outInfo,    nullptr);
            vkUpdateDescriptorSets(ctx_.device(), 8, w, 0, nullptr);
        }
    }

    void DeferredShade::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                       uint32_t width, uint32_t height, uint32_t envMipCount) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        const uint32_t pc[4] = {envMipCount, width, height, 0u};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    }

}// namespace threepp::vulkan
