#include "threepp/renderers/vulkan/GbufResolve.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/gbuf_resolve.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuf_resolve_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuf_resolve_depth.vert.spv.h"

namespace threepp::vulkan {

    GbufResolve::GbufResolve(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        createComputePipeline();
        createDepthPipeline();
    }

    GbufResolve::~GbufResolve() {
        VkDevice d = ctx_.device();
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipe_)           vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_)     vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        if (depthDescPool_)  vkDestroyDescriptorPool(d, depthDescPool_, nullptr);
        if (depthPipe_)      vkDestroyPipeline(d, depthPipe_, nullptr);
        if (depthPipeLayout_) vkDestroyPipelineLayout(d, depthPipeLayout_, nullptr);
        if (depthDsLayout_)  vkDestroyDescriptorSetLayout(d, depthDsLayout_, nullptr);
        if (depthSampler_)   vkDestroySampler(d, depthSampler_, nullptr);
    }

    void GbufResolve::createComputePipeline() {
        VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(gbufResolve)");

        // Bindings 0-5: MS combined-image-samplers (normal/motion/ids/uv/
        // albedo/depth). Bindings 6-10: single-sample STORAGE resolve
        // targets (normal/motion/ids/uv/albedo) — depth has no compute
        // resolve target (see createDepthPipeline).
        VkDescriptorSetLayoutBinding b[11]{};
        for (uint32_t i = 0; i < 6; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        for (uint32_t i = 6; i < 11; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 11;
        dslci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dslci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(gbufResolve)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = 3 * sizeof(uint32_t);// width, height, sampleCount

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(gbufResolve)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kGbufResolveCompSpv);
        smci.pCode    = kGbufResolveCompSpv;
        VkShaderModule sm = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &sm),
              "vkCreateShaderModule(gbuf_resolve)");

        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName  = "main";
        cpci.layout       = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(gbufResolve)");
        vkDestroyShaderModule(d, sm, nullptr);
    }

    void GbufResolve::createDepthPipeline() {
        VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &depthSampler_), "vkCreateSampler(gbufResolveDepth)");

        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;// depthMS
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1].binding         = 1;
        b[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;// idsResolved
        b[1].descriptorCount = 1;
        b[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 2;
        dslci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dslci, nullptr, &depthDsLayout_),
              "vkCreateDescriptorSetLayout(gbufResolveDepth)");

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &depthDsLayout_;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &depthPipeLayout_),
              "vkCreatePipelineLayout(gbufResolveDepth)");

        VkShaderModuleCreateInfo vsmci{};
        vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vsmci.codeSize = sizeof(kGbufResolveDepthVertSpv);
        vsmci.pCode    = kGbufResolveDepthVertSpv;
        VkShaderModule vm = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &vsmci, nullptr, &vm),
              "vkCreateShaderModule(gbuf_resolve_depth.vert)");
        VkShaderModuleCreateInfo fsmci{};
        fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fsmci.codeSize = sizeof(kGbufResolveDepthFragSpv);
        fsmci.pCode    = kGbufResolveDepthFragSpv;
        VkShaderModule fm = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &fsmci, nullptr, &fm),
              "vkCreateShaderModule(gbuf_resolve_depth.frag)");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vm;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fm;
        stages[1].pName  = "main";

        // No vertex input — fullscreen triangle derives positions from
        // gl_VertexIndex (see gbuf_resolve_depth.vert).
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;// output (depthResolved) is single-sample

        // Depth ALWAYS write, ALWAYS pass — every pixel in the fullscreen
        // triangle writes gl_FragDepth unconditionally. Per the Vulkan spec,
        // depthWriteEnable only has effect when depthTestEnable is TRUE (a
        // disabled depth test implicitly disables the write too) — so this
        // must enable the test with an always-pass compare op, NOT disable
        // it, even though nothing else contends for depthResolved here.
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_ALWAYS;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates    = dynStates;

        // Dynamic rendering (no VkRenderPass/VkFramebuffer) — mirrors the
        // overlay pass's approach. Depth-only: no color attachments.
        VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
        VkPipelineRenderingCreateInfo rci{};
        rci.sType                = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rci.depthAttachmentFormat = depthFmt;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext               = &rci;
        gpci.stageCount          = 2;
        gpci.pStages             = stages;
        gpci.pVertexInputState   = &vi;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState      = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState   = &ms;
        gpci.pDepthStencilState  = &ds;
        gpci.pColorBlendState    = &cb;
        gpci.pDynamicState       = &dyn;
        gpci.layout              = depthPipeLayout_;
        gpci.renderPass          = VK_NULL_HANDLE;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &depthPipe_),
              "vkCreateGraphicsPipelines(gbufResolveDepth)");

        vkDestroyShaderModule(d, vm, nullptr);
        vkDestroyShaderModule(d, fm, nullptr);
    }

    void GbufResolve::rewriteDescriptors(const DescriptorWriteInputs& in) {
        VkDevice d = ctx_.device();

        // ── Compute-resolve descriptor sets ─────────────────────────────
        if (descPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(d, descPool_, nullptr);
            descPool_ = VK_NULL_HANDLE;
            sets_.clear();
        }
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * 6;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_ * 5;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(gbufResolve)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = framesInFlight_;
        dsai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(d, &dsai, sets_.data()),
              "vkAllocateDescriptorSets(gbufResolve)");

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorImageInfo msInfo[6]{};
            const VkImageView msViews[6] = {in.normalMS[f], in.motionMS[f], in.idsMS[f],
                                            in.uvMS[f], in.albedoMS[f], in.depthMS[f]};
            for (uint32_t i = 0; i < 6; ++i) {
                msInfo[i].sampler     = sampler_;
                msInfo[i].imageView   = msViews[i];
                msInfo[i].imageLayout = (i == 5) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            VkDescriptorImageInfo storeInfo[5]{};
            const VkImageView storeViews[5] = {in.normalResolved[f], in.motionResolved[f],
                                               in.idsResolved[f], in.uvResolved[f], in.albedoResolved[f]};
            for (uint32_t i = 0; i < 5; ++i) {
                storeInfo[i].imageView   = storeViews[i];
                storeInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }

            VkWriteDescriptorSet w[11]{};
            for (uint32_t i = 0; i < 6; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = sets_[f];
                w[i].dstBinding      = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[i].pImageInfo      = &msInfo[i];
            }
            for (uint32_t i = 0; i < 5; ++i) {
                w[6 + i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[6 + i].dstSet          = sets_[f];
                w[6 + i].dstBinding      = 6 + i;
                w[6 + i].descriptorCount = 1;
                w[6 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[6 + i].pImageInfo      = &storeInfo[i];
            }
            vkUpdateDescriptorSets(d, 11, w, 0, nullptr);
        }

        // ── Depth-resolve descriptor sets ───────────────────────────────
        if (depthDescPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(d, depthDescPool_, nullptr);
            depthDescPool_ = VK_NULL_HANDLE;
            depthSets_.clear();
        }
        VkDescriptorPoolSize dsizes[1]{};
        dsizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        dsizes[0].descriptorCount = framesInFlight_ * 2;
        VkDescriptorPoolCreateInfo ddpci{};
        ddpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ddpci.maxSets       = framesInFlight_;
        ddpci.poolSizeCount = 1;
        ddpci.pPoolSizes    = dsizes;
        check(vkCreateDescriptorPool(d, &ddpci, nullptr, &depthDescPool_),
              "vkCreateDescriptorPool(gbufResolveDepth)");

        std::vector<VkDescriptorSetLayout> dlayouts(framesInFlight_, depthDsLayout_);
        VkDescriptorSetAllocateInfo ddsai{};
        ddsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ddsai.descriptorPool     = depthDescPool_;
        ddsai.descriptorSetCount = framesInFlight_;
        ddsai.pSetLayouts        = dlayouts.data();
        depthSets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(d, &ddsai, depthSets_.data()),
              "vkAllocateDescriptorSets(gbufResolveDepth)");

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorImageInfo depthMsInfo{};
            depthMsInfo.sampler     = depthSampler_;
            depthMsInfo.imageView   = in.depthMS[f];
            depthMsInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo idsInfo{};
            idsInfo.sampler     = depthSampler_;
            idsInfo.imageView   = in.idsResolved[f];
            idsInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet w[2]{};
            w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet          = depthSets_[f];
            w[0].dstBinding      = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo      = &depthMsInfo;
            w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet          = depthSets_[f];
            w[1].dstBinding      = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].pImageInfo      = &idsInfo;
            vkUpdateDescriptorSets(d, 2, w, 0, nullptr);
        }
    }

    void GbufResolve::recordComputeResolve(VkCommandBuffer cb, uint32_t frame,
                                           uint32_t width, uint32_t height, uint32_t sampleCount) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &sets_[frame], 0, nullptr);
        const uint32_t pc[3] = {width, height, sampleCount};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1u);
    }

    void GbufResolve::recordDepthResolve(VkCommandBuffer cb, uint32_t frame,
                                         uint32_t width, uint32_t height,
                                         VkImageView /*depthMSView*/, VkImageView /*idsResolvedView*/,
                                         VkImageView depthResolvedView) {
        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = depthResolvedView;
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;// every pixel written unconditionally
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.offset = {0, 0};
        ri.renderArea.extent = {width, height};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depthAtt;
        vkCmdBeginRendering(cb, &ri);

        VkViewport vp{};
        vp.width    = static_cast<float>(width);
        vp.height   = static_cast<float>(height);
        vp.minDepth = 0.f;
        vp.maxDepth = 1.f;
        vkCmdSetViewport(cb, 0, 1, &vp);
        VkRect2D scissor{};
        scissor.extent = {width, height};
        vkCmdSetScissor(cb, 0, 1, &scissor);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeLayout_,
                                0, 1, &depthSets_[frame], 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);

        vkCmdEndRendering(cb);
    }

}// namespace threepp::vulkan
