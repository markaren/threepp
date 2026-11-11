#include "threepp/renderers/vulkan/BillboardHdrLayer.hpp"

#include "threepp/renderers/vulkan/TransientPool.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

// The sprites: the field billboard vertex stage as it is, and the BB_HDR_LAYER
// variant of its fragment stage (linear, two outputs). The composite and the
// depth copy: the shared fullscreen triangle and their own fragment stages.
#include "threepp/renderers/vulkan/shaders/billboard_layer_depth.frag.ms.spv.h"
#include "threepp/renderers/vulkan/shaders/billboard_layer_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_fullscreen.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_billboard.frag.layer.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_billboard.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_hdr_layer.frag.spv.h"

namespace threepp::vulkan {

    namespace {

        VkShaderModule makeModule(VkDevice d, const uint32_t* spv, size_t bytes, const char* label) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes;
            smci.pCode    = spv;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smci, nullptr, &m), label);
            return m;
        }

        constexpr VkColorComponentFlags kRgb =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
        constexpr VkColorComponentFlags kRgba = kRgb | VK_COLOR_COMPONENT_A_BIT;

        // The fixed-function state every pipeline here shares: no vertex
        // input, no culling, dynamic viewport and scissor.
        struct CommonState {
            VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            VkPipelineViewportStateCreateInfo      vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            VkDynamicState                         dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                                                   VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo       dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            CommonState(VkPrimitiveTopology topology) {
                ia.topology      = topology;
                vp.viewportCount = 1;
                vp.scissorCount  = 1;
                rs.polygonMode   = VK_POLYGON_MODE_FILL;
                rs.cullMode      = VK_CULL_MODE_NONE;
                rs.frontFace     = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                rs.lineWidth     = 1.0f;
                dyn.dynamicStateCount = 2;
                dyn.pDynamicStates    = dynStates;
            }
            void apply(VkGraphicsPipelineCreateInfo& g) const {
                g.pVertexInputState   = &vi;
                g.pInputAssemblyState = &ia;
                g.pViewportState      = &vp;
                g.pRasterizationState = &rs;
                g.pDynamicState       = &dyn;
            }
        };

    }// namespace

    // ── The shared pipelines ────────────────────────────────────────────────
    BillboardHdrLayerPipelines::BillboardHdrLayerPipelines(VulkanContext& ctx, VkPipelineLayout spriteLayout,
                                                           VkFormat colorFormat,
                                                           VkSampleCountFlagBits overlaySamples)
        : ctx_(ctx) {
        // texelFetch ignores filtering, but a combined image sampler still
        // needs a sampler object; NEAREST states the intent.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(bbhdr)");

        createSpritePipelines(spriteLayout);
        createCompositePipelines(colorFormat, overlaySamples);
        createCopyPipelines();
    }

    BillboardHdrLayerPipelines::~BillboardHdrLayerPipelines() {
        const VkDevice d = ctx_.device();
        for (VkPipeline p : {spriteAdd_, spriteOver_, compositeOver_[0], compositeOver_[1], compositeAdd_[0],
                             compositeAdd_[1], copyPipe_, copyPipeMs_})
            if (p) vkDestroyPipeline(d, p, nullptr);
        if (compositeLayout_)   vkDestroyPipelineLayout(d, compositeLayout_, nullptr);
        if (copyLayout_)        vkDestroyPipelineLayout(d, copyLayout_, nullptr);
        if (compositeDsLayout_) vkDestroyDescriptorSetLayout(d, compositeDsLayout_, nullptr);
        if (copyDsLayout_)      vkDestroyDescriptorSetLayout(d, copyDsLayout_, nullptr);
        if (sampler_)           vkDestroySampler(d, sampler_, nullptr);
    }

    // ── One view's layer ────────────────────────────────────────────────────
    BillboardHdrLayer::BillboardHdrLayer(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight,
                                         const BillboardHdrLayerPipelines& pipes, bool secondaryView)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight), pipes_(pipes),
          variant_(secondaryView ? 1u : 0u) {
        over_.resize(framesInFlight_);
        add_.resize(framesInFlight_);
        createDescriptors();
    }

    BillboardHdrLayer::~BillboardHdrLayer() {
        if (descPool_) vkDestroyDescriptorPool(ctx_.device(), descPool_, nullptr);
        destroyImages();
    }

    void BillboardHdrLayer::destroyImages() {
        const VkDevice d = ctx_.device();
        for (auto& img : over_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : add_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : depth_) destroyImage2D(ctx_.allocator(), d, img);
        depth_.clear();
        width_ = height_ = 0;
    }

    // A dedicated allocation, not the TransientPool: the glow pass's scratch
    // already sits in the Tail group under the frame slots, and two classes
    // sharing slot numbers there would alias each other's images.
    Image2D BillboardHdrLayer::createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                                           VkImageAspectFlags aspect, const char* label) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = format;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = usage;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // No pool: createImageMaybePooled then gives the image an allocation
        // of its own.
        check(createImageMaybePooled(ctx_.allocator(), nullptr, TransientSpan{0}, 0, ici, &out.image, &out.alloc),
              label);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = aspect;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vulkan::createImageView(bbhdr)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view, label);
        return out;
    }

    // Every image here lives in GENERAL for its whole life — a legal layout for
    // a colour or depth attachment and for a sampled read — so the passes that
    // write and read them need no layout dance, as in BillboardGlowPass.
    void BillboardHdrLayer::transitionFresh(const std::vector<std::pair<VkImage, VkImageAspectFlags>>& images) {
        if (images.empty()) return;
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(bbhdr)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(bbhdr)");

        std::vector<VkImageMemoryBarrier> bs(images.size());
        for (std::size_t i = 0; i < images.size(); ++i) {
            VkImageMemoryBarrier& b = bs[i];
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = images[i].first;
            b.subresourceRange.aspectMask = images[i].second;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(bs.size()), bs.data());

        check(vkEndCommandBuffer(cb), "end one-shot cb(bbhdr)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(bbhdr)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(bbhdr)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    bool BillboardHdrLayer::ensureImages(uint32_t width, uint32_t height, bool ownDepth) {
        if (width == 0 || height == 0) return false;
        const bool sameExtent = width == width_ && height == height_;
        if (sameExtent && (!ownDepth || hasOwnDepth())) return true;

        std::vector<std::pair<VkImage, VkImageAspectFlags>> fresh;
        if (!sameExtent) {
            // Called in the prepare window, after this frame slot's fence. On
            // a resize the other slot's frame may still be reading the old
            // images, so wait before they go. Rare.
            if (width_ > 0) vkDeviceWaitIdle(ctx_.device());
            destroyImages();
            width_  = width;
            height_ = height;
            constexpr VkImageUsageFlags kColorUse = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            for (uint32_t f = 0; f < framesInFlight_; ++f) {
                over_[f] = createImage(width, height, kColorFormat, kColorUse, VK_IMAGE_ASPECT_COLOR_BIT,
                                       "vmaCreateImage(bbhdr.over)");
                add_[f]  = createImage(width, height, kColorFormat, kColorUse, VK_IMAGE_ASPECT_COLOR_BIT,
                                       "vmaCreateImage(bbhdr.add)");
                fresh.emplace_back(over_[f].image, VK_IMAGE_ASPECT_COLOR_BIT);
                fresh.emplace_back(add_[f].image, VK_IMAGE_ASPECT_COLOR_BIT);
            }
            rewriteDescriptors();
        }
        if (ownDepth && !hasOwnDepth()) {
            depth_.resize(framesInFlight_);
            for (uint32_t f = 0; f < framesInFlight_; ++f) {
                depth_[f] = createImage(width, height, kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                        VK_IMAGE_ASPECT_DEPTH_BIT, "vmaCreateImage(bbhdr.depth)");
                fresh.emplace_back(depth_[f].image, VK_IMAGE_ASPECT_DEPTH_BIT);
            }
            std::fill(copySetSerials_.begin(), copySetSerials_.end(), uint64_t{0});
        }
        transitionFresh(fresh);
        return true;
    }

    // ── The sprite pipelines ────────────────────────────────────────────────
    // The field billboard pipeline state (vertex-less strip, no culling,
    // reverse-Z GREATER_OR_EQUAL test with depth writes OFF, one sample) with
    // two colour attachments and the blends the class comment lists.
    void BillboardHdrLayerPipelines::createSpritePipelines(VkPipelineLayout spriteLayout) {
        const VkDevice d = ctx_.device();
        VkShaderModule vert = makeModule(d, kParticleFieldBillboardVertSpv, sizeof(kParticleFieldBillboardVertSpv),
                                         "vkCreateShaderModule(bbhdr.sprite.vert)");
        VkShaderModule frag = makeModule(d, kParticleFieldBillboardLayerFragSpv,
                                         sizeof(kParticleFieldBillboardLayerFragSpv),
                                         "vkCreateShaderModule(bbhdr.sprite.frag)");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName  = "main";

        const CommonState common(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);

        const VkFormat fmts[2] = {kColorFormat, kColorFormat};
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 2;
        prci.pColorAttachmentFormats = fmts;
        prci.depthAttachmentFormat   = kDepthFormat;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // [0] = the alpha-over layer, [1] = the additive layer.
        VkPipelineColorBlendAttachmentState cbas[2]{};
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 2;
        cb.pAttachments    = cbas;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext              = &prci;
        gpci.stageCount         = 2;
        gpci.pStages            = stages;
        common.apply(gpci);
        gpci.pMultisampleState  = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState   = &cb;
        gpci.layout             = spriteLayout;

        // ADDITIVE fields: nothing reaches the alpha-over layer (empty write
        // mask), the radiance sums into the additive one.
        cbas[0].colorWriteMask      = 0;
        cbas[1].blendEnable         = VK_TRUE;
        cbas[1].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas[1].colorBlendOp        = VK_BLEND_OP_ADD;
        cbas[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbas[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas[1].alphaBlendOp        = VK_BLEND_OP_ADD;
        cbas[1].colorWriteMask      = kRgb;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &spriteAdd_),
              "vkCreateGraphicsPipelines(bbhdr.sprite.add)");

        // ALPHA-OVER fields: premultiplied over, colour and coverage alike, in
        // the alpha-over layer; in the additive layer only the attenuation,
        // dst * (1 - coverage), of the light the sprite now covers.
        cbas[0].blendEnable         = VK_TRUE;
        cbas[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cbas[0].colorBlendOp        = VK_BLEND_OP_ADD;
        cbas[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cbas[0].alphaBlendOp        = VK_BLEND_OP_ADD;
        cbas[0].colorWriteMask      = kRgba;
        cbas[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbas[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &spriteOver_),
              "vkCreateGraphicsPipelines(bbhdr.sprite.over)");

        vkDestroyShaderModule(d, vert, nullptr);
        vkDestroyShaderModule(d, frag, nullptr);
    }

    // ── The display-space composite pipelines ───────────────────────────────
    // Fullscreen triangles inside the caller's render-pass instance: its colour
    // format, depth format (declared, a pipeline must agree with the instance's
    // attachments, but not tested) and sample count. Alpha is masked out of the
    // write: the frame's alpha is not part of the composite.
    void BillboardHdrLayerPipelines::createCompositePipelines(VkFormat colorFormat,
                                                              VkSampleCountFlagBits overlaySamples) {
        const VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding bnd[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            bnd[i].binding         = i;
            bnd[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd[i].descriptorCount = 1;
            bnd[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings    = bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &compositeDsLayout_),
              "vkCreateDescriptorSetLayout(bbhdr.composite)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.size       = sizeof(BillboardHdrLayer::CompositePC);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &compositeDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &compositeLayout_),
              "vkCreatePipelineLayout(bbhdr.composite)");

        VkShaderModule vert = makeModule(d, kOverlayFullscreenVertSpv, sizeof(kOverlayFullscreenVertSpv),
                                         "vkCreateShaderModule(bbhdr.composite.vert)");
        VkShaderModule frag = makeModule(d, kParticleFieldHdrLayerFragSpv, sizeof(kParticleFieldHdrLayerFragSpv),
                                         "vkCreateShaderModule(bbhdr.composite.frag)");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName  = "main";

        const CommonState common(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        // The layers are one value per pixel, fetched at gl_FragCoord, so a
        // multisampled instance (the overlay under Canvas antialiasing) runs
        // this once per pixel and writes every covered sample alike.
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState cbas{};
        cbas.blendEnable         = VK_TRUE;
        cbas.colorBlendOp        = VK_BLEND_OP_ADD;
        cbas.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbas.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbas.alphaBlendOp        = VK_BLEND_OP_ADD;
        cbas.colorWriteMask      = kRgb;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &cbas;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext              = &prci;
        gpci.stageCount         = 2;
        gpci.pStages            = stages;
        common.apply(gpci);
        gpci.pMultisampleState  = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState   = &cb;
        gpci.layout             = compositeLayout_;

        // Two render-pass shapes: [0] the primary overlay pass (its D32 depth
        // attachment, its sample count), [1] a secondary view colour target
        // (no depth, one sample).
        for (uint32_t v = 0; v < 2; ++v) {
            prci.depthAttachmentFormat = v == 0 ? kDepthFormat : VK_FORMAT_UNDEFINED;
            ms.rasterizationSamples    = v == 0 ? overlaySamples : VK_SAMPLE_COUNT_1_BIT;
            // Over: premultiplied alpha-over.
            cbas.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &compositeOver_[v]),
                  "vkCreateGraphicsPipelines(bbhdr.composite.over)");
            // Add: the screen blend, dst + src * (1 - dst).
            cbas.srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            cbas.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &compositeAdd_[v]),
                  "vkCreateGraphicsPipelines(bbhdr.composite.add)");
        }

        vkDestroyShaderModule(d, vert, nullptr);
        vkDestroyShaderModule(d, frag, nullptr);
    }

    // ── The depth copy ──────────────────────────────────────────────────────
    // A depth-only fullscreen draw writing gl_FragDepth from a texelFetch of
    // the overlay's depth (billboard_layer_depth.frag). The test is ALWAYS so
    // the write happens (Vulkan gates depth writes on the test being enabled).
    void BillboardHdrLayerPipelines::createCopyPipelines() {
        const VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding bnd{};
        bnd.binding         = 0;
        bnd.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd.descriptorCount = 1;
        bnd.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings    = &bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &copyDsLayout_),
              "vkCreateDescriptorSetLayout(bbhdr.copy)");
        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &copyDsLayout_;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &copyLayout_), "vkCreatePipelineLayout(bbhdr.copy)");

        VkShaderModule vert   = makeModule(d, kOverlayFullscreenVertSpv, sizeof(kOverlayFullscreenVertSpv),
                                           "vkCreateShaderModule(bbhdr.copy.vert)");
        VkShaderModule frag   = makeModule(d, kBillboardLayerDepthFragSpv, sizeof(kBillboardLayerDepthFragSpv),
                                           "vkCreateShaderModule(bbhdr.copy.frag)");
        VkShaderModule fragMs = makeModule(d, kBillboardLayerDepthMsFragSpv, sizeof(kBillboardLayerDepthMsFragSpv),
                                           "vkCreateShaderModule(bbhdr.copy.frag.ms)");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].pName  = "main";

        const CommonState common(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.depthAttachmentFormat = kDepthFormat;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext              = &prci;
        gpci.stageCount         = 2;
        gpci.pStages            = stages;
        common.apply(gpci);
        gpci.pMultisampleState  = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState   = &cb;
        gpci.layout             = copyLayout_;

        stages[1].module = frag;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &copyPipe_),
              "vkCreateGraphicsPipelines(bbhdr.copy)");
        stages[1].module = fragMs;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &copyPipeMs_),
              "vkCreateGraphicsPipelines(bbhdr.copy.ms)");

        vkDestroyShaderModule(d, vert, nullptr);
        vkDestroyShaderModule(d, frag, nullptr);
        vkDestroyShaderModule(d, fragMs, nullptr);
    }

    void BillboardHdrLayer::createDescriptors() {
        VkDescriptorPoolSize size{};
        size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = 3 * framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 2 * framesInFlight_;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &size;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_), "vkCreateDescriptorPool(bbhdr)");

        auto alloc = [&](std::vector<VkDescriptorSet>& sets, VkDescriptorSetLayout layout) {
            std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, layout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = framesInFlight_;
            ai.pSetLayouts        = layouts.data();
            sets.resize(framesInFlight_);
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()), "vkAllocateDescriptorSets(bbhdr)");
        };
        alloc(compositeSets_, pipes_.compositeDsLayout_);
        alloc(copySets_, pipes_.copyDsLayout_);
        copySetSerials_.assign(framesInFlight_, uint64_t{0});
    }

    void BillboardHdrLayer::rewriteDescriptors() {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorImageInfo ii[2]{};
            ii[0] = {pipes_.sampler_, over_[f].view, VK_IMAGE_LAYOUT_GENERAL};
            ii[1] = {pipes_.sampler_, add_[f].view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w[2]{};
            for (uint32_t n = 0; n < 2; ++n) {
                w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet          = compositeSets_[f];
                w[n].dstBinding      = n;
                w[n].descriptorCount = 1;
                w[n].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[n].pImageInfo      = &ii[n];
            }
            vulkan::updateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
        }
        // The copy sets name the overlay's depth, which the caller reallocates
        // on the same events that bring us here: forget what they hold.
        std::fill(copySetSerials_.begin(), copySetSerials_.end(), uint64_t{0});
    }

    void BillboardHdrLayer::recordDepthCopy(VkCommandBuffer cb, uint32_t frame, VkImageView srcView,
                                            bool srcMultisampled) {
        // Rewritten only when the caller hands over a different view (by
        // serial: a reallocated view can keep its handle value), which happens
        // only when that attachment was reallocated with the device idle.
        const uint64_t serial = imageViewSerial(srcView);
        if (copySetSerials_[frame] != serial) {
            VkDescriptorImageInfo ii{pipes_.sampler_, srcView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = copySets_[frame];
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &ii;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
            copySetSerials_[frame] = serial;
        }

        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = depth_[frame].view;
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;// every pixel is written
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{};
        ri.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = {width_, height_};
        ri.layerCount        = 1;
        ri.pDepthAttachment  = &depthAtt;
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0.f, 0.f, float(width_), float(height_), 0.f, 1.f};
        vkCmdSetViewport(cb, 0, 1, &vp);
        VkRect2D sc{{0, 0}, {width_, height_}};
        vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, srcMultisampled ? pipes_.copyPipeMs_ : pipes_.copyPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes_.copyLayout_, 0, 1, &copySets_[frame], 0,
                                nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
    }

    void BillboardHdrLayer::depthWritten(VkCommandBuffer cb) {
        // Written as an attachment by the copy (and the stamp), tested by the
        // sprites next: same image, same layout, a different access.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void BillboardHdrLayer::beginSprites(VkCommandBuffer cb, uint32_t frame, VkImageView depthView,
                                         VkImageLayout depthLayout, const VkViewport& viewport,
                                         const VkRect2D& scissor) const {
        VkRenderingAttachmentInfo col[2]{};
        const VkImageView views[2] = {over_[frame].view, add_[frame].view};
        for (uint32_t i = 0; i < 2; ++i) {
            col[i].sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            col[i].imageView   = views[i];
            col[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            // CLEAR: each frame's layer holds that frame's sprites and nothing
            // else; transparent black is "no sprite" to every consumer.
            col[i].loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
            col[i].storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
            col[i].clearValue.color = {{0.f, 0.f, 0.f, 0.f}};
        }
        // Loaded, read-only, not stored.
        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = depthView;
        depthAtt.imageLayout = depthLayout;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_NONE;

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = {width_, height_};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 2;
        ri.pColorAttachments    = col;
        ri.pDepthAttachment     = &depthAtt;
        vkCmdBeginRendering(cb, &ri);
        vkCmdSetViewport(cb, 0, 1, &viewport);
        vkCmdSetScissor(cb, 0, 1, &scissor);
    }

    void BillboardHdrLayer::recordComposite(VkCommandBuffer cb, uint32_t frame, CompositePC pc) const {
        // Under the caller's viewport and scissor: the sprites were drawn under
        // the same ones, and the layer is at the caller's extent, so a fetch at
        // gl_FragCoord is the pixel a sprite covered.
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes_.compositeLayout_, 0, 1,
                                &compositeSets_[frame], 0, nullptr);
        // Over first, then the additive light on top: the additive layer
        // already carries the dimming of every alpha-over sprite drawn after
        // it, so this order reproduces the order the sprites were drawn in.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes_.compositeOver_[variant_]);
        pc.part = 0u;
        vkCmdPushConstants(cb, pipes_.compositeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes_.compositeAdd_[variant_]);
        pc.part = 1u;
        vkCmdPushConstants(cb, pipes_.compositeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cb, 3, 1, 0, 0);
    }

    void BillboardHdrLayer::declareSprites(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame,
                                           bool ownDepth) const {
        constexpr auto kGeneral = VK_IMAGE_LAYOUT_GENERAL;
        pass.use(g.importImage("bbhdr.over", over_[frame].image, VK_IMAGE_ASPECT_COLOR_BIT, 1, kGeneral),
                 rg::colorAttachment(kGeneral));
        pass.use(g.importImage("bbhdr.add", add_[frame].image, VK_IMAGE_ASPECT_COLOR_BIT, 1, kGeneral),
                 rg::colorAttachment(kGeneral));
        if (ownDepth) {
            pass.use(g.importImage("bbhdr.depth", depth_[frame].image, VK_IMAGE_ASPECT_DEPTH_BIT, 1, kGeneral),
                     rg::Access{VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                kGeneral, true});
        }
    }

    void BillboardHdrLayer::declareRead(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame,
                                        VkPipelineStageFlags2 stage) const {
        constexpr auto kGeneral = VK_IMAGE_LAYOUT_GENERAL;
        pass.use(g.importImage("bbhdr.over", over_[frame].image, VK_IMAGE_ASPECT_COLOR_BIT, 1, kGeneral),
                 rg::sampled(kGeneral, stage));
        pass.use(g.importImage("bbhdr.add", add_[frame].image, VK_IMAGE_ASPECT_COLOR_BIT, 1, kGeneral),
                 rg::sampled(kGeneral, stage));
    }

}// namespace threepp::vulkan
