#include "threepp/renderers/vulkan/BillboardGlowPass.hpp"
#include "threepp/renderers/vulkan/TransientPool.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

// The depth reduction that gives the source pass something to test against:
// the shared fullscreen triangle, plus the two variants of the reduce itself.
#include "threepp/renderers/vulkan/shaders/billboard_glow_depth.frag.ms.spv.h"
#include "threepp/renderers/vulkan/shaders/billboard_glow_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_fullscreen.vert.spv.h"

#include <algorithm>

namespace threepp::vulkan {

    BillboardGlowPass::BillboardGlowPass(VulkanContext& ctx, VkCommandPool cmdPool,
                                         uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight),
          pyramid_(ctx, framesInFlight, kMaxLevels, "bbglow") {
        src_.resize(framesInFlight_);
        pyr_.resize(framesInFlight_ * kMaxLevels);
        depth_.resize(framesInFlight_);
        createPipelines();
        createDescriptors();
    }

    BillboardGlowPass::~BillboardGlowPass() {
        const VkDevice d = ctx_.device();
        if (reducePipe_)        vkDestroyPipeline(d, reducePipe_, nullptr);
        if (reducePipeMs_)      vkDestroyPipeline(d, reducePipeMs_, nullptr);
        if (reducePipeLayout_)  vkDestroyPipelineLayout(d, reducePipeLayout_, nullptr);
        if (compositeDsLayout_) vkDestroyDescriptorSetLayout(d, compositeDsLayout_, nullptr);
        if (reduceDsLayout_)    vkDestroyDescriptorSetLayout(d, reduceDsLayout_, nullptr);
        if (descPool_)          vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (depthSampler_)      vkDestroySampler(d, depthSampler_, nullptr);
        destroyImages();
    }

    void BillboardGlowPass::destroyImages() {
        const VkDevice d = ctx_.device();
        for (auto& img : src_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : pyr_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : depth_) destroyImage2D(ctx_.allocator(), d, img);
        // The reduce sets name an image this pass does not own, and the caller
        // reallocates that image on the same events that bring us here. Forget
        // what they hold; recordDepthReduce rewrites on the next mismatch.
        std::fill(reduceSetSerials_.begin(), reduceSetSerials_.end(), uint64_t{0});
        levels_ = 0;
        srcW_ = srcH_ = dispW_ = dispH_ = 0;
    }

    Image2D BillboardGlowPass::createImage(uint32_t w, uint32_t h,
                                           VkImageUsageFlags usage, const char* label, uint32_t slot) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = VK_FORMAT_R16G16B16A16_SFLOAT;

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

        check(createImageMaybePooled(ctx_.allocator(), pool_, transientGroup(TransientPhase::Tail, poolView_),
                                     transientSlot(slot), ici, &out.image, &out.alloc),
              label);

        // A pooled image is discarded at its first use in every frame.
        if (out.alloc != VK_NULL_HANDLE) transitionFreshImage(out.image, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vulkan::createImageView(bbglow)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view, label);
        return out;
    }

    // The depth companion of createImage. Same "GENERAL forever" contract — a
    // depth/stencil attachment is legal in GENERAL, and this one is written by
    // the reduce and read by the source pass in the same submit, so it never
    // needs a second layout.
    Image2D BillboardGlowPass::createDepthImage(uint32_t w, uint32_t h, const char* label, uint32_t slot) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = kDepthFormat;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        check(createImageMaybePooled(ctx_.allocator(), pool_, transientGroup(TransientPhase::Tail, poolView_),
                                     transientSlot(slot), ici, &out.image, &out.alloc),
              label);

        if (out.alloc != VK_NULL_HANDLE) transitionFreshImage(out.image, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vulkan::createImageView(bbglow.depth)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view, label);
        return out;
    }

    // Every image in this pass lives in GENERAL for its whole life and never
    // transitions again. GENERAL is a legal layout for a colour attachment, for
    // a storage image and for a sampled read, so one layout serves the render
    // pass, the pyramid and the composite — the same "one layout forever"
    // property ViewContext::colorTarget keeps, and for the same reason: a
    // per-frame layout dance across three consumers is a barrier bug waiting to
    // happen, and the pass is bandwidth-trivial either way.
    void BillboardGlowPass::transitionFreshImage(VkImage img, VkImageAspectFlags aspect) {
        submitOneShot(ctx_.device(), ctx_.graphicsQueue(), cmdPool_, "bbglow", [&](VkCommandBuffer cb) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = img;
            b.subresourceRange.aspectMask = aspect;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }

    bool BillboardGlowPass::ensureImages(uint32_t displayWidth, uint32_t displayHeight) {

        if (displayWidth == dispW_ && displayHeight == dispH_ && levels_ > 0) return true;
        if (displayWidth < 8 || displayHeight < 8) return false;

        destroyImages();
        dispW_ = displayWidth;
        dispH_ = displayHeight;
        srcW_  = std::max(displayWidth / 2u, 1u);
        srcH_  = std::max(displayHeight / 2u, 1u);

        levels_ = BloomPyramid::levelCount(srcW_, srcH_, kMaxLevels);

        // All three are rewritten every frame (the reduce, a CLEAR, the
        // pyramid) and dead after the overlay composite: transient scratch.
        for (uint32_t f = 0; f < src_.size(); ++f)
            src_[f] = createImage(srcW_, srcH_,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_STORAGE_BIT,
                                  "vmaCreateImage(bbglow.src)", f);
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            for (uint32_t l = 0; l < levels_; ++l) {
                const VkExtent2D e = BloomPyramid::levelExtent(srcW_, srcH_, l);
                pyr_[f * kMaxLevels + l] = createImage(e.width, e.height,
                                                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                       "vmaCreateImage(bbglow.pyr)", f);
            }
        for (uint32_t f = 0; f < depth_.size(); ++f)
            depth_[f] = createDepthImage(srcW_, srcH_, "vmaCreateImage(bbglow.depth)", f);
        rewriteDescriptors();
        return true;
    }

    void BillboardGlowPass::createPipelines() {
        const VkDevice d = ctx_.device();

        // The composite's set: one sampled image, read by the FRAGMENT stage
        // (the pyramid's sets are compute-only).
        {
            VkDescriptorSetLayoutBinding bnd{};
            bnd.binding         = 0;
            bnd.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd.descriptorCount = 1;
            bnd.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 1;
            dlci.pBindings    = &bnd;
            check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &compositeDsLayout_),
                  "vkCreateDescriptorSetLayout(bbglow.composite)");
        }

        createReducePipelines();
    }

    // ── The depth reduction ─────────────────────────────────────────────────
    // A fullscreen triangle that texelFetches the overlay pass's full-extent
    // depth and writes the min of the four texels each half-extent pixel covers
    // (see billboard_glow_depth.frag for why min). Both variants are built up
    // front: which one runs depends on the Canvas antialiasing setting, which
    // is fixed for the run but not known to this class, and the second pipeline
    // is a few hundred bytes.
    void BillboardGlowPass::createReducePipelines() {
        const VkDevice d = ctx_.device();

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
        check(vkCreateSampler(d, &sci, nullptr, &depthSampler_), "vkCreateSampler(bbglow.depth)");

        VkDescriptorSetLayoutBinding bnd{};
        bnd.binding         = 0;
        bnd.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd.descriptorCount = 1;
        bnd.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings    = &bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &reduceDsLayout_),
              "vkCreateDescriptorSetLayout(bbglow.reduce)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.size       = sizeof(uint32_t) * 2;// uvec2 srcExtent
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &reduceDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &reducePipeLayout_),
              "vkCreatePipelineLayout(bbglow.reduce)");

        VkShaderModule vert = createShaderModule(d, kOverlayFullscreenVertSpv, sizeof(kOverlayFullscreenVertSpv),
                                                 "vkCreateShaderModule(bbglow.reduce.vert)");
        VkShaderModule frag = createShaderModule(d, kBillboardGlowDepthFragSpv, sizeof(kBillboardGlowDepthFragSpv),
                                                 "vkCreateShaderModule(bbglow.reduce.frag)");
        VkShaderModule fragMs = createShaderModule(d, kBillboardGlowDepthMsFragSpv, sizeof(kBillboardGlowDepthMsFragSpv),
                                                   "vkCreateShaderModule(bbglow.reduce.frag.ms)");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].pName  = "main";

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

        VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates    = dynStates;

        // Depth-only: no colour attachment at all. The test must be ENABLED for
        // the write to happen (Vulkan gates depthWriteEnable on it), so it is
        // ALWAYS — every pixel of the target is overwritten by construction.
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_ALWAYS;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineRenderingCreateInfo prci{};
        prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.depthAttachmentFormat = kDepthFormat;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext               = &prci;
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
        gpci.layout              = reducePipeLayout_;

        stages[1].module = frag;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &reducePipe_),
              "vkCreateGraphicsPipelines(bbglow.reduce)");
        stages[1].module = fragMs;
        check(vkCreateGraphicsPipelines(d, ctx_.pipelineCache(), 1, &gpci, nullptr, &reducePipeMs_),
              "vkCreateGraphicsPipelines(bbglow.reduce.ms)");

        vkDestroyShaderModule(d, vert, nullptr);
        vkDestroyShaderModule(d, frag, nullptr);
        vkDestroyShaderModule(d, fragMs, nullptr);
    }

    // One allocation of every set, at construction (the pyramid's own, sized
    // for kMaxLevels, at its construction). Sets are REWRITTEN on resize and
    // never otherwise, which is what keeps this pass out of the VUID-03047 zone:
    // a descriptor here can only be updated at a point where the renderer has
    // already idled the device for the swapchain.
    void BillboardGlowPass::createDescriptors() {
        const uint32_t comp = framesInFlight_;
        const uint32_t red  = framesInFlight_;

        VkDescriptorPoolSize size{};
        size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = comp + red;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = comp + red;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &size;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(bbglow)");

        auto alloc = [&](std::vector<VkDescriptorSet>& sets, VkDescriptorSetLayout layout,
                         uint32_t count) {
            std::vector<VkDescriptorSetLayout> layouts(count, layout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = count;
            ai.pSetLayouts        = layouts.data();
            sets.resize(count);
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()),
                  "vkAllocateDescriptorSets(bbglow)");
        };
        alloc(compositeSets_, compositeDsLayout_, comp);
        alloc(reduceSets_, reduceDsLayout_, red);
        reduceSetSerials_.assign(red, uint64_t{0});
    }

    void BillboardGlowPass::rewriteDescriptors() {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            pyramid_.writeSets(f, src_[f].view, &pyr_[f * kMaxLevels], levels_);

            VkDescriptorImageInfo cIn{};
            cIn.sampler     = pyramid_.sampler();
            cIn.imageView   = pyr_[f * kMaxLevels].view;
            cIn.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = compositeSets_[f];
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &cIn;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        }
    }

    void BillboardGlowPass::recordDepthReduce(VkCommandBuffer cb, uint32_t frame,
                                              VkImageView srcView, bool srcMultisampled,
                                              VkExtent2D srcExtent) {
        if (levels_ == 0) return;

        const bool haveSrc = srcView != VK_NULL_HANDLE && srcExtent.width > 0 &&
                             srcExtent.height > 0 && reducePipe_ != VK_NULL_HANDLE;

        // The set names an attachment this pass does not own, so it is rewritten
        // whenever the caller hands over a different view — which only happens
        // when that attachment is reallocated, with the device idled. Different
        // by SERIAL: the reallocated attachment's view can keep its value.
        const uint64_t srcSerial = haveSrc ? imageViewSerial(srcView) : 0;
        if (haveSrc && reduceSetSerials_[frame] != srcSerial) {
            VkDescriptorImageInfo ii{};
            ii.sampler   = depthSampler_;
            ii.imageView = srcView;
            // The layout the overlay depth prepass leaves its target in; it
            // serves both the overlay pass's read-only attachment and this
            // sampled read.
            ii.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = reduceSets_[frame];
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &ii;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
            reduceSetSerials_[frame] = srcSerial;
        }

        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = depth_[frame].view;
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        // With a source, every pixel is written, so the prior contents are
        // irrelevant. Without one, the CLEAR is the whole pass: reverse-Z far is
        // 0, and GREATER_OR_EQUAL against 0 passes for everything.
        depthAtt.loadOp  = haveSrc ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.clearValue.depthStencil = {0.0f, 0u};

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.offset    = {0, 0};
        ri.renderArea.extent    = {srcW_, srcH_};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 0;
        ri.pDepthAttachment     = &depthAtt;
        vkCmdBeginRendering(cb, &ri);
        if (haveSrc) {
            VkViewport vp{0.f, 0.f, float(srcW_), float(srcH_), 0.f, 1.f};
            vkCmdSetViewport(cb, 0, 1, &vp);
            VkRect2D sc{{0, 0}, {srcW_, srcH_}};
            vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              srcMultisampled ? reducePipeMs_ : reducePipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, reducePipeLayout_, 0, 1,
                                    &reduceSets_[frame], 0, nullptr);
            const uint32_t pc[2] = {srcExtent.width, srcExtent.height};
            vkCmdPushConstants(cb, reducePipeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
            vkCmdDraw(cb, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cb);

        // Written as an attachment here, read as one by the source pass's depth
        // test; same image, same layout, different access.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void BillboardGlowPass::recordPyramid(VkCommandBuffer cb, uint32_t frame, float threshold) {

        if (levels_ == 0) return;

        // The billboard draw wrote `src` as a COLOUR ATTACHMENT; make it visible
        // to the first downsample's sampled read.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);

        // ── firstLevel, and why it is normally OFF here ─────────────────────
        // bloom_down's `firstLevel` switches on two things that exist to protect
        // a SCENE from its own highlights: the Karis 1/(1+luma) reweighting
        // (firefly suppression) and the soft-knee bright pass (only highlights
        // bloom). Both are actively wrong on this target, because this target
        // holds NOTHING BUT the highlights.
        //
        // Measured, on the campfire: with them on, a 3-px spark is Karis-
        // suppressed to about a fifth of its radiance by the 13-tap combine and
        // then bright-passed to nothing, and the whole feature came out at maxD
        // 7/255 — invisible, and inside the backend's own run-to-run noise. A
        // spark IS a firefly by construction; suppressing fireflies here is
        // suppressing the subject.
        //
        // So the default (glowThreshold == 0) runs the plain energy-conserving
        // 13-tap at every level. A field that DOES want only its brightest
        // particles to halo can still ask for a knee, and then the first level
        // behaves exactly as the scene pyramid's does.
        //
        // clampMax stays 0 (off): the scene pyramid clamps per tap to stop
        // sub-pixel specular flicker pulsing the halo radius, and this input
        // has no specular in it at all — it is a handful of authored emissive
        // sprites whose radiance is a deliberate number.
        BloomPyramid::Chain chain;
        chain.input      = {srcW_, srcH_};
        chain.base       = {srcW_, srcH_};
        chain.levels     = levels_;
        chain.threshold  = threshold;
        chain.clampMax   = 0.f;
        chain.firstLevel = (threshold > 0.f) ? 1u : 0u;
        pyramid_.record(cb, frame, chain);

        // Level 0 is next read by the composite draw's FRAGMENT stage; the
        // render graph orders that read (declareComposite).
    }

    void BillboardGlowPass::declare(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame,
                                    VkImage srcDepth) const {
        if (levels_ == 0) return;
        constexpr auto kGeneral = VK_IMAGE_LAYOUT_GENERAL;
        constexpr auto kDsRo    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        constexpr auto kTests   = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        if (srcDepth != VK_NULL_HANDLE) {
            pass.use(g.importImage("overlay.depth", srcDepth, VK_IMAGE_ASPECT_DEPTH_BIT, 1, kDsRo),
                     rg::sampled(kDsRo, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
        }
        // The reduce writes the half-extent depth, the glow draw tests against it.
        pass.use(g.importImage("glow.depth", depth_[frame].image, VK_IMAGE_ASPECT_DEPTH_BIT, 1, kGeneral),
                 rg::Access{kTests,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            kGeneral, true});
        // The glow draw renders the source, the first downsample samples it.
        pass.use(g.importImage("glow.src", src_[frame].image, VK_IMAGE_ASPECT_COLOR_BIT, 1, kGeneral),
                 rg::Access{VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            kGeneral, true});
        for (uint32_t l = 0; l < levels_; ++l) {
            pass.use(g.importImage("glow.pyr", pyr_[frame * kMaxLevels + l].image, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                                   kGeneral),
                     rg::Access{VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                kGeneral, true});
        }
    }

    void BillboardGlowPass::declareComposite(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame) const {
        if (levels_ == 0) return;
        pass.use(g.importImage("glow.pyr", pyr_[frame * kMaxLevels].image, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                               VK_IMAGE_LAYOUT_GENERAL),
                 rg::sampled(VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
    }

}// namespace threepp::vulkan
