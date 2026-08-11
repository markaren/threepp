#include "threepp/renderers/vulkan/BillboardGlowPass.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

// The SHARED bloom shaders. Reusing the SPIR-V rather than writing a second
// pyramid is the whole point of F4 item 1's "run the EXISTING bloom_down/up on
// that target alone": the push block, the descriptor shape and the Karis /
// soft-knee / 13-tap behaviour are all identical, only the images differ.
#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/bloom_up.comp.spv.h"

#include <algorithm>

namespace threepp::vulkan {

    BillboardGlowPass::BillboardGlowPass(VulkanContext& ctx, VkCommandPool cmdPool,
                                         uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        src_.resize(framesInFlight_);
        pyr_.resize(framesInFlight_ * kMaxLevels);
        createPipelines();
        createDescriptors();
    }

    BillboardGlowPass::~BillboardGlowPass() {
        const VkDevice d = ctx_.device();
        if (downPipe_)          vkDestroyPipeline(d, downPipe_, nullptr);
        if (upPipe_)            vkDestroyPipeline(d, upPipe_, nullptr);
        if (bloomPipeLayout_)   vkDestroyPipelineLayout(d, bloomPipeLayout_, nullptr);
        if (bloomDsLayout_)     vkDestroyDescriptorSetLayout(d, bloomDsLayout_, nullptr);
        if (compositeDsLayout_) vkDestroyDescriptorSetLayout(d, compositeDsLayout_, nullptr);
        if (descPool_)          vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)           vkDestroySampler(d, sampler_, nullptr);
        destroyImages();
    }

    void BillboardGlowPass::destroyImages() {
        const VkDevice d = ctx_.device();
        for (auto& img : src_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : pyr_) destroyImage2D(ctx_.allocator(), d, img);
        levels_ = 0;
        srcW_ = srcH_ = dispW_ = dispH_ = 0;
    }

    Image2D BillboardGlowPass::createImage(uint32_t w, uint32_t h,
                                           VkImageUsageFlags usage, const char* label) {
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

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr), label);

        transitionFreshImage(out.image);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vkCreateImageView(bbglow)");
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
    void BillboardGlowPass::transitionFreshImage(VkImage img) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(bbglow)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(bbglow)");

        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(bbglow)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(bbglow)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(bbglow)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    bool BillboardGlowPass::ensureImages(uint32_t displayWidth, uint32_t displayHeight) {

        if (displayWidth == dispW_ && displayHeight == dispH_ && levels_ > 0) return true;
        if (displayWidth < 8 || displayHeight < 8) return false;

        destroyImages();
        dispW_ = displayWidth;
        dispH_ = displayHeight;
        srcW_  = std::max(displayWidth / 2u, 1u);
        srcH_  = std::max(displayHeight / 2u, 1u);

        levels_ = 0;
        while (levels_ < kMaxLevels &&
               std::min(srcW_ >> (levels_ + 1u), srcH_ >> (levels_ + 1u)) >= 8u)
            ++levels_;
        if (levels_ == 0) levels_ = 1;

        for (auto& img : src_)
            img = createImage(srcW_, srcH_,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                      VK_IMAGE_USAGE_STORAGE_BIT,
                              "vmaCreateImage(bbglow.src)");
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            for (uint32_t l = 0; l < levels_; ++l)
                pyr_[f * kMaxLevels + l] =
                        createImage(std::max(srcW_ >> (l + 1u), 1u), std::max(srcH_ >> (l + 1u), 1u),
                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    "vmaCreateImage(bbglow.pyr)");
        rewriteDescriptors();
        return true;
    }

    static VkPipeline makeComputePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                                      const uint32_t* spv, size_t spvBytes, const char* label) {
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spvBytes;
        smci.pCode    = spv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), label);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        check(vkCreateComputePipelines(d, cache, 1, &cpci, nullptr, &pipe), label);
        vkDestroyShaderModule(d, mod, nullptr);
        return pipe;
    }

    void BillboardGlowPass::createPipelines() {
        const VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(bbglow)");

        // (sampler @0, storage @1) + 28 B push — byte for byte BloomPass's, and
        // it has to be: the SPIR-V is the same module.
        {
            VkDescriptorSetLayoutBinding bnd[2]{};
            bnd[0].binding         = 0;
            bnd[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd[0].descriptorCount = 1;
            bnd[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bnd[1].binding         = 1;
            bnd[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bnd[1].descriptorCount = 1;
            bnd[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 2;
            dlci.pBindings    = bnd;
            check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &bloomDsLayout_),
                  "vkCreateDescriptorSetLayout(bbglow.bloom)");

            VkPushConstantRange pc{};
            pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc.size       = 28;
            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &bloomDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pc;
            check(vkCreatePipelineLayout(d, &plci, nullptr, &bloomPipeLayout_),
                  "vkCreatePipelineLayout(bbglow.bloom)");
        }

        // The composite's set: one sampled image, read by the FRAGMENT stage
        // (BloomPass's sets are compute-only — this is the one shape difference).
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

        downPipe_ = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomDownCompSpv,
                                    sizeof(kBloomDownCompSpv), "vkCreateComputePipelines(bbglow_down)");
        upPipe_   = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomUpCompSpv,
                                    sizeof(kBloomUpCompSpv), "vkCreateComputePipelines(bbglow_up)");
    }

    // One allocation of every set, at construction, sized for kMaxLevels
    // regardless of the runtime chain depth. Sets are REWRITTEN on resize and
    // never otherwise, which is what keeps this pass out of the VUID-03047 zone:
    // a descriptor here can only be updated at a point where the renderer has
    // already idled the device for the swapchain.
    void BillboardGlowPass::createDescriptors() {
        const uint32_t down = framesInFlight_ * kMaxLevels;
        const uint32_t up   = framesInFlight_ * kMaxLevels;
        const uint32_t comp = framesInFlight_;

        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = down + up + comp;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = down + up;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = down + up + comp;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
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
        alloc(downSets_, bloomDsLayout_, down);
        alloc(upSets_, bloomDsLayout_, up);
        alloc(compositeSets_, compositeDsLayout_, comp);
    }

    void BillboardGlowPass::rewriteDescriptors() {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            auto sampled = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.sampler     = sampler_;
                i.imageView   = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };
            auto storage = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.imageView   = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };
            auto writePair = [&](VkDescriptorSet ds, const VkDescriptorImageInfo* s,
                                 const VkDescriptorImageInfo* d) {
                VkWriteDescriptorSet w[2]{};
                for (int n = 0; n < 2; ++n) {
                    w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[n].dstSet          = ds;
                    w[n].dstBinding      = static_cast<uint32_t>(n);
                    w[n].descriptorCount = 1;
                }
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[0].pImageInfo     = s;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[1].pImageInfo     = d;
                vkUpdateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
            };

            for (uint32_t l = 0; l < levels_; ++l) {
                VkDescriptorImageInfo dIn = sampled(l == 0 ? src_[f].view
                                                           : pyr_[f * kMaxLevels + l - 1].view);
                VkDescriptorImageInfo dOut = storage(pyr_[f * kMaxLevels + l].view);
                writePair(downSets_[f * kMaxLevels + l], &dIn, &dOut);
                if (l + 1 < levels_) {
                    VkDescriptorImageInfo uIn  = sampled(pyr_[f * kMaxLevels + l + 1].view);
                    VkDescriptorImageInfo uOut = storage(pyr_[f * kMaxLevels + l].view);
                    writePair(upSets_[f * kMaxLevels + l], &uIn, &uOut);
                }
            }

            VkDescriptorImageInfo cIn = sampled(pyr_[f * kMaxLevels].view);
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = compositeSets_[f];
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &cIn;
            vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        }
    }

    void BillboardGlowPass::recordPyramid(VkCommandBuffer cb, uint32_t frame, float threshold) {

        if (levels_ == 0) return;

        auto barrier = [&](VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                           VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = srcStage;
            mb.srcAccessMask = srcAccess;
            mb.dstStageMask  = dstStage;
            mb.dstAccessMask = dstAccess;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        };
        auto computeBarrier = [&] {
            barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        };

        // The billboard draw wrote `src` as a COLOUR ATTACHMENT; make it visible
        // to the first downsample's sampled read.
        barrier(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        struct BloomPc {
            uint32_t srcW, srcH, dstW, dstH;
            float threshold, clampMax;
            uint32_t firstLevel;
        };
        auto lw = [&](uint32_t l) { return std::max(srcW_ >> (l + 1u), 1u); };
        auto lh = [&](uint32_t l) { return std::max(srcH_ >> (l + 1u), 1u); };

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
        const uint32_t firstLevel = (threshold > 0.f) ? 1u : 0u;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, downPipe_);
        for (uint32_t l = 0; l < levels_; ++l) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipeLayout_, 0, 1,
                                    &downSets_[frame * kMaxLevels + l], 0, nullptr);
            // clampMax stays 0 (off): the scene pyramid clamps per tap to stop
            // sub-pixel specular flicker pulsing the halo radius, and this input
            // has no specular in it at all — it is a handful of authored
            // emissive sprites whose radiance is a deliberate number.
            BloomPc pc{l == 0 ? srcW_ : lw(l - 1), l == 0 ? srcH_ : lh(l - 1),
                       lw(l), lh(l), threshold, 0.f, l == 0 ? firstLevel : 0u};
            vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (lw(l) + 7u) / 8u, (lh(l) + 7u) / 8u, 1);
            computeBarrier();
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, upPipe_);
        for (uint32_t l = levels_ - 1; l > 0; --l) {
            const uint32_t dst = l - 1;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipeLayout_, 0, 1,
                                    &upSets_[frame * kMaxLevels + dst], 0, nullptr);
            BloomPc pc{lw(l), lh(l), lw(dst), lh(dst), 0.f, 0.f, 0u};
            vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (lw(dst) + 7u) / 8u, (lh(dst) + 7u) / 8u, 1);
            computeBarrier();
        }

        // Level 0 is next read by the composite draw's FRAGMENT stage, which the
        // compute-to-compute barriers above do not cover.
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}// namespace threepp::vulkan
