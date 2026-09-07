#include "threepp/renderers/vulkan/BloomPass.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/bloom_up.comp.spv.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace threepp::vulkan {

    BloomPass::BloomPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        sceneHdr_.resize(framesInFlight_);
        pyr_.resize(framesInFlight_ * kMaxLevels);
        createPipelines();
        createDescriptorPool();
    }

    BloomPass::~BloomPass() {
        VkDevice d = ctx_.device();
        if (downPipe_)       vkDestroyPipeline(d, downPipe_, nullptr);
        if (upPipe_)         vkDestroyPipeline(d, upPipe_, nullptr);
        if (bloomPipeLayout_) vkDestroyPipelineLayout(d, bloomPipeLayout_, nullptr);
        if (bloomDsLayout_)  vkDestroyDescriptorSetLayout(d, bloomDsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        destroyImages();
    }

    void BloomPass::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : sceneHdr_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : pyr_)      destroyImage2D(ctx_.allocator(), d, img);
    }

    Image2D BloomPass::createStorageSampledImage(uint32_t w, uint32_t h, const char* label) {
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
        // TRANSFER_SRC for the determinism audit's sceneHdr readback
        // (VulkanRenderer::readSceneHdrDebug) — a usage bit is free.
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
              label);

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
              "vkCreateImageView(bloom)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view,  label);
        return out;
    }

    void BloomPass::transitionFreshImage(VkImage img) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(bloom)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(bloom)");

        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(bloom)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(bloom)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(bloom)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void BloomPass::createImages(uint32_t width, uint32_t height) {
        destroyImages();
        width_  = width;
        height_ = height;
        // Pyramid levels halve from half res down to ~1/64 of the frame; stop
        // before a level's short side drops under 8 texels (a 3×3 tent on a
        // smaller level is all edge-clamp, contributing only blocky smear).
        levels_ = 0;
        while (levels_ < kMaxLevels &&
               std::min(width >> (levels_ + 1u), height >> (levels_ + 1u)) >= 8u)
            ++levels_;
        if (levels_ == 0) levels_ = 1;// degenerate tiny extent: keep the half-res level
        for (auto& img : sceneHdr_)
            img = createStorageSampledImage(width_, height_, "vmaCreateImage(bloom.sceneHdr)");
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            for (uint32_t l = 0; l < levels_; ++l)
                pyr_[f * kMaxLevels + l] = createStorageSampledImage(
                        std::max(width_ >> (l + 1u), 1u), std::max(height_ >> (l + 1u), 1u),
                        "vmaCreateImage(bloom.pyr)");
    }

    static VkPipeline makeComputePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                                      const uint32_t* spv, size_t spvBytes,
                                      const char* label) {
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

    void BloomPass::createPipelines() {
        VkDevice d = ctx_.device();

        if (sampler_ == VK_NULL_HANDLE) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod       = 0.f;
            check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(bloom)");
        }

        // Bloom (down + up) layout: combined sampler @0, storage image @1.
        {
            VkDescriptorSetLayoutBinding bnd[2]{};
            bnd[0].binding = 0;
            bnd[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd[0].descriptorCount = 1;
            bnd[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bnd[1].binding = 1;
            bnd[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bnd[1].descriptorCount = 1;
            bnd[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 2;
            dlci.pBindings = bnd;
            check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &bloomDsLayout_),
                  "vkCreateDescriptorSetLayout(bloom)");

            VkPushConstantRange pc{};
            pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc.offset = 0;
            pc.size = 28;// down: 4×u32 + 2×float + u32 firstLevel ; up: 4×u32
            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &bloomDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pc;
            check(vkCreatePipelineLayout(d, &plci, nullptr, &bloomPipeLayout_),
                  "vkCreatePipelineLayout(bloom)");
        }

        downPipe_ = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomDownCompSpv,
                                    sizeof(kBloomDownCompSpv), "vkCreateComputePipelines(bloom_down)");
        upPipe_   = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomUpCompSpv,
                                    sizeof(kBloomUpCompSpv), "vkCreateComputePipelines(bloom_up)");
    }

    void BloomPass::createDescriptorPool() {
        // Sized for kMaxLevels regardless of the runtime level count (the pool
        // is created before the surface size is known).
        const uint32_t downSets = framesInFlight_ * kMaxLevels;
        const uint32_t upSets   = framesInFlight_ * kMaxLevels;// only levels-1 used
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = downSets + upSets;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = downSets + upSets;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = downSets + upSets;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(bloom)");

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
                  "vkAllocateDescriptorSets(bloom)");
        };
        alloc(downSets_, bloomDsLayout_, downSets);
        alloc(upSets_,   bloomDsLayout_, upSets);
    }

    void BloomPass::rewriteDescriptors() {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            auto sampled = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.sampler = sampler_;
                i.imageView = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };
            auto storage = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.imageView = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };

            auto writePair = [&](VkDescriptorSet ds, const VkDescriptorImageInfo* src,
                                 const VkDescriptorImageInfo* dst) {
                VkWriteDescriptorSet w[2]{};
                for (int n = 0; n < 2; ++n) {
                    w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[n].dstSet          = ds;
                    w[n].dstBinding      = static_cast<uint32_t>(n);
                    w[n].descriptorCount = 1;
                }
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[0].pImageInfo     = src;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[1].pImageInfo     = dst;
                vkUpdateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
            };

            // Down chain: sceneHdr → pyr[0], then pyr[l-1] → pyr[l].
            // Up chain: pyr[l+1] (tent-sampled) accumulates into pyr[l].
            for (uint32_t l = 0; l < levels_; ++l) {
                VkDescriptorImageInfo dIn  = sampled(l == 0 ? sceneHdr_[f].view
                                                            : pyr_[f * kMaxLevels + l - 1].view);
                VkDescriptorImageInfo dOut = storage(pyr_[f * kMaxLevels + l].view);
                writePair(downSets_[f * kMaxLevels + l], &dIn, &dOut);
                if (l + 1 < levels_) {
                    VkDescriptorImageInfo uIn  = sampled(pyr_[f * kMaxLevels + l + 1].view);
                    VkDescriptorImageInfo uOut = storage(pyr_[f * kMaxLevels + l].view);
                    writePair(upSets_[f * kMaxLevels + l], &uIn, &uOut);
                }
            }
        }
    }

    void BloomPass::recordPyramid(VkCommandBuffer cb, uint32_t frame,
                                  uint32_t width, uint32_t height,
                                  float bloomIntensity, float bloomThreshold,
                                  float bloomClamp) {
        if (bloomIntensity <= 0.0f) return;

        auto barrier = [&]() {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        };

        // The shade/resolve wrote sceneHdr (compute); make it visible.
        barrier();

        struct BloomPc { uint32_t srcW, srcH, dstW, dstH; float threshold, clampMax; uint32_t firstLevel; };
        auto levelW = [&](uint32_t l) { return std::max(width_  >> (l + 1u), 1u); };
        auto levelH = [&](uint32_t l) { return std::max(height_ >> (l + 1u), 1u); };

        // Progressive downsample: sceneHdr → pyr[0] (Karis + soft-knee
        // bright pass + per-tap clamp) → pyr[1] → … (plain 13-tap).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, downPipe_);
        for (uint32_t l = 0; l < levels_; ++l) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipeLayout_,
                                    0, 1, &downSets_[frame * kMaxLevels + l], 0, nullptr);
            BloomPc pc{l == 0 ? width : levelW(l - 1), l == 0 ? height : levelH(l - 1),
                       levelW(l), levelH(l),
                       bloomThreshold, l == 0 ? bloomClamp : 0.f, l == 0 ? 1u : 0u};
            vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (levelW(l) + 7u) / 8u, (levelH(l) + 7u) / 8u, 1);
            barrier();
        }

        // Progressive upsample walk-back: tent-filter each coarser level
        // and ADD it into the next-finer one, accumulating the whole
        // chain into pyr[0]. PostComposite divides bloomIntensity by the
        // level count, so total energy matches a single-level bloom.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, upPipe_);
        for (uint32_t l = levels_ - 1; l > 0; --l) {
            const uint32_t dst = l - 1;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipeLayout_,
                                    0, 1, &upSets_[frame * kMaxLevels + dst], 0, nullptr);
            BloomPc pc{levelW(l), levelH(l), levelW(dst), levelH(dst), 0.f, 0.f, 0u};
            vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (levelW(dst) + 7u) / 8u, (levelH(dst) + 7u) / 8u, 1);
            barrier();
        }
        // Last barrier above already covers pyramid write → PostComposite read.
    }

}// namespace threepp::vulkan
