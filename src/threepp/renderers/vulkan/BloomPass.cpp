#include "threepp/renderers/vulkan/BloomPass.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/bloom_blur.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/composite.comp.spv.h"

#include <array>
#include <cstring>

namespace threepp::vulkan {

    BloomPass::BloomPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        sceneHdr_.resize(framesInFlight_);
        bloomA_.resize(framesInFlight_);
        bloomB_.resize(framesInFlight_);
        createPipelines();
        createDescriptorPool();
    }

    BloomPass::~BloomPass() {
        VkDevice d = ctx_.device();
        if (downPipe_)       vkDestroyPipeline(d, downPipe_, nullptr);
        if (blurPipe_)       vkDestroyPipeline(d, blurPipe_, nullptr);
        if (compPipe_)       vkDestroyPipeline(d, compPipe_, nullptr);
        if (bloomPipeLayout_) vkDestroyPipelineLayout(d, bloomPipeLayout_, nullptr);
        if (compPipeLayout_)  vkDestroyPipelineLayout(d, compPipeLayout_, nullptr);
        if (bloomDsLayout_)  vkDestroyDescriptorSetLayout(d, bloomDsLayout_, nullptr);
        if (compDsLayout_)   vkDestroyDescriptorSetLayout(d, compDsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        destroyImages();
    }

    void BloomPass::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : sceneHdr_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : bloomA_)   destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : bloomB_)   destroyImage2D(ctx_.allocator(), d, img);
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
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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
        halfW_  = (width  + 1u) / 2u;
        halfH_  = (height + 1u) / 2u;
        for (auto& img : sceneHdr_)
            img = createStorageSampledImage(width_, height_, "vmaCreateImage(bloom.sceneHdr)");
        for (auto& img : bloomA_)
            img = createStorageSampledImage(halfW_, halfH_, "vmaCreateImage(bloom.A)");
        for (auto& img : bloomB_)
            img = createStorageSampledImage(halfW_, halfH_, "vmaCreateImage(bloom.B)");
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

        // Bloom (down + blur) layout: combined sampler @0, storage image @1.
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
            pc.size = 24;// down: 4×u32 + 2×float ; blur: 2×u32 + 2×float
            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &bloomDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pc;
            check(vkCreatePipelineLayout(d, &plci, nullptr, &bloomPipeLayout_),
                  "vkCreatePipelineLayout(bloom)");
        }

        // Composite layout: combined samplers @0,1 ; storage images @2,3.
        {
            VkDescriptorSetLayoutBinding bnd[4]{};
            bnd[0].binding = 0;
            bnd[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd[1].binding = 1;
            bnd[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bnd[2].binding = 2;
            bnd[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bnd[3].binding = 3;
            bnd[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            for (auto& b : bnd) { b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 4;
            dlci.pBindings = bnd;
            check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &compDsLayout_),
                  "vkCreateDescriptorSetLayout(composite)");

            VkPushConstantRange pc{};
            pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc.offset = 0;
            pc.size = 24;// 6×u32
            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &compDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pc;
            check(vkCreatePipelineLayout(d, &plci, nullptr, &compPipeLayout_),
                  "vkCreatePipelineLayout(composite)");
        }

        downPipe_ = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomDownCompSpv,
                                    sizeof(kBloomDownCompSpv), "vkCreateComputePipelines(bloom_down)");
        blurPipe_ = makeComputePipe(d, ctx_.pipelineCache(), bloomPipeLayout_, kBloomBlurCompSpv,
                                    sizeof(kBloomBlurCompSpv), "vkCreateComputePipelines(bloom_blur)");
        compPipe_ = makeComputePipe(d, ctx_.pipelineCache(), compPipeLayout_, kCompositeCompSpv,
                                    sizeof(kCompositeCompSpv), "vkCreateComputePipelines(composite)");
    }

    void BloomPass::createDescriptorPool() {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * 5;// down1 + blurH1 + blurV1 + comp2
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_ * 5;// down1 + blurH1 + blurV1 + comp2

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_ * 4;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(bloom)");

        auto alloc = [&](std::vector<VkDescriptorSet>& sets, VkDescriptorSetLayout layout) {
            std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, layout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = framesInFlight_;
            ai.pSetLayouts        = layouts.data();
            sets.resize(framesInFlight_);
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()),
                  "vkAllocateDescriptorSets(bloom)");
        };
        alloc(downSets_,  bloomDsLayout_);
        alloc(blurHSets_, bloomDsLayout_);
        alloc(blurVSets_, bloomDsLayout_);
        alloc(compSets_,  compDsLayout_);
    }

    void BloomPass::rewriteDescriptors(const DescriptorWriteInputs& in) {
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

            // down: sceneHdr (sampled) -> bloomA (storage)
            VkDescriptorImageInfo dIn  = sampled(sceneHdr_[f].view);
            VkDescriptorImageInfo dOut = storage(bloomA_[f].view);
            // blurH: bloomA (sampled) -> bloomB (storage)
            VkDescriptorImageInfo hIn  = sampled(bloomA_[f].view);
            VkDescriptorImageInfo hOut = storage(bloomB_[f].view);
            // blurV: bloomB (sampled) -> bloomA (storage)
            VkDescriptorImageInfo vIn  = sampled(bloomB_[f].view);
            VkDescriptorImageInfo vOut = storage(bloomA_[f].view);
            // composite: sceneHdr (sampled), bloomA (sampled), gbuf (storage) -> taaInput (storage)
            VkDescriptorImageInfo cScene = sampled(sceneHdr_[f].view);
            VkDescriptorImageInfo cBloom = sampled(bloomA_[f].view);
            VkDescriptorImageInfo cGbuf  = storage(in.gbufPerFrame[f]);
            VkDescriptorImageInfo cOut   = storage(in.taaInputPerFrame[f]);

            VkWriteDescriptorSet w[10]{};
            auto setw = [&](int n, VkDescriptorSet ds, uint32_t bind,
                            VkDescriptorType t, const VkDescriptorImageInfo* info) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet = ds;
                w[n].dstBinding = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType = t;
                w[n].pImageInfo = info;
            };
            setw(0, downSets_[f],  0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dIn);
            setw(1, downSets_[f],  1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &dOut);
            setw(2, blurHSets_[f], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hIn);
            setw(3, blurHSets_[f], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &hOut);
            setw(4, blurVSets_[f], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &vIn);
            setw(5, blurVSets_[f], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &vOut);
            setw(6, compSets_[f],  0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cScene);
            setw(7, compSets_[f],  1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cBloom);
            setw(8, compSets_[f],  2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cGbuf);
            setw(9, compSets_[f],  3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cOut);
            vkUpdateDescriptorSets(ctx_.device(), 10, w, 0, nullptr);
        }
    }

    void BloomPass::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                   uint32_t width, uint32_t height,
                                   uint32_t toneMapping, uint32_t exposureBits,
                                   bool bgIsSolidColor, float bloomIntensity,
                                   float bloomThreshold, float bloomClamp) {
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

        // denoise.comp (resolve) wrote sceneHdr (compute); make it visible.
        barrier();

        if (bloomIntensity > 0.0f) {
            const uint32_t ghx = (halfW_ + 7u) / 8u;
            const uint32_t ghy = (halfH_ + 7u) / 8u;

            struct DownPc { uint32_t srcW, srcH, dstW, dstH; float threshold, clampMax; };
            struct BlurPc { uint32_t w, h; float dx, dy; };

            // Downsample sceneHdr -> bloomA (half res, Karis + soft-knee bright pass).
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, downPipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    bloomPipeLayout_, 0, 1, &downSets_[frame], 0, nullptr);
            DownPc dpc{width, height, halfW_, halfH_, bloomThreshold, bloomClamp};
            vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpc), &dpc);
            vkCmdDispatch(cb, ghx, ghy, 1);
            barrier();

            // Two separable-Gaussian iterations (H, V) for a wide soft glow.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipe_);
            BlurPc hpc{halfW_, halfH_, 1.0f, 0.0f};
            BlurPc vpc{halfW_, halfH_, 0.0f, 1.0f};
            for (int iter = 0; iter < 2; ++iter) {
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        bloomPipeLayout_, 0, 1, &blurHSets_[frame], 0, nullptr);
                vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hpc), &hpc);
                vkCmdDispatch(cb, ghx, ghy, 1);
                barrier();
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        bloomPipeLayout_, 0, 1, &blurVSets_[frame], 0, nullptr);
                vkCmdPushConstants(cb, bloomPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vpc), &vpc);
                vkCmdDispatch(cb, ghx, ghy, 1);
                barrier();
            }
        }

        // Composite: sceneHdr (+ bloomA) -> tone map -> sRGB -> TAA input.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, compPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compPipeLayout_, 0, 1, &compSets_[frame], 0, nullptr);
        uint32_t intensityBits;
        std::memcpy(&intensityBits, &bloomIntensity, sizeof(intensityBits));
        const uint32_t cpc[6] = {toneMapping, exposureBits,
                                 bgIsSolidColor ? 1u : 0u, intensityBits,
                                 width, height};
        vkCmdPushConstants(cb, compPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), cpc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
        // Composite wrote the TAA input; TaaResolve::recordResolve's pre-barrier
        // makes it visible to the temporal resolve.
    }

}// namespace threepp::vulkan
