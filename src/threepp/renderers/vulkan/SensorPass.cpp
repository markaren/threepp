#include "threepp/renderers/vulkan/SensorPass.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/sensor_image.comp.spv.h"

#include <cstring>

namespace threepp::vulkan {

    namespace {

        // Mirror of sensor_image.comp's push-constant block (std430: the vec4s
        // are 16-byte aligned, which the field order below keeps true).
        struct SensorPush {
            uint32_t width;
            uint32_t height;
            uint32_t flags;
            uint32_t lensModel;

            float normK[4];     // 16
            float radial[4];    // 32
            float tangential[2];// 48
            float overscan;     // 56
            uint32_t frameSeed; // 60

            float fullWell;     // 64
            float readNoise;    // 68
            float darkElectrons;// 72
            float prnu;         // 76
            float isoGain;      // 80
            float pad[3];       // 84 → 96
        };
        static_assert(sizeof(SensorPush) == 96, "push constants must match sensor_image.comp");

        constexpr uint32_t kFlagDistort = 1u;
        constexpr uint32_t kFlagNoise   = 2u;

    }// namespace

    SensorPass::SensorPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        snapshot_.resize(framesInFlight_);
        boundDst_.assign(framesInFlight_, VK_NULL_HANDLE);
        createPipeline();
        createDescriptorPool();
    }

    SensorPass::~SensorPass() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
        destroySnapshots();
    }

    void SensorPass::destroySnapshots() {
        VkDevice d = ctx_.device();
        for (auto& img : snapshot_) destroyImage2D(ctx_.allocator(), d, img);
        width_ = height_ = 0;
        boundDst_.assign(framesInFlight_, VK_NULL_HANDLE);
    }

    void SensorPass::createPipeline() {
        VkSamplerCreateInfo sci{};
        sci.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        // LINEAR: the warp lands between texels almost everywhere, and nearest
        // would alias the whole image. CLAMP_TO_EDGE is what makes an ideal ray
        // that falls outside the rendered field degrade to a stretched border
        // rather than wrapping — see setLensOverscan for the real remedy.
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_),
              "vkCreateSampler(sensorPass)");

        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding         = 1;
        b[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(sensorPass)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(SensorPush);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(sensorPass)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kSensorImageCompSpv);
        smci.pCode    = kSensorImageCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(sensorPass)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci,
                                       nullptr, &pipe_),
              "vkCreateComputePipelines(sensorPass)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);
    }

    void SensorPass::createDescriptorPool() {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(sensorPass)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = framesInFlight_;
        dsai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, sets_.data()),
              "vkAllocateDescriptorSets(sensorPass)");
    }

    void SensorPass::resize(uint32_t width, uint32_t height) {
        if (width == width_ && height == height_ && snapshot_[0].image != VK_NULL_HANDLE) return;
        if (width == 0u || height == 0u) return;
        destroySnapshots();
        width_  = width;
        height_ = height;

        VkDevice d = ctx_.device();
        for (auto& img : snapshot_) {
            img.width  = width;
            img.height = height;
            img.format = VK_FORMAT_B8G8R8A8_UNORM;// matches the swapchain

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = img.format;
            ici.extent        = {width, height, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(sensorPass.snapshot)");

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = img.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(d, &vci, nullptr, &img.view),
                  "vkCreateImageView(sensorPass.snapshot)");
            ctx_.setObjectName(img.image, "sensorPass.snapshot");
            ctx_.setObjectName(img.view, "sensorPass.snapshot");
        }

        // Snapshots start UNDEFINED; the per-frame copy transitions them, and
        // its barrier declares UNDEFINED as the old layout (contents are
        // overwritten wholesale), so no one-shot priming submit is needed.
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorImageInfo srcI{};
            srcI.sampler     = sampler_;
            srcI.imageView   = snapshot_[f].view;
            srcI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = sets_[f];
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &srcI;
            vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
        }
    }

    void SensorPass::record(VkCommandBuffer cb, uint32_t frame,
                            VkImage swapImage, VkImageView swapView,
                            uint32_t width, uint32_t height, const Params& p) {
        if (!p.active() || pipe_ == VK_NULL_HANDLE) return;
        resize(width, height);
        if (snapshot_[frame].image == VK_NULL_HANDLE) return;
        if (width != width_ || height != height_) return;

        // The destination is the swapchain image for THIS frame's acquired
        // index, which rotates independently of the frame-in-flight slot, so
        // refresh the storage binding whenever it differs from what this slot
        // last pointed at. (The prior frame using this slot has retired — the
        // caller's framesInFlight fence guarantees it — so rewriting here
        // cannot touch a descriptor an in-flight command buffer still reads.)
        if (boundDst_[frame] != swapView) {
            VkDescriptorImageInfo dstI{};
            dstI.imageView   = swapView;
            dstI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = sets_[frame];
            w.dstBinding      = 1;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo      = &dstI;
            vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
            boundDst_[frame] = swapView;
        }

        // ── Snapshot: swapchain → scratch ───────────────────────────────────
        // The swapchain arrives in GENERAL, written by the overlay pass
        // (colour attachment) and/or the resolve/RCAS (compute store).
        {
            VkImageMemoryBarrier2 bs[2]{};
            bs[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            bs[0].srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bs[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            bs[0].dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bs[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bs[0].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            bs[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[0].image = swapImage;
            bs[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            // The scratch is fully overwritten, so its previous contents are
            // discardable — UNDEFINED avoids a needless read-back dependency.
            bs[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            bs[1].srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bs[1].srcAccessMask = 0;
            bs[1].dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bs[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            bs[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            bs[1].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            bs[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[1].image = snapshot_[frame].image;
            bs[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo di{};
            di.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.imageMemoryBarrierCount = 2;
            di.pImageMemoryBarriers    = bs;
            vkCmdPipelineBarrier2(cb, &di);
        }

        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent         = {width, height, 1};
        vkCmdCopyImage(cb, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       snapshot_[frame].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);

        // ── Scratch → sampled, swapchain → GENERAL for the store ────────────
        {
            VkImageMemoryBarrier2 bs[2]{};
            bs[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            bs[0].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bs[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            bs[0].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bs[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            bs[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            bs[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[0].image = snapshot_[frame].image;
            bs[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            bs[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            bs[1].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bs[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bs[1].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bs[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            bs[1].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bs[1].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            bs[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bs[1].image = swapImage;
            bs[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo di{};
            di.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.imageMemoryBarrierCount = 2;
            di.pImageMemoryBarriers    = bs;
            vkCmdPipelineBarrier2(cb, &di);
        }

        SensorPush pc{};
        pc.width     = width;
        pc.height    = height;
        pc.flags     = (p.distortActive ? kFlagDistort : 0u) | (p.noiseActive ? kFlagNoise : 0u);
        pc.lensModel = p.lensModel;
        std::memcpy(pc.normK, p.normK, sizeof(pc.normK));
        std::memcpy(pc.radial, p.radial, sizeof(pc.radial));
        std::memcpy(pc.tangential, p.tangential, sizeof(pc.tangential));
        pc.overscan      = p.overscan;
        pc.frameSeed     = p.frameSeed;
        pc.fullWell      = p.fullWell;
        pc.readNoise     = p.readNoise;
        pc.darkElectrons = p.darkElectrons;
        pc.prnu          = p.prnu;
        pc.isoGain       = p.isoGain;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    }

}// namespace threepp::vulkan
