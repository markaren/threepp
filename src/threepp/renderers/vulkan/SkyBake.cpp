#include "threepp/renderers/vulkan/SkyBake.hpp"

#include "threepp/renderers/vulkan/EnvPrefilter.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/sky_preetham.comp.spv.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace threepp::vulkan {

    namespace {

        // sky_preetham.comp's push constants.
        struct SkyPc {
            float sunDir[4];// xyz unit toward the sun; w = sunPosition.y as given
            float up[4];    // xyz; w = turbidity
            float coeffs[4];// rayleigh, mieCoefficient, mieDirectionalG, withDisc
        };

    }// namespace

    SkyBake::SkyBake(VulkanContext& ctx, EnvPrefilter& prefilter)
        : ctx_(ctx), prefilter_(prefilter) {
        createPipeline();
    }

    SkyBake::~SkyBake() {
        VkDevice d = ctx_.device();
        for (VkImageView v : mipViews_) vkDestroyImageView(d, v, nullptr);
        mipViews_.clear();
        destroyImage2D(ctx_.allocator(), d, image_);
        if (descPool_) vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipeline_) vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_) vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
    }

    void SkyBake::createPipeline() {
        // One storage image: mip 0 of the env image.
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings    = &b;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(skyBake)");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(SkyPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(skyBake)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kSkyPreethamCompSpv);
        smci.pCode    = kSkyPreethamCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(skyBake)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(skyBake)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // The bake's set and one prefilter set per mip above 0, allocated once
        // with the image and kept: the views they name never change, so no
        // frame in flight can be left holding a stale one.
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[0].descriptorCount = EnvPrefilter::kMaxEnvMips;
        ps[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[1].descriptorCount = EnvPrefilter::kMaxEnvMips;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = EnvPrefilter::kMaxEnvMips;
        pci.poolSizeCount = static_cast<uint32_t>(ps.size());
        pci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &pci, nullptr, &descPool_),
              "vkCreateDescriptorPool(skyBake)");
    }

    void SkyBake::createImage() {
        image_.width  = kWidth;
        image_.height = kHeight;
        image_.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        const uint32_t fullMips = static_cast<uint32_t>(
                std::floor(std::log2(static_cast<float>(std::max(kWidth, kHeight))))) + 1u;
        image_.mipLevels = std::min(fullMips, EnvPrefilter::kMaxEnvMips);

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = image_.format;
        ici.extent        = {kWidth, kHeight, 1};
        ici.mipLevels     = image_.mipLevels;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        applyConcurrentSharing(ici);
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &image_.image, &image_.alloc, nullptr),
              "vmaCreateImage(skyBake)");
        ctx_.setObjectName(image_.image, "envPmrem (Sky bake)");

        // The full-chain view and the sampler every env lookup reads through —
        // the same as EnvPrefilter::buildPmrem gives an HDRI.
        {
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = image_.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = image_.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.baseMipLevel = 0;
            vci.subresourceRange.levelCount = image_.mipLevels;
            vci.subresourceRange.layerCount = 1;
            check(createImageView(ctx_.device(), &vci, nullptr, &image_.view),
                  "createImageView(skyBake)");
            ctx_.setObjectName(image_.view, "envPmrem (Sky bake)");
        }
        {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod       = 0.0f;
            sci.maxLod       = static_cast<float>(image_.mipLevels - 1);
            check(vkCreateSampler(ctx_.device(), &sci, nullptr, &image_.sampler),
                  "vkCreateSampler(skyBake)");
        }

        // One single-level view per mip: mip 0 for the bake and as the
        // prefilter's source, the rest for the prefilter's targets.
        mipViews_.reserve(image_.mipLevels);
        for (uint32_t mip = 0; mip < image_.mipLevels; ++mip) {
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = image_.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = image_.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.baseMipLevel = mip;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            VkImageView v = VK_NULL_HANDLE;
            check(createImageView(ctx_.device(), &vci, nullptr, &v), "createImageView(skyBake mip)");
            mipViews_.push_back(v);
        }

        // The sets, once.
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &dsLayout_;
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, &bakeSet_), "vkAllocateDescriptorSets(skyBake)");
            VkDescriptorImageInfo dst{};
            dst.imageView   = mipViews_[0];
            dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = bakeSet_;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo      = &dst;
            updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        }
        mipSets_.reserve(image_.mipLevels);
        for (uint32_t mip = 1; mip < image_.mipLevels; ++mip) {
            const VkDescriptorSetLayout layout = prefilter_.setLayout();
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, &set), "vkAllocateDescriptorSets(skyBake prefilter)");
            prefilter_.writeSet(set, mipViews_[0], mipViews_[mip]);
            mipSets_.push_back(set);
        }
        firstUse_ = true;
    }

    const Image2D& SkyBake::image() {
        if (image_.image == VK_NULL_HANDLE) createImage();
        return image_;
    }

    void SkyBake::dispatchBake(VkCommandBuffer cb, const PreethamSkyParams& p, bool withDisc) const {
        SkyPc pc{};
        pc.sunDir[0] = p.sunDir[0];
        pc.sunDir[1] = p.sunDir[1];
        pc.sunDir[2] = p.sunDir[2];
        pc.sunDir[3] = p.sunPosY;
        pc.up[0]     = p.up[0];
        pc.up[1]     = p.up[1];
        pc.up[2]     = p.up[2];
        pc.up[3]     = p.turbidity;
        pc.coeffs[0] = p.rayleigh;
        pc.coeffs[1] = p.mieCoefficient;
        pc.coeffs[2] = p.mieDirectionalG;
        pc.coeffs[3] = withDisc ? 1.0f : 0.0f;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &bakeSet_, 0, nullptr);
        vkCmdDispatch(cb, (kWidth + 7u) / 8u, (kHeight + 7u) / 8u, 1u);
    }

    void SkyBake::record(VkCommandBuffer cb, const PreethamSkyParams& p, bool sunFreeMips) {
        const Image2D& img = image();
        const auto barrier = [&](uint32_t baseMip, uint32_t mipCount,
                                 VkImageLayout from, VkImageLayout to,
                                 VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
            VkImageMemoryBarrier br{};
            br.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.oldLayout           = from;
            br.newLayout           = to;
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.image               = img.image;
            br.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
            br.subresourceRange.baseMipLevel = baseMip;
            br.subresourceRange.levelCount   = mipCount;
            br.subresourceRange.layerCount   = 1;
            br.srcAccessMask = srcAccess;
            br.dstAccessMask = dstAccess;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &br);
        };

        // Every mip to GENERAL: from UNDEFINED the first time, after that from
        // the layout the frame samples it in, once the previous frame's
        // readers (any stage) are done with it.
        barrier(0, img.mipLevels,
                firstUse_ ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                firstUse_ ? 0u : VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                firstUse_ ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        firstUse_ = false;

        // Mip 0: the source the mips integrate.
        dispatchBake(cb, p, /*withDisc=*/!sunFreeMips);
        barrier(0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        prefilter_.recordMips(cb, kWidth, kHeight, img.mipLevels, mipSets_.data(), kPrefilterSamples);

        if (sunFreeMips) {
            // Mip 0 again, with the disc, once the prefilter has read it.
            barrier(0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            dispatchBake(cb, p, /*withDisc=*/true);
        }

        // Every mip to the layout the frame's readers expect.
        barrier(0, img.mipLevels, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }

}// namespace threepp::vulkan
