#include "threepp/renderers/vulkan/SensorPass.hpp"
#include "threepp/renderers/vulkan/TransientPool.hpp"

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
        constexpr uint32_t kFlagRig     = 4u;

        // Mirror of sensor_image.comp's RigUbo (std140).
        struct RigUboData {
            uint32_t info[4];       // x = faceMask
            float    margin[4];     // xy = the primary's margin in uv
            float    faceRect[4][4];// (u0, v0, 1/(u1-u0), 1/(v1-v0))
        };
        static_assert(sizeof(RigUboData) == 96, "must match sensor_image.comp's RigUbo");

    }// namespace

    SensorPass::SensorPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        snapshot_.resize(framesInFlight_);
        boundDst_.assign(framesInFlight_, 0);
        boundFaces_.assign(framesInFlight_, {});
        createPipeline();
        createDescriptorPool();
        rigUbo_.resize(framesInFlight_);
        rigUboMapped_.assign(framesInFlight_, nullptr);
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            rigUbo_[f] = createBuffer(ctx_.allocator(), ctx_.device(), sizeof(RigUboData),
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), rigUbo_[f].alloc, &info);
            rigUboMapped_[f] = info.pMappedData;
            std::memset(rigUboMapped_[f], 0, sizeof(RigUboData));
            VkDescriptorBufferInfo bi{rigUbo_[f].handle, 0, sizeof(RigUboData)};
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = sets_[f];
            w.dstBinding      = 3;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo     = &bi;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        }
    }

    SensorPass::~SensorPass() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
        destroySnapshots();
        for (auto& b : rigUbo_) destroyBuffer(ctx_.allocator(), b);
    }

    void SensorPass::destroySnapshots() {
        VkDevice d = ctx_.device();
        for (auto& img : snapshot_) destroyImage2D(ctx_.allocator(), d, img);
        width_ = height_ = 0;
        boundDst_.assign(framesInFlight_, 0);
        boundFaces_.assign(framesInFlight_, {});
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

        VkDescriptorSetLayoutBinding b[4]{};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding         = 1;
        b[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        // The fisheye rig's side views and its layout block.
        b[2].binding         = 2;
        b[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[2].descriptorCount = lensrig::kFaceCount;
        b[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[3].binding         = 3;
        b[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[3].descriptorCount = 1;
        b[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 4;
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
        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * (1u + lensrig::kFaceCount);
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_;
        sizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[2].descriptorCount = framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 3;
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
        for (uint32_t f = 0; f < snapshot_.size(); ++f) {
            auto& img  = snapshot_[f];
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
            // Copied into and read back within the stage: transient scratch
            // (the pool's Tail group).
            check(createImageMaybePooled(ctx_.allocator(), pool_, transientGroup(TransientPhase::Tail, poolView_),
                                         transientSlot(f), ici, &img.image, &img.alloc),
                  "vmaCreateImage(sensorPass.snapshot)");

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = img.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vulkan::createImageView(d, &vci, nullptr, &img.view),
                  "vulkan::createImageView(sensorPass.snapshot)");
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
            vulkan::updateDescriptorSets(d, 1, &w, 0, nullptr);
            // Every face slot starts on the snapshot: a valid image the shader
            // never reads while its faceMask bit is clear.
            VkDescriptorImageInfo faceI[lensrig::kFaceCount];
            for (auto& fi : faceI) fi = srcI;
            w.dstBinding      = 2;
            w.descriptorCount = lensrig::kFaceCount;
            w.pImageInfo      = faceI;
            vulkan::updateDescriptorSets(d, 1, &w, 0, nullptr);
            boundFaces_[f].fill(0);
        }
    }

    bool SensorPass::prepare(uint32_t frame, VkImageView swapView, uint32_t width, uint32_t height,
                             const Params& p) {
        if (!p.active() || pipe_ == VK_NULL_HANDLE) return false;
        resize(width, height);
        if (snapshot_[frame].image == VK_NULL_HANDLE) return false;
        if (width != width_ || height != height_) return false;

        // The destination is the swapchain image for THIS frame's acquired
        // index, which rotates independently of the frame-in-flight slot, so
        // refresh the storage binding whenever it differs from what this slot
        // last pointed at. (The prior frame using this slot has retired — the
        // caller's framesInFlight fence guarantees it — so rewriting here
        // cannot touch a descriptor an in-flight command buffer still reads.)
        // Compared by serial: a recreated swapchain's views can keep their
        // values, and the set would go on naming the destroyed ones.
        const uint64_t swapSerial = imageViewSerial(swapView);
        if (boundDst_[frame] != swapSerial) {
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
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
            boundDst_[frame] = swapSerial;
        }

        // The fisheye rig: its side views by face (each slot's previous use
        // has retired, as above), and this frame's layout block.
        RigUboData ubo{};
        if (p.rigActive) {
            ubo.info[0]   = p.rig.faceMask;
            ubo.margin[0] = p.rig.marginU;
            ubo.margin[1] = p.rig.marginV;
            for (int f = 0; f < lensrig::kFaceCount; ++f) {
                const auto& r = p.rig.faces[f];
                ubo.faceRect[f][0] = r.u0;
                ubo.faceRect[f][1] = r.v0;
                ubo.faceRect[f][2] = r.u1 > r.u0 ? 1.f / (r.u1 - r.u0) : 0.f;
                ubo.faceRect[f][3] = r.v1 > r.v0 ? 1.f / (r.v1 - r.v0) : 0.f;
                const bool     live   = (p.rig.faceMask & (1u << f)) != 0u && p.faceViews[f] != VK_NULL_HANDLE;
                const uint64_t serial = live ? imageViewSerial(p.faceViews[f]) : 0u;
                if (boundFaces_[frame][f] == serial) continue;
                VkDescriptorImageInfo fi{};
                fi.sampler     = sampler_;
                fi.imageView   = live ? p.faceViews[f] : snapshot_[frame].view;
                fi.imageLayout = live ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w{};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet          = sets_[frame];
                w.dstBinding      = 2;
                w.dstArrayElement = static_cast<uint32_t>(f);
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo      = &fi;
                vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
                boundFaces_[frame][f] = serial;
            }
        }
        std::memcpy(rigUboMapped_[frame], &ubo, sizeof(ubo));
        return true;
    }

    // The swapchain arrives written by the overlay pass (colour attachment)
    // and/or the resolve/RCAS (compute store); the snapshot's previous contents
    // are discarded.
    void SensorPass::recordSnapshot(VkCommandBuffer cb, uint32_t frame, VkImage swapImage,
                                    uint32_t width, uint32_t height) {
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent         = {width, height, 1};
        vkCmdCopyImage(cb, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       snapshot_[frame].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);
    }

    void SensorPass::recordApply(VkCommandBuffer cb, uint32_t frame, uint32_t width, uint32_t height,
                                 const Params& p) {
        SensorPush pc{};
        pc.width     = width;
        pc.height    = height;
        pc.flags     = (p.distortActive ? kFlagDistort : 0u) | (p.noiseActive ? kFlagNoise : 0u) |
                       (p.rigActive ? kFlagRig : 0u);
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
