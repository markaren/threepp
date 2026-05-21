#include "threepp/renderers/vulkan/EventCameraDetector.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/event_detect.comp.spv.h"

#include <array>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    namespace {

        struct PC {
            uint32_t width;
            uint32_t height;
            float    threshold;
            float    decay;
            float    minLuma;
            uint32_t maxEventsPerPixel;
            uint32_t firstFrame;
            uint32_t _pad;
        };

        // 16x16 = 256 threads is a sensible compute-shader block size on
        // all desktop GPUs; matches the dispatch math in event_detect.comp.
        constexpr uint32_t kLocalX = 8;
        constexpr uint32_t kLocalY = 8;

        VkImageMemoryBarrier fullColorBarrier(VkImage img, VkImageLayout oldL,
                                              VkImageLayout newL,
                                              VkAccessFlags srcAccess,
                                              VkAccessFlags dstAccess) {
            VkImageMemoryBarrier b{};
            b.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout                   = oldL;
            b.newLayout                   = newL;
            b.srcAccessMask               = srcAccess;
            b.dstAccessMask               = dstAccess;
            b.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
            b.image                       = img;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            return b;
        }

    }// namespace

    EventCameraDetector::EventCameraDetector(VulkanContext& ctx) : ctx_(ctx) {
        createPipeline();
        allocateDescriptorPool();
    }

    EventCameraDetector::~EventCameraDetector() {
        const VkDevice d = ctx_.device();
        destroyImages();
        for (auto& b : readbackRing_) destroyBuffer(ctx_.allocator(), b);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
    }

    void EventCameraDetector::createPipeline() {
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kEventDetectCompSpv);
        smci.pCode    = kEventDetectCompSpv;
        VkShaderModule shader = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &shader),
              "vkCreateShaderModule(event_detect)");

        // Three bindings:
        //   0 = scene buffer (storage, read)
        //   1 = log-history image (r32f storage, read/write)
        //   2 = accumulator image (rgba8 storage, read/write)
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding         = 1;
        b[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[2].binding         = 2;
        b[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[2].descriptorCount = 1;
        b[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(event_detect)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PC);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(event_detect)");

        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = shader;
        ssci.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = ssci;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(event_detect)");

        vkDestroyShaderModule(ctx_.device(), shader, nullptr);
    }

    void EventCameraDetector::allocateDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[0].descriptorCount = 1;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(event_detect)");

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &dsLayout_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, &descSet_),
              "vkAllocateDescriptorSets(event_detect)");
    }

    void EventCameraDetector::destroyImages() {
        destroyImage2D(ctx_.allocator(), ctx_.device(), logHistoryImg_);
        destroyImage2D(ctx_.allocator(), ctx_.device(), accumulatorImg_);
    }

    void EventCameraDetector::resize(uint32_t width, uint32_t height) {
        if (width == 0 || height == 0) return;
        if (width == width_ && height == height_ && accumulatorImg_.image != VK_NULL_HANDLE) return;

        // Free previous allocations. Pre-destroy must wait for any in-flight
        // dispatch using these images; the renderer's setEventCameraEnabled
        // path issues vkDeviceWaitIdle before calling resize so we're safe.
        destroyImages();

        // Allocate persistent storage images. r32f for the log-luminance
        // history (signed log value per pixel, plenty of dynamic range);
        // rgba8 for the visualisation accumulator (sRGB-domain output).
        auto allocImage = [this](uint32_t w, uint32_t h, VkFormat fmt,
                                  VkImageUsageFlags usage, const char* name) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = fmt;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = fmt;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = usage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci,
                                  &out.image, &out.alloc, nullptr),
                  name);

            VkImageViewCreateInfo vci{};
            vci.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image                       = out.image;
            vci.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
            vci.format                      = fmt;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(event_detect)");
            return out;
        };

        const VkImageUsageFlags storageUsage =
                VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        logHistoryImg_  = allocImage(width, height, VK_FORMAT_R32_SFLOAT,
                                       storageUsage, "evtLogHistory");
        accumulatorImg_ = allocImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                       storageUsage, "evtAccumulator");

        // Transition both images to GENERAL layout via a one-shot cmd
        // buffer. The compute shader and subsequent copies require
        // GENERAL on storage images.
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx_.queueFamilies().graphics;
        VkCommandPool cp = VK_NULL_HANDLE;
        check(vkCreateCommandPool(ctx_.device(), &cpci, nullptr, &cp),
              "vkCreateCommandPool(event_detect transition)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cp;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &cbai, &cb),
              "vkAllocateCommandBuffers(event_detect transition)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);

        std::array<VkImageMemoryBarrier, 2> initBs{
                fullColorBarrier(logHistoryImg_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT),
                fullColorBarrier(accumulatorImg_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT),
        };
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(initBs.size()), initBs.data());
        vkEndCommandBuffer(cb);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence f = VK_NULL_HANDLE;
        vkCreateFence(ctx_.device(), &fci, nullptr, &f);
        vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, f);
        vkWaitForFences(ctx_.device(), 1, &f, VK_TRUE, UINT64_MAX);
        vkDestroyFence(ctx_.device(), f, nullptr);
        vkDestroyCommandPool(ctx_.device(), cp, nullptr);

        // (Re)allocate the readback ring at the new size.
        const VkDeviceSize readbackBytes =
                static_cast<VkDeviceSize>(width) * height * 4;
        for (auto& b : readbackRing_) {
            destroyBuffer(ctx_.allocator(), b);
            b = createBuffer(
                    ctx_.allocator(), ctx_.device(), readbackBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }

        // Update descriptor binding for the storage images. The scene-buffer
        // binding is rewritten lazily by updateSceneBinding() when the host
        // hands us a new VkBuffer handle.
        VkDescriptorImageInfo histInfo{};
        histInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        histInfo.imageView   = logHistoryImg_.view;
        VkDescriptorImageInfo accInfo{};
        accInfo.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
        accInfo.imageView    = accumulatorImg_.view;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet          = descSet_;
        w[0].dstBinding      = 1;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[0].pImageInfo      = &histInfo;
        w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet          = descSet_;
        w[1].dstBinding      = 2;
        w[1].descriptorCount = 1;
        w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo      = &accInfo;
        vkUpdateDescriptorSets(ctx_.device(),
                                static_cast<uint32_t>(w.size()), w.data(),
                                0, nullptr);

        width_     = width;
        height_    = height;
        firstFrame_ = true;
        writeSlot_ = 0;
        currentSceneBuf_ = VK_NULL_HANDLE;
    }

    void EventCameraDetector::updateSceneBinding(VkBuffer sceneBuf) {
        if (sceneBuf == currentSceneBuf_) return;
        VkDescriptorBufferInfo info{};
        info.buffer = sceneBuf;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = descSet_;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &info;
        vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        currentSceneBuf_ = sceneBuf;
    }

    void EventCameraDetector::record(VkCommandBuffer cb, VkBuffer sceneBuf, const Params& params) {
        if (width_ == 0 || height_ == 0 || sceneBuf == VK_NULL_HANDLE) return;

        updateSceneBinding(sceneBuf);

        // Barrier: sceneBuf (just written by the renderer's image→buffer
        // copy in recordSceneCapture) must be visible to the compute
        // shader's buffer reads. Use a global memory barrier for
        // simplicity — the cost is negligible.
        VkBufferMemoryBarrier sceneBarrier{};
        sceneBarrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        sceneBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sceneBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sceneBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sceneBarrier.buffer = sceneBuf;
        sceneBarrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 1, &sceneBarrier, 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelineLayout_, 0, 1, &descSet_, 0, nullptr);

        PC pc{};
        pc.width            = width_;
        pc.height           = height_;
        pc.threshold        = params.threshold;
        pc.decay            = params.decay;
        pc.minLuma          = params.minLuma;
        pc.maxEventsPerPixel = params.maxEventsPerPixel;
        pc.firstFrame       = firstFrame_ ? 1u : 0u;
        pc._pad             = 0;
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(pc), &pc);

        const uint32_t gx = (width_  + kLocalX - 1) / kLocalX;
        const uint32_t gy = (height_ + kLocalY - 1) / kLocalY;
        vkCmdDispatch(cb, gx, gy, 1);

        // Barrier: accumulator GENERAL → TRANSFER_SRC for the copy.
        VkImageMemoryBarrier toSrc = fullColorBarrier(
                accumulatorImg_.image,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toSrc);

        // Copy accumulator → ring slot.
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {width_, height_, 1};
        vkCmdCopyImageToBuffer(cb, accumulatorImg_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                readbackRing_[writeSlot_].handle, 1, &region);

        // Barrier back to GENERAL so the next frame's compute can write.
        VkImageMemoryBarrier toGen = fullColorBarrier(
                accumulatorImg_.image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toGen);

        writeSlot_  = (writeSlot_ + 1) % kRingSize;
        firstFrame_ = false;
    }

    std::vector<unsigned char> EventCameraDetector::readVisualisation() const {
        if (width_ == 0 || height_ == 0) return {};
        const size_t bytes = static_cast<size_t>(width_) * height_ * 4;
        std::vector<unsigned char> rgba(bytes);
        const size_t got = readVisualisationInto(rgba.data(), rgba.size());
        if (got == 0) return {};
        return rgba;
    }

    size_t EventCameraDetector::readVisualisationInto(unsigned char* dst, size_t cap) const {
        if (!dst || width_ == 0 || height_ == 0) return 0;

        // The OLDEST ring slot (the one we'll WRITE next) is guaranteed
        // complete: it was filled at least `kRingSize` record() calls ago,
        // and Vulkan's in-flight fence model ensures each prior submit has
        // signalled by the time the host is back in the per-frame loop.
        const uint32_t oldestSlot = writeSlot_;
        const Buffer& src = readbackRing_[oldestSlot];
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(width_) * height_ * 4;
        if (src.handle == VK_NULL_HANDLE || src.size < bytes) return 0;
        if (cap < static_cast<size_t>(bytes)) return 0;

        vmaInvalidateAllocation(ctx_.allocator(), src.alloc, 0, bytes);
        void* mapped = nullptr;
        if (vmaMapMemory(ctx_.allocator(), src.alloc, &mapped) != VK_SUCCESS) {
            return 0;
        }
        std::memcpy(dst, mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx_.allocator(), src.alloc);
        return static_cast<size_t>(bytes);
    }

}// namespace threepp::vulkan
