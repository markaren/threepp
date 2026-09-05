#include "threepp/renderers/vulkan/EventCameraDetector.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/event_detect.comp.spv.h"

#include <algorithm>
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
            uint32_t frameTimeUs;
            uint32_t prevFrameTimeUs;
        };

        // 16-byte header at the front of every event-stream buffer.
        // Layout matches the GLSL block in event_detect.comp:
        // {count, capacity, overflow, frameTimeUs}.
        struct EventStreamHeader {
            uint32_t count;
            uint32_t capacity;
            uint32_t overflow;
            uint32_t frameTimeUs;
        };
        static_assert(sizeof(EventStreamHeader) == 16,
                      "EventStreamHeader must match GLSL layout");
        static_assert(sizeof(EventCameraDetector::Event) == 16,
                      "Event must match GLSL layout");

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
        for (auto& b : readbackRing_)     destroyBuffer(ctx_.allocator(), b);
        for (auto& b : eventStreamRing_)  destroyBuffer(ctx_.allocator(), b);
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

        // Four bindings:
        //   0 = scene buffer (storage, read)
        //   1 = log-history image (rg32f storage, read/write)
        //   2 = accumulator image (rgba8 storage, read/write)
        //   3 = event stream buffer (storage, atomic append)
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
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
        b[3].binding         = 3;
        b[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[3].descriptorCount = 1;
        b[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

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
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(event_detect)");

        vkDestroyShaderModule(ctx_.device(), shader, nullptr);
    }

    void EventCameraDetector::allocateDescriptorPool() {
        // kRingSize sets — one per ring slot (see descSets_ in the header).
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[0].descriptorCount = 2 * kRingSize;  // (scene buf + event stream buf) per set
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 2 * kRingSize;  // (log history + accumulator) per set

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kRingSize;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(event_detect)");

        std::array<VkDescriptorSetLayout, kRingSize> layouts{};
        layouts.fill(dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = kRingSize;
        dsai.pSetLayouts        = layouts.data();
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, descSets_.data()),
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

        // Allocate persistent storage images. rg32f for the log-luminance
        // history — .x is the emit reference, .y the previous frame's raw
        // sample that the sub-frame timestamp interpolation ramps from
        // (signed log values, plenty of dynamic range); rgba8 for the
        // visualisation accumulator (sRGB-domain output).
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
        logHistoryImg_  = allocImage(width, height, VK_FORMAT_R32G32_SFLOAT,
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

        // (Re)allocate the per-ring-slot event stream buffers, sized to the
        // per-frame worst case for this resolution and per-pixel cap (see
        // kEventStreamCapacityFloor for why they must never overflow).
        streamCapacity_ = requiredCapacity(width, height, perPixelCap_);
        allocateStreams();

        // Update the storage-image bindings (1 = log history, 2 = accumulator)
        // and the per-slot event-stream binding (3) on EVERY ring-slot set.
        // Both storage images are shared across slots; binding 3 is the one
        // resource that differs per set — slot i permanently points at
        // eventStreamRing_[i], so record() only has to *bind* the slot's set.
        // The scene-buffer binding (0) is rewritten lazily by
        // updateSceneBinding() when the host hands us a new VkBuffer handle.
        VkDescriptorImageInfo histInfo{};
        histInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        histInfo.imageView   = logHistoryImg_.view;
        VkDescriptorImageInfo accInfo{};
        accInfo.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
        accInfo.imageView    = accumulatorImg_.view;

        std::array<VkDescriptorBufferInfo, kRingSize> streamInfos{};
        std::array<VkWriteDescriptorSet, kRingSize * 3> w{};
        uint32_t wi = 0;
        for (uint32_t s = 0; s < kRingSize; ++s) {
            streamInfos[s].buffer = eventStreamRing_[s].handle;
            streamInfos[s].offset = 0;
            streamInfos[s].range  = VK_WHOLE_SIZE;

            w[wi].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[wi].dstSet          = descSets_[s];
            w[wi].dstBinding      = 1;
            w[wi].descriptorCount = 1;
            w[wi].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[wi].pImageInfo      = &histInfo;
            ++wi;
            w[wi].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[wi].dstSet          = descSets_[s];
            w[wi].dstBinding      = 2;
            w[wi].descriptorCount = 1;
            w[wi].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[wi].pImageInfo      = &accInfo;
            ++wi;
            w[wi].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[wi].dstSet          = descSets_[s];
            w[wi].dstBinding      = 3;
            w[wi].descriptorCount = 1;
            w[wi].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[wi].pBufferInfo     = &streamInfos[s];
            ++wi;
        }
        vkUpdateDescriptorSets(ctx_.device(), wi, w.data(), 0, nullptr);

        width_     = width;
        height_    = height;
        firstFrame_ = true;
        writeSlot_ = 0;
        currentSceneBuf_ = VK_NULL_HANDLE;
    }

    void EventCameraDetector::updateSceneBinding(VkBuffer sceneBuf) {
        if (sceneBuf == currentSceneBuf_) return;
        // The scene buffer is shared across all ring slots — rewrite binding 0
        // on every set. Only reached on a handle change (resize / first bind),
        // never per-frame, so it can't race an in-flight submission.
        VkDescriptorBufferInfo info{};
        info.buffer = sceneBuf;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, kRingSize> w{};
        for (uint32_t s = 0; s < kRingSize; ++s) {
            w[s].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[s].dstSet          = descSets_[s];
            w[s].dstBinding      = 0;
            w[s].descriptorCount = 1;
            w[s].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[s].pBufferInfo     = &info;
        }
        vkUpdateDescriptorSets(ctx_.device(),
                                static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        currentSceneBuf_ = sceneBuf;
    }

    void EventCameraDetector::record(VkCommandBuffer cb, VkBuffer sceneBuf, const Params& params) {
        if (width_ == 0 || height_ == 0 || sceneBuf == VK_NULL_HANDLE) return;

        updateSceneBinding(sceneBuf);

        // The write-slot's descriptor set already has binding 3 pointing at
        // eventStreamRing_[writeSlot_] (wired once in resize()). Slot rotates
        // per frame so the host's readEventStreamInto (which reads the OLDEST
        // slot) never collides with the GPU's writes, and — because kRingSize
        // (3) > kFramesInFlight (2) — this slot's set is not bound by any
        // still-pending submission, so no per-frame descriptor update is
        // needed here (that update was the in-flight-set hazard / flicker).

        // Zero the event-stream header for this slot. {count=0, capacity,
        // overflow=0, frameTimeUs} — capacity is constant and we don't
        // strictly need to rewrite it every frame, but vkCmdFillBuffer
        // with a single 32-bit value can't hand-place capacity, so we
        // overwrite all 16 bytes with the desired values via vkCmdUpdateBuffer.
        // Cheaper than a roundtrip mapped write.
        const EventStreamHeader hdr{
                0u, streamCapacity_, 0u, params.frameTimeUs};
        vkCmdUpdateBuffer(cb, eventStreamRing_[writeSlot_].handle,
                          0, sizeof(hdr), &hdr);

        // Make both buffer writes (sceneBuf transfer + stream header
        // transfer) visible to the compute shader.
        std::array<VkBufferMemoryBarrier, 2> preBarriers{};
        preBarriers[0].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        preBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        preBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        preBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[0].buffer = sceneBuf;
        preBarriers[0].size   = VK_WHOLE_SIZE;
        preBarriers[1].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        preBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        preBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                       VK_ACCESS_SHADER_WRITE_BIT;
        preBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarriers[1].buffer = eventStreamRing_[writeSlot_].handle;
        preBarriers[1].size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr,
                              static_cast<uint32_t>(preBarriers.size()), preBarriers.data(),
                              0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelineLayout_, 0, 1, &descSets_[writeSlot_], 0, nullptr);

        PC pc{};
        pc.width            = width_;
        pc.height           = height_;
        pc.threshold        = params.threshold;
        pc.decay            = params.decay;
        pc.minLuma          = params.minLuma;
        pc.maxEventsPerPixel = params.maxEventsPerPixel;
        pc.firstFrame       = firstFrame_ ? 1u : 0u;
        pc.frameTimeUs      = params.frameTimeUs;
        // The detector owns the previous stamp — Params only ever carries
        // "now". On the first frame there is no previous sample to ramp
        // from, so collapse the interval (that frame emits nothing anyway).
        pc.prevFrameTimeUs  = firstFrame_ ? params.frameTimeUs : lastFrameTimeUs_;
        vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(pc), &pc);

        const uint32_t gx = (width_  + kLocalX - 1) / kLocalX;
        const uint32_t gy = (height_ + kLocalY - 1) / kLocalY;
        vkCmdDispatch(cb, gx, gy, 1);

        // Event-stream writes need to be flushed to the host visibility
        // domain. The buffer is HOST_VISIBLE but the GPU's storage-buffer
        // writes aren't guaranteed visible to host reads without a barrier
        // pairing SHADER_WRITE → HOST_READ across COMPUTE → HOST stages.
        // The host-side readEventStreamInto also invalidateHostReads()
        // before reading; both are required for correctness on discrete GPUs.
        VkBufferMemoryBarrier streamHostBarrier{};
        streamHostBarrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        streamHostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        streamHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        streamHostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        streamHostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        streamHostBarrier.buffer = eventStreamRing_[writeSlot_].handle;
        streamHostBarrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT,
                              0, 0, nullptr, 1, &streamHostBarrier, 0, nullptr);

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

        writeSlot_       = (writeSlot_ + 1) % kRingSize;
        firstFrame_      = false;
        lastFrameTimeUs_ = params.frameTimeUs;
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

        invalidateHostReads(ctx_.allocator(), src.alloc, 0, bytes);
        void* mapped = nullptr;
        if (vmaMapMemory(ctx_.allocator(), src.alloc, &mapped) != VK_SUCCESS) {
            return 0;
        }
        std::memcpy(dst, mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx_.allocator(), src.alloc);
        return static_cast<size_t>(bytes);
    }

    uint32_t EventCameraDetector::requiredCapacity(uint32_t w, uint32_t h, uint32_t perPixel) const {
        const uint64_t need = static_cast<uint64_t>(w) * h * std::max(1u, perPixel);
        const uint64_t capped = std::min<uint64_t>(need, 0xFFFFFFFFull);
        return std::max<uint32_t>(kEventStreamCapacityFloor, static_cast<uint32_t>(capped));
    }

    void EventCameraDetector::allocateStreams() {
        // Host-visible so the host can mmap and read directly; STORAGE_BUFFER_BIT
        // for the shader's atomic appends, TRANSFER_DST_BIT so the header can be
        // rewritten at frame start. Size = 16B header + capacity events.
        const VkDeviceSize streamBytes =
                sizeof(EventStreamHeader) +
                static_cast<VkDeviceSize>(streamCapacity_) * sizeof(Event);
        for (auto& b : eventStreamRing_) {
            destroyBuffer(ctx_.allocator(), b);
            b = createBuffer(
                    ctx_.allocator(), ctx_.device(), streamBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    void EventCameraDetector::bindStreams() {
        std::array<VkDescriptorBufferInfo, kRingSize> infos{};
        std::array<VkWriteDescriptorSet, kRingSize> w{};
        for (uint32_t s = 0; s < kRingSize; ++s) {
            infos[s].buffer = eventStreamRing_[s].handle;
            infos[s].offset = 0;
            infos[s].range  = VK_WHOLE_SIZE;
            w[s].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[s].dstSet          = descSets_[s];
            w[s].dstBinding      = 3;
            w[s].descriptorCount = 1;
            w[s].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[s].pBufferInfo     = &infos[s];
        }
        vkUpdateDescriptorSets(ctx_.device(), kRingSize, w.data(), 0, nullptr);
    }

    bool EventCameraDetector::needsGrowthFor(uint32_t maxEventsPerPixel) const {
        if (width_ == 0 || height_ == 0) return false;
        return requiredCapacity(width_, height_, maxEventsPerPixel) > streamCapacity_;
    }

    bool EventCameraDetector::setMaxEventsPerPixel(uint32_t maxEventsPerPixel) {
        perPixelCap_ = std::max(1u, maxEventsPerPixel);
        if (!needsGrowthFor(perPixelCap_)) return false;
        // Device idle is the caller's contract (mirrors resize()): the old
        // buffers may be bound by a still-pending submission otherwise.
        streamCapacity_ = requiredCapacity(width_, height_, perPixelCap_);
        allocateStreams();
        bindStreams();
        return true;
    }

    size_t EventCameraDetector::readEventStreamInto(Event* dst, size_t cap,
                                                    bool* overflowed) const {
        if (overflowed) *overflowed = false;
        if (!dst || width_ == 0 || height_ == 0) return 0;

        // Same OLDEST-slot rule as visualisation readback — guaranteed
        // complete by the in-flight-fence wait two frames ago. The
        // event-stream buffer for that slot was also written, then
        // flushed by the COMPUTE → HOST barrier the record() emitted.
        const uint32_t oldestSlot = writeSlot_;
        const Buffer& src = eventStreamRing_[oldestSlot];
        if (src.handle == VK_NULL_HANDLE) return 0;
        const VkDeviceSize hdrBytes = sizeof(EventStreamHeader);
        if (src.size < hdrBytes) return 0;

        invalidateHostReads(ctx_.allocator(), src.alloc, 0, src.size);
        void* mapped = nullptr;
        if (vmaMapMemory(ctx_.allocator(), src.alloc, &mapped) != VK_SUCCESS) {
            return 0;
        }

        const auto* hdr = static_cast<const EventStreamHeader*>(mapped);
        const uint32_t count    = hdr->count;
        const uint32_t capacity = hdr->capacity;
        if (overflowed) *overflowed = (hdr->overflow != 0u);
        const uint32_t effective = std::min(count, capacity);
        const size_t toCopy = std::min(static_cast<size_t>(effective), cap);
        if (toCopy > 0) {
            const auto* events = reinterpret_cast<const Event*>(
                    static_cast<const uint8_t*>(mapped) + sizeof(EventStreamHeader));
            std::memcpy(dst, events, toCopy * sizeof(Event));
        }
        vmaUnmapMemory(ctx_.allocator(), src.alloc);

        // Time-order the packet. Two reasons, both about the stream being
        // usable rather than merely correct:
        //   • A real DVS reads out in time order, and every downstream
        //     consumer (.aedat writers, ROS bridges, time-surface / voxel-grid
        //     encoders) assumes non-decreasing t. With sub-frame interpolated
        //     stamps the events of one frame genuinely span an interval, so
        //     "unordered" would now be visible rather than academic.
        //   • The GPU appends with an atomicAdd, so the raw order is whatever
        //     the scheduler happened to do — it varies run to run on identical
        //     input. Sorting on the full key (t, y, x, polarity) makes the
        //     packet deterministic, which is what replayable datasets need.
        // The cost is an O(n log n) pass over one frame's events on the host;
        // at the ~10-50k events a busy frame produces that is well under the
        // readback it follows.
        std::sort(dst, dst + toCopy, [](const Event& a, const Event& b) {
            if (a.t_us != b.t_us) return a.t_us < b.t_us;
            if (a.y != b.y) return a.y < b.y;
            if (a.x != b.x) return a.x < b.x;
            return a.polarity < b.polarity;
        });
        return toCopy;
    }

}// namespace threepp::vulkan
