#include "VkInfer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace rfdetr {

namespace {

    void vkCheck(VkResult r, const char* what) {
        if (r != VK_SUCCESS) {
            throw std::runtime_error(std::string("VkInfer: ") + what +
                                     " failed (VkResult " + std::to_string(static_cast<int>(r)) + ")");
        }
    }

}// namespace

VkInfer::VkInfer(VkInstance instance, VkDevice device, VkPhysicalDevice phys, VkQueue queue, uint32_t queueFamily)
    : device_(device), phys_(phys), queue_(queue), queueFamily_(queueFamily) {

    detectCoopMat_(instance);

    {
        VkPhysicalDeviceSubgroupProperties sg{};
        sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(phys_, &p2);
        subgroupClustered_ = (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                             (sg.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0 &&
                             sg.subgroupSize >= 8;// the fused attention reduces over clusters of 8
    }
    {
        // The SM / CU count, from the vendor extension that reports it (queried, not enabled).
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
        auto has = [&](const char* name) {
            return std::any_of(exts.begin(), exts.end(),
                               [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
        };
        uint32_t sms = 0;
        if (has(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderSMBuiltinsPropertiesNV sm{};
            sm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV;
            VkPhysicalDeviceProperties2 p2{};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &sm;
            vkGetPhysicalDeviceProperties2(phys_, &p2);
            sms = sm.shaderSMCount;
        } else if (has(VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderCorePropertiesAMD cp{};
            cp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
            VkPhysicalDeviceProperties2 p2{};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &cp;
            vkGetPhysicalDeviceProperties2(phys_, &p2);
            sms = cp.shaderEngineCount * cp.shaderArraysPerEngineCount * cp.computeUnitsPerShaderArray;
        }
        if (sms > 0) smCount_ = sms;
    }

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = queueFamily_;
    vkCheck(vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_), "create command pool");
    // Separate pool for one-shot uploads/readbacks so they never share state with
    // the frame command buffer (which may be mid-recording when an upload runs).
    vkCheck(vkCreateCommandPool(device_, &cpci, nullptr, &oneShotPool_), "create one-shot command pool");

    // Descriptor sets are cached and reused across inferences (not reset per frame),
    // so the pool must hold every unique (pipeline, buffers) set for the whole run.
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 32768;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 8192;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    vkCheck(vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool_), "create descriptor pool");

    dummy_ = allocBuffer(16,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    zero(dummy_.buffer);

    // endFrame() waits on this, not on the whole queue: the queue is the
    // renderer's, and its frames in flight are none of the inference's business.
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCheck(vkCreateFence(device_, &fci, nullptr, &frameFence_), "create frame fence");

    tsEnabled_ = std::getenv("RFDETR_PROFILE") != nullptr;
    if (tsEnabled_) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        tsPeriodNs_ = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kMaxTimestamps;
        if (vkCreateQueryPool(device_, &qpci, nullptr, &tsPool_) != VK_SUCCESS) tsPool_ = VK_NULL_HANDLE;
    }
}

VkInfer::~VkInfer() {
    arenaSlots_.clear();// free pooled buffers before pools/device teardown
    if (frameFence_) vkDestroyFence(device_, frameFence_, nullptr);
    if (tsPool_) vkDestroyQueryPool(device_, tsPool_, nullptr);
    if (descPool_) vkDestroyDescriptorPool(device_, descPool_, nullptr);
    if (oneShotPool_) vkDestroyCommandPool(device_, oneShotPool_, nullptr);
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
}

void VkInfer::detectCoopMat_(VkInstance instance) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
    if (std::none_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
            return std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
        }))
        return;

    // The features VulkanContext enables together with the extension when the device has them all.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{};
    cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &cm;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(phys_, &f2);
    if (!cm.cooperativeMatrix || !f12.vulkanMemoryModel || !f12.shaderFloat16 || !f11.storageBuffer16BitAccess) return;

    // The tensor-core kernels tile a workgroup as 32-wide subgroups.
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(phys_, &p2);
    if (sg.subgroupSize != 32) return;

    auto shapesOf = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!shapesOf) return;
    uint32_t count = 0;
    shapesOf(phys_, &count, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count);
    for (auto& s : shapes) s.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    shapesOf(phys_, &count, shapes.data());
    coopMatF16_ = std::any_of(shapes.begin(), shapes.end(), [](const VkCooperativeMatrixPropertiesKHR& s) {
        return s.MSize == 16 && s.NSize == 16 && s.KSize == 16 && s.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
               s.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && s.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
               s.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && s.scope == VK_SCOPE_SUBGROUP_KHR &&
               !s.saturatingAccumulation;
    });
}

void VkInfer::markTimestamp(const char* label) {
    if (!tsEnabled_ || !tsPool_ || !recording_ || tsCount_ >= kMaxTimestamps) return;
    vkCmdWriteTimestamp(frameCb_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool_, tsCount_);
    tsLabels_.push_back(label);
    tsIsDispatch_.push_back(false);
    ++tsCount_;
}

uint32_t VkInfer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("VkInfer: no compatible memory type");
}

VkTensor VkInfer::allocBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    VkTensor t;
    t.device = device_;
    t.capacity = bytes;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &bci, nullptr, &t.buffer), "create buffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, t.buffer, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device_, &mai, nullptr, &t.memory), "allocate memory");
    vkCheck(vkBindBufferMemory(device_, t.buffer, t.memory, 0), "bind buffer memory");
    return t;
}

VkTensor VkInfer::createOwned(const std::vector<uint32_t>& shape) {
    uint32_t n = 1;
    for (auto s : shape) n *= s;
    VkDeviceSize bytes = static_cast<VkDeviceSize>(n ? n : 1u) * sizeof(float);
    VkTensor t = allocBuffer(bytes,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    t.shape = shape;
    return t;
}

VkTensor VkInfer::createOwnedRaw(VkDeviceSize bytes) {
    if (bytes < 4) bytes = 4;
    return allocBuffer(bytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

Tensor VkInfer::createTensor(std::initializer_list<uint32_t> shape) {
    return createTensorV(std::vector<uint32_t>(shape));
}

VkBuffer VkInfer::acquireArena(VkDeviceSize bytes) {
    if (bytes < 4) bytes = 4;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (arenaCursor_ >= arenaSlots_.size()) {
        // First time this slot is needed: allocate it (once, ever).
        arenaSlots_.push_back(allocBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        ++layoutGen_;
    } else if (arenaSlots_[arenaCursor_].capacity < bytes) {
        // Slot exists but is too small (only happens if a shape grows): regrow.
        arenaSlots_[arenaCursor_] = allocBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        ++layoutGen_;
    }
    return arenaSlots_[arenaCursor_++].buffer;
}

Tensor VkInfer::createTensorV(const std::vector<uint32_t>& shape) {
    uint32_t n = 1;
    for (auto s : shape) n *= s;
    VkDeviceSize bytes = static_cast<VkDeviceSize>(n ? n : 1u) * sizeof(float);
    return Tensor{acquireArena(bytes), shape};
}

Tensor VkInfer::createRaw(VkDeviceSize bytes) {
    return Tensor{acquireArena(bytes), {}};
}

void VkInfer::resetArena() {
    arenaCursor_ = 0;// rewind; buffers are kept and reused next inference
}

template<typename Fn>
void VkInfer::oneShot(Fn&& fn) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = oneShotPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device_, &ai, &cb), "allocate one-shot cmd buffer");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(cb, &bi), "begin one-shot cmd buffer");
    fn(cb);
    vkCheck(vkEndCommandBuffer(cb), "end one-shot cmd buffer");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkCheck(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "submit one-shot");
    vkCheck(vkQueueWaitIdle(queue_), "wait one-shot");
    vkFreeCommandBuffers(device_, oneShotPool_, 1, &cb);
}

void VkInfer::upload(VkBuffer dst, const void* data, VkDeviceSize bytes) {
    ensureUploadStaging(bytes);
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, uploadStaging_.memory, 0, bytes, 0, &mapped), "map staging (upload)");
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(device_, uploadStaging_.memory);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cb, uploadStaging_.buffer, dst, 1, &region);
    });
}

void VkInfer::uploadMany(const std::vector<UploadItem>& items) {
    VkDeviceSize total = 0;
    for (const auto& it : items) total += it.bytes;
    if (total == 0) return;
    ensureUploadStaging(total);

    // Pack every source contiguously into the staging buffer, then issue one
    // copy per item from its slice — all in a single submit+wait.
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, uploadStaging_.memory, 0, total, 0, &mapped), "map staging (uploadMany)");
    std::vector<VkDeviceSize> offsets(items.size());
    VkDeviceSize off = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        offsets[i] = off;
        std::memcpy(static_cast<char*>(mapped) + off, items[i].data, items[i].bytes);
        off += items[i].bytes;
    }
    vkUnmapMemory(device_, uploadStaging_.memory);

    oneShot([&](VkCommandBuffer cb) {
        for (size_t i = 0; i < items.size(); ++i) {
            VkBufferCopy region{};
            region.srcOffset = offsets[i];
            region.size = items[i].bytes;
            vkCmdCopyBuffer(cb, uploadStaging_.buffer, items[i].dst, 1, &region);
        }
    });
}

void VkInfer::ensureUploadStaging(VkDeviceSize bytes) {
    if (uploadStaging_.buffer && uploadStaging_.capacity >= bytes) return;
    uploadStaging_ = allocBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void VkInfer::ensureReadbackStaging(VkDeviceSize bytes) {
    if (readbackStaging_.buffer && readbackStaging_.capacity >= bytes) return;
    // Prefer HOST_CACHED so the host-side memcpy reads from cached (not write-
    // combined) memory — uncached PCIe reads run at ~0.2 GB/s. Keep COHERENT so
    // no manual invalidate is needed.
    const auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    try {
        readbackStaging_ = allocBuffer(bytes, usage,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    } catch (const std::runtime_error&) {
        readbackStaging_ = allocBuffer(bytes, usage,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

void VkInfer::readback(VkBuffer src, void* dst, VkDeviceSize bytes, VkDeviceSize srcOffset) {
    ensureReadbackStaging(bytes);
    oneShot([&](VkCommandBuffer cb) {
        // Make the producing compute writes available to the copy.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.size = bytes;
        vkCmdCopyBuffer(cb, src, readbackStaging_.buffer, 1, &region);
    });
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, readbackStaging_.memory, 0, bytes, 0, &mapped), "map staging (readback)");
    std::memcpy(dst, mapped, bytes);
    vkUnmapMemory(device_, readbackStaging_.memory);
}

void VkInfer::zero(VkBuffer buf) {
    oneShot([&](VkCommandBuffer cb) { vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0u); });
}

void VkInfer::recordFill(VkBuffer dst, VkDeviceSize bytes) {
    if (!recording_) throw std::runtime_error("VkInfer::recordFill outside beginFrame()/endFrame()");
    // Folded into the frame command buffer (no extra submit). A later dispatch
    // that reads `dst` carries a leading TRANSFER_WRITE->SHADER barrier.
    vkCmdFillBuffer(frameCb_, dst, 0, bytes, 0u);
}

void VkInfer::recordCopy(VkBuffer dst, VkBuffer src, VkDeviceSize bytes) {
    if (!recording_) throw std::runtime_error("VkInfer::recordCopy outside beginFrame()/endFrame()");
    // Make the producing compute writes to src available to the transfer read.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(frameCb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy region{};
    region.size = bytes;
    vkCmdCopyBuffer(frameCb_, src, dst, 1, &region);
}

void VkInfer::reserveFrameIO(VkDeviceSize uploadBytes, VkDeviceSize readbackBytes) {
    if (recording_) throw std::runtime_error("VkInfer::reserveFrameIO inside beginFrame()/endFrame()");
    // Growing frees the old buffer, which is safe here: the last frame that used
    // it has been waited on. Freeing a mapped allocation unmaps it.
    uploadBytes += 256;
    readbackBytes += 256;
    if (frameUpload_.capacity < uploadBytes) {
        ++layoutGen_;
        frameUpload_ = allocBuffer(uploadBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkCheck(vkMapMemory(device_, frameUpload_.memory, 0, VK_WHOLE_SIZE, 0, &frameUploadMapped_),
                "map frame upload staging");
    }
    if (frameReadback_.capacity < readbackBytes) {
        ++layoutGen_;
        // HOST_CACHED for the same reason as the one-shot readback staging.
        try {
            frameReadback_ = allocBuffer(readbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        } catch (const std::runtime_error&) {
            frameReadback_ = allocBuffer(readbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        vkCheck(vkMapMemory(device_, frameReadback_.memory, 0, VK_WHOLE_SIZE, 0, &frameReadbackMapped_),
                "map frame readback staging");
    }
}

VkDeviceSize VkInfer::recordUpload(VkBuffer dst, const void* data, VkDeviceSize bytes) {
    if (!recording_) throw std::runtime_error("VkInfer::recordUpload outside beginFrame()/endFrame()");
    const VkDeviceSize off = (frameUploadCursor_ + 15) & ~VkDeviceSize(15);
    if (off + bytes > frameUpload_.capacity)
        throw std::runtime_error("VkInfer::recordUpload: more than reserveFrameIO() reserved");
    // Host writes to coherent memory before the submit are visible to it.
    std::memcpy(static_cast<char*>(frameUploadMapped_) + off, data, size_t(bytes));
    frameUploadCursor_ = off + bytes;

    // dst may still be read by earlier work in this frame: order the copy after it.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(frameCb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy region{};
    region.srcOffset = off;
    region.size = bytes;
    vkCmdCopyBuffer(frameCb_, frameUpload_.buffer, dst, 1, &region);
    // The next dispatch's leading barrier makes the copy visible to its reads.
    return off;
}

void* VkInfer::frameUploadData(VkDeviceSize offset) {
    return static_cast<char*>(frameUploadMapped_) + offset;
}

VkDeviceSize VkInfer::recordReadback(VkBuffer src, VkDeviceSize bytes) {
    if (!recording_) throw std::runtime_error("VkInfer::recordReadback outside beginFrame()/endFrame()");
    const VkDeviceSize off = (frameReadbackCursor_ + 15) & ~VkDeviceSize(15);
    if (off + bytes > frameReadback_.capacity)
        throw std::runtime_error("VkInfer::recordReadback: more than reserveFrameIO() reserved");
    frameReadbackCursor_ = off + bytes;

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(frameCb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy region{};
    region.dstOffset = off;
    region.size = bytes;
    vkCmdCopyBuffer(frameCb_, src, frameReadback_.buffer, 1, &region);
    frameHasReadback_ = true;// endFrame() adds the transfer -> host barrier
    return off;
}

const void* VkInfer::frameReadback(VkDeviceSize offset) const {
    return static_cast<const char*>(frameReadbackMapped_) + offset;
}

void VkInfer::readback2(VkBuffer srcA, void* dstA, VkDeviceSize bytesA,
                        VkBuffer srcB, void* dstB, VkDeviceSize bytesB) {
    VkDeviceSize total = bytesA + bytesB;
    ensureReadbackStaging(total);
    oneShot([&](VkCommandBuffer cb) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy ra{};
        ra.size = bytesA;
        vkCmdCopyBuffer(cb, srcA, readbackStaging_.buffer, 1, &ra);
        VkBufferCopy rb{};
        rb.size = bytesB;
        rb.dstOffset = bytesA;
        vkCmdCopyBuffer(cb, srcB, readbackStaging_.buffer, 1, &rb);
    });
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, readbackStaging_.memory, 0, total, 0, &mapped), "map staging (readback2)");
    std::memcpy(dstA, mapped, bytesA);
    std::memcpy(dstB, static_cast<const char*>(mapped) + bytesA, bytesB);
    vkUnmapMemory(device_, readbackStaging_.memory);
}

VkPipe VkInfer::createPipe(const uint32_t* spv, size_t spvByteCount, uint32_t nSSBO, uint32_t pushBytes,
                           const char* name) {
    VkPipe p;
    p.nSSBO = nSSBO;
    p.pushBytes = pushBytes;
    p.name = name;

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spvByteCount;
    smci.pCode = spv;
    VkShaderModule module = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device_, &smci, nullptr, &module), "create shader module");

    std::vector<VkDescriptorSetLayoutBinding> binds(nSSBO);
    for (uint32_t i = 0; i < nSSBO; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = nSSBO;
    dlci.pBindings = binds.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &p.dsLayout), "create ds layout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = pushBytes;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.dsLayout;
    if (pushBytes > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    vkCheck(vkCreatePipelineLayout(device_, &plci, nullptr, &p.layout), "create pipeline layout");

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = p.layout;
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline),
            "create compute pipeline");
    vkDestroyShaderModule(device_, module, nullptr);
    return p;
}

void VkInfer::destroyPipe(VkPipe& p) {
    if (p.pipeline) vkDestroyPipeline(device_, p.pipeline, nullptr);
    if (p.layout) vkDestroyPipelineLayout(device_, p.layout, nullptr);
    if (p.dsLayout) vkDestroyDescriptorSetLayout(device_, p.dsLayout, nullptr);
    p = VkPipe{};
}

void VkInfer::resetDescriptorCache() {
    // Free all cached sets — their bound buffers are about to become stale.
    ++layoutGen_;
    dsCache_.clear();
    vkResetDescriptorPool(device_, descPool_, 0);
}

void VkInfer::beginFrame(uint64_t replayKey) {
    if (tsEnabled_) hostBegin_ = std::chrono::steady_clock::now();
    replayKey_ = 0;// the frame command buffer is re-recorded
    recordKey_ = replayKey;
    arenaCursor_ = 0;// rewind the activation pool for this inference
    frameUploadCursor_ = 0;
    frameReadbackCursor_ = 0;
    frameHasReadback_ = false;
    // NOTE: the descriptor pool is intentionally NOT reset — sets are cached and
    // reused across inferences (see dispatch()/dsCache_).
    if (frameCb_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmdPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkCheck(vkAllocateCommandBuffers(device_, &ai, &frameCb_), "allocate frame cmd buffer");
    }
    vkCheck(vkResetCommandBuffer(frameCb_, 0), "reset frame cmd buffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = replayKey ? 0u : VkCommandBufferUsageFlags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vkCheck(vkBeginCommandBuffer(frameCb_, &bi), "begin frame cmd buffer");
    recording_ = true;
    if (tsEnabled_ && tsPool_) {
        vkCmdResetQueryPool(frameCb_, tsPool_, 0, kMaxTimestamps);
        tsCount_ = 0;
        tsLabels_.clear();
        tsIsDispatch_.clear();
        markTimestamp("begin");
    }
}

void VkInfer::dispatch(const VkPipe& pipe, const std::vector<VkBuffer>& ssbos,
                       const void* push, uint32_t pushBytes,
                       uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!recording_) throw std::runtime_error("VkInfer::dispatch outside beginFrame()/endFrame()");
    if (static_cast<uint32_t>(ssbos.size()) != pipe.nSSBO)
        throw std::runtime_error("VkInfer::dispatch ssbo count mismatch");

    // Reuse the cached set for this (pipeline, buffers) combo if seen before — the
    // op sequence and its arena/weight buffers are identical every inference, so the
    // set only needs allocating + writing once.
    VkDescriptorSet set = VK_NULL_HANDLE;
    auto key = std::make_pair(pipe.pipeline, ssbos);
    auto cached = dsCache_.find(key);
    if (cached != dsCache_.end()) {
        set = cached->second;
    } else {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &pipe.dsLayout;
        vkCheck(vkAllocateDescriptorSets(device_, &dsai, &set), "allocate descriptor set");

        std::vector<VkDescriptorBufferInfo> infos(pipe.nSSBO);
        std::vector<VkWriteDescriptorSet> writes(pipe.nSSBO);
        for (uint32_t i = 0; i < pipe.nSSBO; ++i) {
            infos[i].buffer = ssbos[i];
            infos[i].offset = 0;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, pipe.nSSBO, writes.data(), 0, nullptr);
        dsCache_.emplace(std::move(key), set);
    }

    // Leading barrier: make prior compute writes / transfer uploads (earlier in
    // submission order, incl. one-shot uploads) visible to this dispatch.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(frameCb_,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(frameCb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(frameCb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
    if (pushBytes > 0 && push)
        vkCmdPushConstants(frameCb_, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
    vkCmdDispatch(frameCb_, gx, gy, gz);
    if (tsEnabled_ && tsPool_ && tsCount_ < kMaxTimestamps) {
        vkCmdWriteTimestamp(frameCb_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool_, tsCount_);
        tsLabels_.push_back(pipe.name);
        tsIsDispatch_.push_back(true);
        ++tsCount_;
    }
}

void VkInfer::endFrame() {
    if (!recording_) return;
    const bool prof = tsEnabled_ && tsPool_ && tsCount_ > 0;
    if (prof) markTimestamp("end");
    if (frameHasReadback_) {
        // Make the readback copies visible to the host reads after the wait.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(frameCb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkCheck(vkEndCommandBuffer(frameCb_), "end frame cmd buffer");
    recording_ = false;
    if (recordKey_) {
        replayKey_ = recordKey_;
        replayGen_ = layoutGen_;
        recordKey_ = 0;
    }
    submitAndWait_();
}

void VkInfer::replayFrame() {
    if (recording_ || !replayable(replayKey_)) throw std::runtime_error("VkInfer::replayFrame: no replayable frame");
    if (tsEnabled_) hostBegin_ = std::chrono::steady_clock::now();
    submitAndWait_();
}

void VkInfer::submitAndWait_() {
    const bool prof = tsEnabled_ && tsPool_ && tsCount_ > 0;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frameCb_;
    const auto hostSubmit = std::chrono::steady_clock::now();
    vkCheck(vkQueueSubmit(queue_, 1, &si, frameFence_), "submit frame");
    vkCheck(vkWaitForFences(device_, 1, &frameFence_, VK_TRUE, UINT64_MAX), "wait frame");
    vkCheck(vkResetFences(device_, 1, &frameFence_), "reset frame fence");
    if (tsEnabled_) {
        const auto hostDone = std::chrono::steady_clock::now();
        using msd = std::chrono::duration<double, std::milli>;
        std::cerr << "    [host] begin to submit " << msd(hostSubmit - hostBegin_).count() << " ms, submit to done "
                  << msd(hostDone - hostSubmit).count() << " ms\n";
    }

    if (prof) {
        std::vector<uint64_t> ts(tsCount_);
        if (vkGetQueryPoolResults(device_, tsPool_, 0, tsCount_, tsCount_ * sizeof(uint64_t),
                                  ts.data(), sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            auto ms = [&](uint32_t a, uint32_t b) { return double(ts[b] - ts[a]) * double(tsPeriodNs_) / 1e6; };
            std::cerr << "    [gpu] frame total: " << ms(0, tsCount_ - 1) << " ms\n";
            // Marks: the time since the previous mark.
            uint32_t lastMark = 0;
            for (uint32_t i = 1; i < tsCount_; ++i) {
                if (tsIsDispatch_[i]) continue;
                const double d = ms(lastMark, i);
                if (d > 0.05) std::cerr << "    [gpu]   " << tsLabels_[i] << ": " << d << " ms\n";
                lastMark = i;
            }
            // Kernels: a dispatch owns the time since the timestamp before it
            // (its leading barrier included), summed per kernel, largest first.
            std::map<std::string, std::pair<int, double>> kernels;
            for (uint32_t i = 1; i < tsCount_; ++i) {
                if (!tsIsDispatch_[i]) continue;
                auto& k = kernels[tsLabels_[i]];
                ++k.first;
                k.second += ms(i - 1, i);
            }
            std::vector<std::pair<std::string, std::pair<int, double>>> byTime(kernels.begin(), kernels.end());
            std::sort(byTime.begin(), byTime.end(),
                      [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
            for (const auto& [name, k] : byTime)
                std::cerr << "    [gpu] kernel " << name << " x" << k.first << ": " << k.second << " ms\n";
        }
    }
}

}// namespace rfdetr
