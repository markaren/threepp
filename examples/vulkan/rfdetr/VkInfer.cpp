#include "VkInfer.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
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

VkInfer::VkInfer(VkDevice device, VkPhysicalDevice phys, VkQueue queue, uint32_t queueFamily)
    : device_(device), phys_(phys), queue_(queue), queueFamily_(queueFamily) {

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = queueFamily_;
    vkCheck(vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_), "create command pool");
    // Separate pool for one-shot uploads/readbacks so they never share state with
    // the frame command buffer (which may be mid-recording when an upload runs).
    vkCheck(vkCreateCommandPool(device_, &cpci, nullptr, &oneShotPool_), "create one-shot command pool");

    // One descriptor set per dispatch in a frame; a YOLOv8n pass is a few
    // hundred dispatches. Reset per beginFrame().
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

    tsEnabled_ = std::getenv("RTDETR_PROFILE") != nullptr;
    if (tsEnabled_) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        tsPeriodNs_ = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 64;
        if (vkCreateQueryPool(device_, &qpci, nullptr, &tsPool_) != VK_SUCCESS) tsPool_ = VK_NULL_HANDLE;
    }
}

VkInfer::~VkInfer() {
    arenaSlots_.clear();// free pooled buffers before pools/device teardown
    if (tsPool_) vkDestroyQueryPool(device_, tsPool_, nullptr);
    if (descPool_) vkDestroyDescriptorPool(device_, descPool_, nullptr);
    if (oneShotPool_) vkDestroyCommandPool(device_, oneShotPool_, nullptr);
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
}

void VkInfer::markTimestamp(const char* label) {
    if (!tsEnabled_ || !tsPool_ || !recording_ || tsCount_ >= 64) return;
    vkCmdWriteTimestamp(frameCb_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool_, tsCount_);
    tsLabels_.push_back(label);
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
    } else if (arenaSlots_[arenaCursor_].capacity < bytes) {
        // Slot exists but is too small (only happens if a shape grows): regrow.
        arenaSlots_[arenaCursor_] = allocBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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

void VkInfer::readback(VkBuffer src, void* dst, VkDeviceSize bytes) {
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

VkPipe VkInfer::createPipe(const uint32_t* spv, size_t spvByteCount, uint32_t nSSBO, uint32_t pushBytes) {
    VkPipe p;
    p.nSSBO = nSSBO;
    p.pushBytes = pushBytes;

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
    dsCache_.clear();
    vkResetDescriptorPool(device_, descPool_, 0);
}

void VkInfer::beginFrame() {
    arenaCursor_ = 0;// rewind the activation pool for this inference
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
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(frameCb_, &bi), "begin frame cmd buffer");
    recording_ = true;
    if (tsEnabled_ && tsPool_) {
        vkCmdResetQueryPool(frameCb_, tsPool_, 0, 64);
        tsCount_ = 0;
        tsLabels_.clear();
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
}

void VkInfer::endFrame() {
    if (!recording_) return;
    const bool prof = tsEnabled_ && tsPool_ && tsCount_ > 0;
    if (prof) markTimestamp("end");
    vkCheck(vkEndCommandBuffer(frameCb_), "end frame cmd buffer");
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frameCb_;
    vkCheck(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "submit frame");
    vkCheck(vkQueueWaitIdle(queue_), "wait frame");
    recording_ = false;

    if (prof) {
        std::vector<uint64_t> ts(tsCount_);
        if (vkGetQueryPoolResults(device_, tsPool_, 0, tsCount_, tsCount_ * sizeof(uint64_t),
                                  ts.data(), sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            double total = double(ts[tsCount_ - 1] - ts[0]) * double(tsPeriodNs_) / 1e6;
            std::cerr << "    [gpu] frame total: " << total << " ms\n";
            for (uint32_t i = 1; i < tsCount_; ++i) {
                double ms = double(ts[i] - ts[i - 1]) * double(tsPeriodNs_) / 1e6;
                if (ms > 0.30) std::cerr << "    [gpu]   " << tsLabels_[i] << ": " << ms << " ms\n";
            }
        }
    }
}

}// namespace rfdetr
