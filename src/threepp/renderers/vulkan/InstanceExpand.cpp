#include "threepp/renderers/vulkan/InstanceExpand.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/instance_expand.comp.spv.h"

#include <algorithm>
#include <cstring>

namespace threepp::vulkan {

    namespace {
        struct ExpandPc {
            uint32_t spanCount;
            uint32_t totalWork;
        };
        static_assert(sizeof(ExpandPc) == 8, "instance_expand push-constant drift");

        constexpr VkDeviceSize kMatBytes   = 16 * sizeof(float);// one instance matrix
        constexpr uint32_t     kMinSpans   = 64;
        constexpr uint32_t     kMinMats    = 4096;
        constexpr uint32_t     kMinEntries = 4096;
    }// namespace

    InstanceExpand::InstanceExpand(VulkanContext& ctx, uint32_t framesInFlight,
                                   RetireBufferFn retireFn)
        : ctx_(ctx), framesInFlight_(framesInFlight), retireFn_(std::move(retireFn)) {
        spanBufs_.resize(framesInFlight_);
        spanPtrs_.resize(framesInFlight_, nullptr);
        matBufs_.resize(framesInFlight_);
        matrixPtrs_.resize(framesInFlight_, nullptr);
        matFresh_.resize(framesInFlight_, 0u);
        cached_.resize(framesInFlight_);
        createPipeline();
    }

    InstanceExpand::~InstanceExpand() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        for (auto& b : spanBufs_) destroyBuffer(ctx_.allocator(), b);
        for (auto& b : matBufs_)  destroyBuffer(ctx_.allocator(), b);
        destroyBuffer(ctx_.allocator(), world_);
    }

    void InstanceExpand::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 3;
        dlci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(instance_expand)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(ExpandPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(instance_expand)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kInstanceExpandCompSpv);
        smci.pCode    = kInstanceExpandCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod),
              "vkCreateShaderModule(instance_expand)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(instance_expand)");
        vkDestroyShaderModule(d, mod, nullptr);

        VkDescriptorPoolSize size{};
        size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = framesInFlight_ * 3;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &size;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(instance_expand)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        dsai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(d, &dsai, sets_.data()),
              "vkAllocateDescriptorSets(instance_expand)");
    }

    void InstanceExpand::retireBuffer(Buffer& b) {
        if (b.handle == VK_NULL_HANDLE) return;
        if (retireFn_) {
            retireFn_(std::move(b));// zeroes b; the reallocation can't double-free
        } else {
            vkDeviceWaitIdle(ctx_.device());
            destroyBuffer(ctx_.allocator(), b);
        }
    }

    void InstanceExpand::ensureCapacity(uint32_t frame, uint32_t spanCount,
                                        uint32_t matrixCount, uint32_t entryCount) {
        // Per-frame SpanDesc pool. Host-mapped and rewritten whole each frame
        // (spans are O(meshes), not O(instances)), so a plain destroy on growth
        // is safe: this slot's fence has signaled.
        const VkDeviceSize spanBytes = VkDeviceSize(std::max(spanCount, 1u)) * sizeof(SpanDesc);
        if (spanBufs_[frame].size < spanBytes) {
            destroyBuffer(ctx_.allocator(), spanBufs_[frame]);
            const VkDeviceSize cap = std::max<VkDeviceSize>(spanBytes * 2,
                                                           kMinSpans * sizeof(SpanDesc));
            spanBufs_[frame] = createBuffer(ctx_.allocator(), ctx_.device(), cap,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VMA_MEMORY_USAGE_AUTO,
                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), spanBufs_[frame].alloc, &info);
            spanPtrs_[frame] = static_cast<SpanDesc*>(info.pMappedData);
        }

        // Per-frame instance-matrix pool. Written per SPAN, version-gated, so a
        // fresh allocation must be announced (takeMatrixPoolFresh) — the host's
        // "span S is already at version V in this slot" state describes the
        // buffer that just went away.
        const VkDeviceSize matBytes = VkDeviceSize(std::max(matrixCount, 1u)) * kMatBytes;
        if (matBufs_[frame].size < matBytes) {
            destroyBuffer(ctx_.allocator(), matBufs_[frame]);
            const VkDeviceSize cap = std::max<VkDeviceSize>(matBytes * 2, kMinMats * kMatBytes);
            matBufs_[frame] = createBuffer(ctx_.allocator(), ctx_.device(), cap,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VMA_MEMORY_USAGE_AUTO,
                                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                   VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), matBufs_[frame].alloc, &info);
            matrixPtrs_[frame] = static_cast<float*>(info.pMappedData);
            matFresh_[frame]   = 1u;
        }

        // Shared output. Indexed by ENTRY index (so a later stage can read it
        // with the same index space DrawInfo/TLAS use), device-local, and
        // TRANSFER_SRC for the verification readback. Single, not per-fif:
        // written and consumed inside one command buffer — but growth can land
        // while a sibling frame still names the old handle, hence the retire
        // queue rather than a destroy.
        if (worldCapacity_ < entryCount) {
            retireBuffer(world_);
            const uint32_t cap = std::max(entryCount * 2u, kMinEntries);
            world_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                  VkDeviceSize(cap) * kMatBytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO);
            worldCapacity_ = cap;
        }
    }

    void InstanceExpand::rewriteSet(uint32_t frame) {
        auto& c = cached_[frame];
        if (c.spans == spanBufs_[frame].handle && c.mats == matBufs_[frame].handle &&
            c.world == world_.handle)
            return;
        c = {spanBufs_[frame].handle, matBufs_[frame].handle, world_.handle};

        VkDescriptorBufferInfo bi[3]{
                {spanBufs_[frame].handle, 0, VK_WHOLE_SIZE},
                {matBufs_[frame].handle, 0, VK_WHOLE_SIZE},
                {world_.handle, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = sets_[frame];
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &bi[i];
        }
        vkUpdateDescriptorSets(ctx_.device(), 3, w, 0, nullptr);
    }

    void InstanceExpand::prepareFrame(uint32_t frame, uint32_t spanCount,
                                      uint32_t matrixCount, uint32_t entryCount) {
        ensureCapacity(frame, spanCount, matrixCount, entryCount);
        rewriteSet(frame);
    }

    bool InstanceExpand::takeMatrixPoolFresh(uint32_t frame) {
        const bool f = matFresh_[frame] != 0u;
        matFresh_[frame] = 0u;
        return f;
    }

    void InstanceExpand::flushSpans(uint32_t frame, uint32_t spanCount) {
        if (spanCount == 0u || spanBufs_[frame].alloc == VK_NULL_HANDLE) return;
        flushHostWrites(ctx_.allocator(), spanBufs_[frame].alloc, 0,
                        VkDeviceSize(spanCount) * sizeof(SpanDesc));
    }

    void InstanceExpand::flushMatrices(uint32_t frame, uint32_t firstMatrix, uint32_t matrixCount) {
        if (matrixCount == 0u || matBufs_[frame].alloc == VK_NULL_HANDLE) return;
        flushHostWrites(ctx_.allocator(), matBufs_[frame].alloc,
                        VkDeviceSize(firstMatrix) * kMatBytes,
                        VkDeviceSize(matrixCount) * kMatBytes);
    }

    void InstanceExpand::record(VkCommandBuffer cb, uint32_t frame,
                                uint32_t spanCount, uint32_t totalWork) {
        if (spanCount == 0u || totalWork == 0u || pipe_ == VK_NULL_HANDLE) return;

        // Host writes (spans + matrices) → compute reads, and the previous
        // frame's compute writes into the SHARED output → this frame's writes
        // (WAW on a single buffer). Nothing else reads the output yet, so this
        // is the whole synchronization story for stage 1; the barrier the
        // consumers will need arrives with them.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &dep);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &sets_[frame], 0, nullptr);
        const ExpandPc pc{spanCount, totalWork};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (totalWork + 63u) / 64u, 1, 1);

        // Trailing: the readback path is the only consumer, and it drains the
        // device and issues its own barrier. A stage-2 consumer adds its own.
    }

    bool InstanceExpand::readWorldMatrices(VkCommandPool cmdPool, VkQueue queue,
                                           uint32_t entryCount, std::vector<float>& out) {
        if (entryCount == 0u || world_.handle == VK_NULL_HANDLE) return false;
        const uint32_t n = std::min(entryCount, worldCapacity_);
        const VkDeviceSize bytes = VkDeviceSize(n) * kMatBytes;

        check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(instanceExpandCheck)");

        Buffer staging = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb),
              "alloc one-shot cb(instanceExpandCheck)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(instanceExpandCheck)");
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &dep);
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cb, world_.handle, staging.handle, 1, &region);
        check(vkEndCommandBuffer(cb), "end one-shot cb(instanceExpandCheck)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "submit one-shot(instanceExpandCheck)");
        check(vkQueueWaitIdle(queue), "wait one-shot(instanceExpandCheck)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool, 1, &cb);

        invalidateHostReads(ctx_.allocator(), staging.alloc, 0, bytes);
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(ctx_.allocator(), staging.alloc, &info);
        out.resize(size_t(n) * 16u);
        std::memcpy(out.data(), info.pMappedData, size_t(bytes));
        destroyBuffer(ctx_.allocator(), staging);
        return true;
    }

}// namespace threepp::vulkan
