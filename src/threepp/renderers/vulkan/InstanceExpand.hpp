// InstanceExpand — the compute side of GPU per-instance world matrices.
//
// Stage 1 of plans/gpu-driven-instances.md, and deliberately only stage 1:
// this pass PRODUCES what the CPU already produces and no consumer reads it.
// Its reason to exist is the comparison — VulkanRenderer::instanceExpandCheck
// copies the output back and checks it against MeshEntry::worldMatrix — so
// that when DrawInfo, motion, cull and the TLAS descriptors move onto the GPU
// in stages 2-5 they move onto a producer already proven equal to the CPU.
//
// Frame flow (Impl::prepareInstanceExpansion + Impl::recordInstanceExpansion):
//
//   1. prepareFrame — grow this frame-in-flight's SpanDesc + instance-matrix
//      pools and the shared world-matrix buffer, then rewrite the frame's
//      descriptor set if any handle moved. Runs AFTER the per-frame fence wait
//      and BEFORE recording, which is what makes writing this slot's buffers
//      and its descriptor set safe (invariant 4 — the VUID-03047 zone).
//   2. the host fills spanPtr()/matrixPtr() (version-gated; see EntrySpan's
//      meshWorld / instMatVersion contract) and flushes.
//   3. record — one dispatch over the summed instance count.
//
// Buffer ownership follows OcclusionCull exactly: the CPU-written inputs are
// per-frame-in-flight host-mapped buffers, and the GPU-only output is SINGLE
// (written and read inside one command buffer; the leading barrier's queue
// scope orders cross-frame reuse). Growth never destroys in place — a grown-out
// shared buffer goes to the renderer's frame-serial retire queue, because
// entry-list churn that frees a buffer a sibling frame still names is a device
// drain at best and a device-lost at worst (invariant 5).

#ifndef THREEPP_VULKAN_INSTANCE_EXPAND_HPP
#define THREEPP_VULKAN_INSTANCE_EXPAND_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class InstanceExpand {

    public:
        // Matches instance_expand.comp's SpanDesc (std430, 80 B).
        struct SpanDesc {
            float    world[16];// mesh->matrixWorld the CPU baked the entries from
            uint32_t firstEntry;
            uint32_t count;
            uint32_t matBase; // first matrix in the instance-matrix pool
            uint32_t workBase;// exclusive prefix sum of count
        };
        static_assert(sizeof(SpanDesc) == 80, "instance_expand SpanDesc drift");

        // Defers the free of a grown-out SHARED buffer (the world-matrix
        // output) to the renderer's frame-serial retire queue. Same contract
        // and same reason as OcclusionCull::RetireBufferFn.
        using RetireBufferFn = std::function<void(Buffer&&)>;

        InstanceExpand(VulkanContext& ctx, uint32_t framesInFlight,
                       RetireBufferFn retireFn = {});
        ~InstanceExpand();
        InstanceExpand(const InstanceExpand&) = delete;
        InstanceExpand& operator=(const InstanceExpand&) = delete;

        // Grow this frame's pools to hold `spanCount` spans and `matrixCount`
        // instance matrices, and the shared output to hold `entryCount`
        // matrices; then rewrite this frame's descriptor set if anything moved.
        void prepareFrame(uint32_t frame, uint32_t spanCount,
                          uint32_t matrixCount, uint32_t entryCount);

        // True exactly once per (re)allocation of THIS frame's instance-matrix
        // pool: the new allocation holds garbage, so the caller's per-span
        // "already uploaded at version V" bookkeeping for this frame is void
        // and every span must be re-sent. Clears the flag.
        [[nodiscard]] bool takeMatrixPoolFresh(uint32_t frame);

        // Persistently-mapped views of this frame's pools (valid after
        // prepareFrame). matrixPtr is 16 floats per matrix, column-major —
        // the same layout as FloatBufferAttribute's array, so a span's block
        // is one memcpy.
        [[nodiscard]] SpanDesc* spanPtr(uint32_t frame) { return spanPtrs_[frame]; }
        [[nodiscard]] float*    matrixPtr(uint32_t frame) { return matrixPtrs_[frame]; }

        // Flush host writes (no-ops on coherent memory; portability, same as
        // OcclusionCull::flushMeta). Matrix range is in MATRICES, not bytes.
        void flushSpans(uint32_t frame, uint32_t spanCount);
        void flushMatrices(uint32_t frame, uint32_t firstMatrix, uint32_t matrixCount);

        // One dispatch over totalWork instances. Leading barrier: host writes
        // + any prior frame's use of the shared output → compute.
        void record(VkCommandBuffer cb, uint32_t frame,
                    uint32_t spanCount, uint32_t totalWork);

        [[nodiscard]] VkBuffer worldBuffer() const { return world_.handle; }
        // Matrices the shared output currently holds (>= the entryCount the
        // last prepareFrame asked for).
        [[nodiscard]] uint32_t worldCapacity() const { return worldCapacity_; }

        // Copy the first `entryCount` output matrices back to the host, 16
        // floats each. DRAINS THE DEVICE — a debug/test path only (it backs
        // VulkanRenderer::instanceExpandCheck). Returns false when there is
        // nothing to read.
        bool readWorldMatrices(VkCommandPool cmdPool, VkQueue queue,
                               uint32_t entryCount, std::vector<float>& out);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;
        RetireBufferFn retireFn_;

        std::vector<Buffer>    spanBufs_;// [fif] host-mapped SpanDesc[]
        std::vector<SpanDesc*> spanPtrs_;
        std::vector<Buffer>    matBufs_;// [fif] host-mapped float[16·N]
        std::vector<float*>    matrixPtrs_;
        std::vector<uint8_t>   matFresh_;// [fif] pool was just (re)allocated
        Buffer   world_{};                // device-local, STORAGE | TRANSFER_SRC
        uint32_t worldCapacity_ = 0;      // in matrices

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [fif]

        struct CachedInputs {
            VkBuffer spans = VK_NULL_HANDLE;
            VkBuffer mats  = VK_NULL_HANDLE;
            VkBuffer world = VK_NULL_HANDLE;
        };
        std::vector<CachedInputs> cached_;// [fif]

        void createPipeline();
        void retireBuffer(Buffer& b);
        void ensureCapacity(uint32_t frame, uint32_t spanCount,
                            uint32_t matrixCount, uint32_t entryCount);
        void rewriteSet(uint32_t frame);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_INSTANCE_EXPAND_HPP
