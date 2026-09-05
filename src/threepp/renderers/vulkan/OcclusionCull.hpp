// OcclusionCull — the compute side of two-phase GPU occlusion culling.
//
// Frame flow (recordCommandBuffer, gated on setOcclusionCulling; the raster
// pass itself is split by Impl into a STORE variant + a LOAD variant of
// the same framebuffer/pipelines — load/store ops don't break render-pass
// compatibility, so nothing graphics-side is duplicated):
//
//   1. recordFilter    — phase-1 indirect buffer: last frame's visible set
//                        (persistent visBits, one bit PER INSTANCE — a
//                        per-mesh base + instanceIndex from occlCullBitFor,
//                        so each instance of an InstancedMesh predicts
//                        independently; deformers always draw — their CPU
//                        AABB is stale).
//   2. raster pass A   — draws phase 1; depth ends sampleable.
//   3. farthest HiZ    — HiZPyramid::record(reduceMin=true) mid-frame.
//   4. recordCullTest  — every record's world AABB vs the pyramid; the
//                        NEWLY visible go to the phase-2 indirect buffer;
//                        visBits refreshed for next frame.
//   5. raster pass B   — draws phase 2 (loadOp LOAD).
//
// Records are copied with instanceCount zeroed, never compacted, so
// firstInstance (the DrawInfo index the vertex shader consumes) is
// untouched and the bucket offsets/counts the draw calls use stay valid
// for both phases. Camera cuts self-heal in ONE frame: a wrong phase-1 set
// leaves the pyramid sparse, the test then passes nearly everything into
// phase 2 — a slow frame, not a wrong one.
//
// The phase buffers and visBits are single (not per frame-in-flight):
// GPU-only, written and consumed inside one command buffer; the leading
// barriers' queue-scope orders cross-frame reuse (same rule as the DoF
// scratch). The CPU-written inputs (source commands + AABB meta) are
// per-frame mapped buffers.

#ifndef THREEPP_VULKAN_OCCLUSION_CULL_HPP
#define THREEPP_VULKAN_OCCLUSION_CULL_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class OcclusionCull {

    public:
        // Matches occl_cull.comp's CullMeta (std430, 32 B).
        struct CullMeta {
            float    aabbMin[3];
            uint32_t cullBit;// per-INSTANCE visBits index (Impl::occlCullBitFor)
            float    aabbMax[3];
            uint32_t flags;// bit 0 = always draw (deformers / missing bounds)
        };

        // Hands a stale shared phase/visBits Buffer to the renderer's frame-serial
        // retire queue instead of an immediate vmaDestroyBuffer. ensureCapacity
        // grows these SINGLE (non-per-fif) buffers mid-record, when the sibling
        // frame-in-flight may still reference the old handle on the GPU; retiring
        // defers the free until its fence has provably signaled (VulkanRetireQueue).
        // The lambda captures Impl and forwards to Impl::retire. Optional:
        // if unset, ensureCapacity falls back to vkDeviceWaitIdle + destroy.
        using RetireBufferFn = std::function<void(Buffer&&)>;

        OcclusionCull(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight,
                      RetireBufferFn retireFn = {});
        ~OcclusionCull();
        OcclusionCull(const OcclusionCull&) = delete;
        OcclusionCull& operator=(const OcclusionCull&) = delete;

        // Grow the per-frame meta buffer + the shared phase buffers to hold
        // `drawCount` records and the visBits store to hold `bitDomain` bits
        // (the cull-bit allocator's high-water mark), then (re)write this
        // frame's descriptor sets if any input changed (safe: the frame's
        // fence retired this fif's prior use). Call BEFORE filling metaPtr
        // each frame. A grown visBits buffer is re-armed ALL-VISIBLE inside
        // the next recordFilter (the fill needs a command buffer) — one
        // conservative frame, never a hole.
        struct FrameInputs {
            VkBuffer    srcCmds   = VK_NULL_HANDLE;// CPU-built indirect records
            VkBuffer    rasterCam = VK_NULL_HANDLE;// jittered-VP UBO
            VkImageView hizView   = VK_NULL_HANDLE;// farthest pyramid (all mips)
            VkSampler   hizSampler = VK_NULL_HANDLE;
        };
        void prepareFrame(uint32_t frame, uint32_t drawCount, uint32_t bitDomain,
                          const FrameInputs& in);

        // Persistently-mapped CullMeta array for this frame (valid after
        // prepareFrame; capacity >= drawCount).
        [[nodiscard]] CullMeta* metaPtr(uint32_t frame) { return metaPtrs_[frame]; }

        // Flush the first `drawCount` records written through metaPtr — call
        // once after the frame's write batch (non-coherent-heap portability;
        // no-op on coherent memory).
        void flushMeta(uint32_t frame, uint32_t drawCount);

        // Phase-1 filter dispatch. Leading barrier: prior frames' indirect
        // reads + this frame's host meta writes → compute.
        void recordFilter(VkCommandBuffer cb, uint32_t frame, uint32_t drawCount);

        // Occlusion-test dispatch (after the farthest-HiZ build; its
        // trailing barrier already covers the sampled reads). Trailing
        // barrier here: phase-2 writes + visBits → indirect reads.
        void recordCullTest(VkCommandBuffer cb, uint32_t frame, uint32_t drawCount,
                            uint32_t hizMips, VkExtent2D hizExtent);

        [[nodiscard]] VkBuffer phase1Buffer() const { return phase1_.handle; }
        [[nodiscard]] VkBuffer phase2Buffer() const { return phase2_.handle; }

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        RetireBufferFn retireFn_;// defers stale shared-buffer frees; see the ctor doc

        std::vector<Buffer>    metaBufs_;// [fif] host-mapped CullMeta[]
        std::vector<CullMeta*> metaPtrs_;
        Buffer phase1_{};// device-local, INDIRECT | STORAGE
        Buffer phase2_{};
        Buffer visBits_{};// 1 bit per instance (occlCullBitFor domain), init ALL-VISIBLE
        bool   visBitsNeedInit_ = true;// fresh/grown buffer → fill 0xFF in recordFilter

    public:
        // Forget what was visible last frame: the next filter starts from
        // all-visible, as a fresh buffer does. The bits are per-frame temporal
        // state like any history, and the renderer's resetTemporalHistory()
        // restarts them with the rest.
        void resetVisibility() { visBitsNeedInit_ = true; }

    private:
        uint32_t capacity_ = 0;// records the buffers currently hold

        // 1×1 GENERAL R32F stand-in for the pyramid in the filter set (the
        // real pyramid may not have been laid out yet when filter runs).
        Image2D dummyHiz_{};

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> filterSets_;// [fif] dst = phase1, hiz = dummy
        std::vector<VkDescriptorSet> cullSets_;  // [fif] dst = phase2, hiz = real

        // Change detection for prepareFrame's descriptor rewrites.
        struct CachedInputs {
            VkBuffer    srcCmds = VK_NULL_HANDLE;
            VkBuffer    rasterCam = VK_NULL_HANDLE;
            VkImageView hizView = VK_NULL_HANDLE;
            VkBuffer    meta = VK_NULL_HANDLE;
            VkBuffer    phase1 = VK_NULL_HANDLE;
            VkBuffer    phase2 = VK_NULL_HANDLE;
            VkBuffer    visBits = VK_NULL_HANDLE;// handle changes when the bit domain grows
        };
        std::vector<CachedInputs> cached_;// [fif]

        void createPipeline();
        void createDummyHiz();
        // Free a grown-out shared buffer safely: hand it to the retire queue
        // (deferred until its last in-flight frame's fence signals) or, if no
        // callback was wired, fall back to a full device drain + destroy.
        void retireBuffer(Buffer& b);
        void ensureCapacity(uint32_t frame, uint32_t drawCount, uint32_t bitDomain);
        void rewriteSets(uint32_t frame, const FrameInputs& in);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_OCCLUSION_CULL_HPP
