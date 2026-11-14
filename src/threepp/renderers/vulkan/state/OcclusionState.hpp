// OcclusionState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns occlusion culling: the cull pass and its Hi-Z pyramid, the occlusion render passes, and the per-mesh
// cull-bit ranges.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct OcclusionState {
        // ── Two-phase GPU occlusion culling (setOcclusionCulling) ───────────
        // Phase 1 draws last frame's visible set; a FARTHEST-depth pyramid
        // (occlHiz_, min-reduce) is
        // built mid-frame from that depth; the cull compute tests every
        // record's world AABB against it and phase 2 draws only the newly
        // visible (render passes A/B below — same framebuffer + pipelines,
        // load/store-op variants are render-pass compatible; MSAA gets its
        // own variant pair and the pyramid reduces the raw MS attachment's
        // samples at mip 0). Deformers always draw (their CPU AABB is stale
        // — the frustum-cull rule). Default OFF: the single-pass path
        // records byte-identically and occlHiz_'s image isn't even
        // allocated. Gate: no split-screen scissor.
        std::unique_ptr<vulkan::OcclusionCull> occl_;
        std::unique_ptr<vulkan::HiZPyramid>    occlHiz_;
        bool occlusionCullingEnabled_ = false;
        VkRenderPass occlRenderPassA_   = VK_NULL_HANDLE;// CLEAR + STORE, depth → sampleable
        VkRenderPass occlRenderPassB_   = VK_NULL_HANDLE;// LOAD, final layouts as the single pass
        VkRenderPass occlRenderPassAMS_ = VK_NULL_HANDLE;// MSAA siblings (created with the MS
        VkRenderPass occlRenderPassBMS_ = VK_NULL_HANDLE;// pass on the msaa toggle)
        // True while THIS frame's raster actually ran two-phase (consumers
        // in the same recordCommandBuffer body branch on it).
        bool occlActiveThisFrame_ = false;

        // Per-INSTANCE occlusion-cull bits. The two-phase cull's persistent
        // visBits history needs one bit per drawn instance — keying it by the
        // per-OBJECT stable id above made phase 1 all-or-nothing for an
        // InstancedMesh (every instance shared one bit, and phase 2's
        // per-instance results raced on it, so a 10k-tree field either drew
        // whole or not at all and effectively never culled). Each mesh
        // reserves a contiguous bit range [base, base+capacity) the first
        // time it is drawn (capacity = its full instance count, 1 for plain
        // meshes); an entry's bit = base + instanceIndex. If count() later
        // grows past the reservation the mesh is re-based — the abandoned
        // bits go stale, which at worst mispredicts phase 1 for one frame
        // (phase 2 recovers the same frame, the cull's standing guarantee).
        // Definition moved to VulkanSceneTypes.hpp.
        using OcclBitRange = vulkan::impl::OcclBitRange;
        std::unordered_map<unsigned int, OcclBitRange> occlBitRanges_;// Object3D::id -> range
        uint32_t occlBitDomain_ = 0;// one past the highest reserved bit
        // One-entry memo: an InstancedMesh's entries are contiguous in the
        // record loop, so a 100k-instance field pays ONE map lookup per
        // frame, not 100k (the same reason the type probes are cached on
        // the entry). Refreshed on every miss; a re-based range refreshes
        // it too since the miss path rewrites it.
        unsigned int occlBitMemoMeshId_ = 0;// Object3D ids start at 1 — 0 = empty
        OcclBitRange occlBitMemoRange_{0u, 0u};
    };

}// namespace threepp::vulkan::impl
