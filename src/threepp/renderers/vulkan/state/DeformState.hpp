// DeformState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns GPU deformation recorded into the frame: the skinning and tet-skinning pipelines and the per-frame
// deform, refit and resync queues.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_DEFORMSTATE_HPP
#define THREEPP_VULKAN_STATE_DEFORMSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct DeformState {
        // List of SkinnedMeshState pointers whose bones changed this frame.
        // ensureSceneBuilt populates this (uploads bone matrices to the GPU
        // buffer); recordCommandBuffer consumes it by recording skinning
        // dispatch + BLAS rebuild into the main per-frame cmd buffer with
        // barriers. Cleared at the end of recordCommandBuffer.
        std::vector<SkinnedMeshState*> pendingSkinnedRebuilds_;

        std::vector<TetMeshState*> pendingTetRebuilds_;

        // GrassMesh deforms queued in ensureSceneBuilt, recorded into the frame
        // command buffer in recordCommandBuffer (no blocking submit) — same
        // pattern as pendingSkinnedRebuilds_. Cleared at end of recordCommandBuffer.
        std::vector<std::pair<GrassMesh*, GrassMeshState*>> pendingGrassDeforms_;
        // Grass fields this frame's slot edit placed (new, or attached again):
        // their deform runs, but like every entry the structural rebuild
        // places they carry no moved bit on that frame (a moved bit is a
        // shorter history for thirty frames). Cleared with the deform queue.
        std::vector<const Mesh*> grassPlacedThisFrame_;

        // Graduated per-frame dynamic plain meshes (BlasRecord::perFrameDynamic)
        // whose attributes changed this frame — staging upload + GPU copy +
        // batched BLAS refit recorded into the frame cb by
        // recordDynamicGeomRefits, same pattern as pendingSkinnedRebuilds_.
        // Cleared there.
        std::vector<vulkan::impl::GeomRefreshOp> pendingDynamicGeomRefits_;
        // Graduated records whose first CLEAN frame follows a dirty run —
        // recordDynamicGeomRefits copies vertex→prevVertex once so their
        // motion vectors collapse back to zero (frame-cb twin of the
        // prevVertexResyncPending pass).
        std::vector<vulkan::impl::BlasRecord*> pendingDynamicPrevResyncs_;

        // DisplacedMesh (FFT water) deforms queued in ensureSceneBuilt and
        // recorded into the frame command buffer by recordCommandBuffer — the
        // same no-mid-frame-submit pattern as pendingGrassDeforms_. The float
        // is the elapsed-seconds timestamp captured at stage time (drives the
        // dynamic spectrum). The old path recorded the FFT→displace→BLAS chain
        // into a one-shot and BLOCKED on its completion every frame — a full
        // CPU⇄GPU sync whose stall scaled with everything already in flight
        // (tens of ms on large scenes). CPU height-field mirrors now read the
        // readback buffers at stage time instead (one frame late).
        std::vector<std::tuple<DisplacedMesh*, DisplacedMeshState*, float>> pendingDisplacedDeforms_;

        // ── GPU skinning compute pipeline ──────────────────────────────────
        // Replaces the cpuSkin() loop. One dispatch per skinned mesh per
        // frame when bones change, recorded into the main per-frame command
        // buffer alongside the BLAS rebuild. Frees the CPU thread from
        // the synchronous per-vertex linear-blend math + blocking BLAS
        // submit (was ~10 ms / frame on stormtrooper-density meshes).
        //
        // Pipeline + descriptor pool live in vulkan/SkinningPipeline.{hpp,cpp};
        // per-mesh descriptor sets (state->skinDescSet) live in SkinnedMeshState.
        std::unique_ptr<vulkan::SkinningPipeline> skinning_;
        // GPU tetrahedral-skinning pipeline for PhysX soft bodies (mirrors skinning_).
        std::unique_ptr<vulkan::TetSkinningPipeline> tetSkinning_;

        VkDeviceSize deformHostBytes_ = 0;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_DEFORMSTATE_HPP
