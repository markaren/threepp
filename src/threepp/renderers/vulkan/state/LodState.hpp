// LodState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns auto-LOD: the background worker thread and its job and result queues, the level budget, the landing
// batches, and the per-entry LOD fade and level buffers.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct LodState {
        // ── Automatic mesh LOD (setAutoLod; ON by default) ──────────────────
        // Measured: Bistro (per-pixel-bound worst case) neutral, fjord flight
        // +32% FPS, quality below animation noise, switch frames stall-free
        // (geomDescs ring). setAutoLod(false) remains as the manual override /
        // debug escape.
        bool autoLod_ = true;
        // Screen-space error budget τ for the selection pass, in RENDER-scale
        // pixels (setAutoLodError). The pass picks the coarsest level whose
        // projected simplification error stays under τ; the raise-hysteresis
        // margin scales with it (0.8·τ). 0.75 px = the validated visually-
        // lossless default; larger trades silhouette error for triangle
        // throughput (1.5-2 px is a sensible perf setting under TAA).
        float lodErrorPx_ = 0.75f;
        // Set by the selection pass in ensureSceneBuilt (VulkanCoreScene.cpp)
        // when ANY entry's chosen level changed this frame; read right after
        // to fold into the same "force a full TLAS rebuild" trigger the
        // deformer paths already use (blasDeformed) — an AS-reference change
        // per VkAccelerationStructureInstanceKHR is only unambiguously legal
        // under MODE_BUILD, not the incremental MODE_UPDATE refit. Also marks
        // geomDescsDirty_ whole-array so RT secondary rays (reflections/GI/
        // lidar/probe update) read the SAME index buffer the swapped BLAS
        // was built from — without it, gl_PrimitiveID from a hit against a
        // coarser BLAS would misindex the still-LOD0
        // GeometryDesc::indexAddress.
        bool lodChangedThisFrame_ = false;

        // Manual threepp::LOD subtrees, rebuilt every FULL scene expansion
        // (cleared at the start of the traverseVisible walk). A mesh entry
        // under one of these is exempt from auto-LOD — its levels are
        // already hand-authored; auto-LOD swapping index buffers underneath
        // an author-selected discrete level would fight that choice. Membership
        // test walks Object3D::parent by raw pointer — no dynamic_cast.
        std::unordered_set<const Object3D*> manualLodLevelRoots_;
        // Background chain-generation worker: a single lazily-started
        // std::thread (not a pool — LOD job volume is low: one job per
        // eligible unique geometry, ever, per geomVersion) draining a
        // mutex+condvar job queue. Jobs snapshot (copy) position/index data
        // at enqueue time, so the worker never touches live BufferGeometry /
        // Vulkan state — no lifetime races with a scene the main thread
        // mutates or tears down while a job runs. Joined in ~Impl before
        // anything Vulkan-related is torn down (the worker is pure CPU).
        std::thread lodWorker_;
        std::mutex lodJobMutex_;
        std::condition_variable lodJobCv_;
        bool lodWorkerStop_ = false;
        // Definition moved to VulkanGeometryState.hpp.
        using LodJob = vulkan::impl::LodJob;
        std::deque<LodJob> lodJobQueue_;
        // Definition moved to VulkanGeometryState.hpp.
        using LodResult = vulkan::impl::LodResult;
        std::mutex lodResultMutex_;
        std::deque<LodResult> lodResultQueue_;
        // Signalled once per result. Only a pinned-clock drain waits on it
        // (drainLodResults): there the frame a chain lands on must not
        // depend on how fast the worker ran.
        std::condition_variable lodResultCv_;
        // Running byte totals for every finalized LOD index buffer + BLAS
        // storage, updated as chains finalize / are evicted — cheap O(1)
        // bookkeeping instead of a full blasCache walk every frame. Gates
        // the hard cap on new chain GENERATION (already-finalized chains are
        // never evicted early to enforce it — see setAutoLod's doc comment).
        uint64_t lodIndexBytes_ = 0;
        uint64_t lodBlasBytes_  = 0;
        bool lodBudgetWarned_ = false;
        static constexpr uint64_t kLodByteBudget = 256ull * 1024ull * 1024ull;
        // Unique-geometry chain-state counters (NOT per-entry), maintained
        // at enqueue/drain time — surfaced via autoLodStats().
        uint32_t lodChainsReadyCount_  = 0;
        uint32_t lodChainsQueuedCount_ = 0;
        // setReplayableLodSchedule: with the clock pinned, drainLodResults
        // waits for the worker (on) or takes what it has (off).
        bool lodReplayableSchedule_ = true;
        // AutoLodStats::drainWaitMs: the time those waits took, cumulative.
        double lodDrainWaitMs_ = 0.0;
        VulkanRenderer::AutoLodStats autoLodStats_{};

        // The drain used to end in a submit + vkQueueWaitIdle, i.e. a wait
        // for up to both frames in flight inside render() on every
        // frame a chain landed (the twin: 18 ms of a 24 ms ensureSceneBuilt
        // while its chains landed). Now the builds go out in a fenced,
        // unwaited submit and the chain lands kLodLandDrains drains later,
        // when that fence has signalled (the frame fences between make the
        // wait at landing a formality); compaction is a second such submit.
        // Each landing is a fixed drain count after its submit, so the
        // schedule is the same on the wall clock and under setSimTime.
        using LodLandingBatch = vulkan::impl::LodLandingBatch;
        static constexpr uint64_t kLodLandDrains = kFramesInFlight + 1;
        uint64_t lodDrainSerial_ = 0;
        uint64_t lodBatchSerial_ = 0;
        std::vector<LodLandingBatch> lodLanding_;

        // The per-entry LOD fade words (entryLodFade_) for the GPU producers:
        // the DrawInfo flag bits and the TLAS mask of a crossfading instance.
        // Per frame-in-flight, host-mapped, re-sent when lodFadeVersion_ moved:
        // the spans whose words changed since that slot's last upload
        // (lodFadeDirty_, as motionDirty_ does for the motion matrices), whole
        // after a rebuild, a slot edit or growth. Bound by the sets
        // prepareDrawGen / prepareTlasGen rewrite every frame, so growth is a
        // plain swap.
        std::array<DescDirtyRanges, kFramesInFlight> lodFadeDirty_{};

        std::array<Buffer, kFramesInFlight> lodFadeBuffers_{};
        std::array<uint32_t, kFramesInFlight> lodFadeUploaded_{};
        std::array<size_t, kFramesInFlight> lodFadeUploadedCount_{};
        uint32_t lodFadeVersion_ = 1;

        // MeshEntry::lodLevel for the GPU producers, a byte per entry: the
        // CPU's selection, read by the spans with an auto-LOD chain. Gathered
        // from the spans with a level above 0 (EntrySpan::lodNonZero), re-sent
        // when lodLevelVersion_ moved: a selection that changed a level, or
        // entries laid out anew.
        std::array<Buffer, kFramesInFlight> lodLevelBuffers_{};
        std::array<uint32_t, kFramesInFlight> lodLevelUploaded_{};
        std::array<size_t, kFramesInFlight> lodLevelUploadedCount_{};
        uint32_t lodLevelVersion_ = 1;
        std::vector<uint8_t> lodLevelScratch_;

        // Dithered LOD crossfade (Mesh::setFade, InstancedMesh::setFadeAt).
        // One word per entry in the high-half layout of vulkan_shared.h
        // (kLodFade*), 0 for no fade: buildIndirectDrawData ORs it into
        // DrawInfo.flags, where gbuffer.frag dithers against it. The rest of
        // the renderer (TLAS masks, shadow-map casters, overlay occluders)
        // asks lodFadeCasts, so a crossfading pair is ONE level to them.
        // Kept beside the entries rather than in the MaterialDescs: fades
        // change every frame of a transition on scattered instances, and a
        // 4-byte word beats re-flushing 624-byte descs range by range.
        std::vector<uint32_t> entryLodFade_;
        // Bumped whenever an entry's lodFadeCasts flips, so the cached static
        // shadow cascades (shadowStaticHash) redraw with the other level.
        uint32_t lodFadeCastEpoch_ = 0;
    };

}// namespace threepp::vulkan::impl
