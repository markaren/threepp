// GpuDrivenState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns GPU-driven work generation: instance expansion, the draw generator and the TLAS generator, with their
// spans, work counts and self-checks.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_GPUDRIVENSTATE_HPP
#define THREEPP_VULKAN_STATE_GPUDRIVENSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct GpuDrivenState {
        // ── GPU per-instance world matrices (stage 1) ───────────────────────
        // instance_expand.comp recomputes, per InstancedMesh span, exactly what
        // ensureSceneBuilt's lean refresh bakes into MeshEntry::worldMatrix.
        // Its readers are draw_gen.comp (gpuDrawGen_, below) and tlas_gen.comp
        // (gpuTlasGen_); without them it is added cost, and
        // VulkanRenderer::instanceExpandCheck verifies that it agrees with the
        // CPU.
        std::unique_ptr<vulkan::InstanceExpand> instExpand_;
        // OFF by default, as are its readers: on its own the pass is measured
        // slower (~+1.0 ms wall at 85k grains, 8 of 8 interleaved A/B pairs).
        // setGpuInstanceExpansion(true) opts in, which the draw and TLAS
        // generation and VulkanInstanceExpand_test need.
        bool gpuInstanceExpand_ = false;// setGpuInstanceExpansion; the A/B lever
        // The GPU path's span list: an index into entrySpans_ plus the two
        // prefix sums the shader needs. Instanced spans only. Rebuilt (and
        // compared) every frame — it is O(spans), not O(instances), and a diff
        // is what tells the per-fif upload bookkeeping below that its
        // per-span slots no longer mean the same thing.
        struct InstExpandSpan {
            uint32_t spanIdx  = 0;// index into entrySpans_
            uint32_t count    = 0;// instances actually uploaded (clamped to the attribute)
            uint32_t matBase  = 0;// first matrix in the pool
            uint32_t workBase = 0;// exclusive prefix sum of count
            bool operator==(const InstExpandSpan& o) const noexcept {
                return spanIdx == o.spanIdx && count == o.count &&
                       matBase == o.matBase && workBase == o.workBase;
            }
        };
        std::vector<InstExpandSpan> instExpandSpans_, instExpandScratch_;

        // ── GPU draw records (stage 2; DrawGen.hpp, draw_gen.comp) ─────────
        // OFF by default, and only with the expansion on (its world matrices
        // are the input): setGpuDrawGeneration. The primary view's records of
        // eligible instanced spans (static, no occlusion culling this frame)
        // are written by compute; everything
        // else stays the CPU's.
        std::unique_ptr<vulkan::DrawGen> drawGen_;
        bool gpuDrawGen_ = false;
        // The frame's dispatch, filled by buildIndirectDrawData (primary view)
        // and uploaded by prepareDrawGen; drawGenActive_ gates the graph pass.
        std::vector<vulkan::DrawGen::SpanDesc> drawGenSpans_;
        std::vector<vulkan::DrawGen::LodRow>   drawGenLodRows_;// the spans' auto-LOD levels
        uint32_t drawGenWork_   = 0;
        bool     drawGenActive_ = false;
        float    drawGenPlanes_[24]{};
        // setGpuDrawGenerationCheck: the CPU path also builds the GPU spans'
        // records, into drawGenExpected_, for drawGenCheck to compare.
        bool gpuDrawGenCheck_ = false;
        struct DrawGenExpected {
            uint32_t entry  = 0;
            uint32_t record = 0;
            bool     culled = false;
            vulkan::impl::DrawInfoGpu draw{};
            VkDrawIndirectCommand     cmd{};
        };
        std::vector<DrawGenExpected> drawGenExpected_;
        uint32_t drawGenExpectedFrame_ = 0;

        // ── GPU TLAS instances (stage 5; TlasGen.hpp, tlas_gen.comp) ───────
        // OFF by default, needs the expansion: setGpuTlasInstances. The lean
        // TLAS fill writes the CPU's instances to the front of the frame's
        // instance buffer and plans tlasGenSpans_; the dispatch appends theirs.
        std::unique_ptr<vulkan::TlasGen> tlasGen_;
        bool gpuTlasGen_ = false;
        std::vector<vulkan::TlasGen::SpanDesc> tlasGenSpans_;
        std::vector<uint64_t> tlasGenLodRows_;// the spans' auto-LOD level BLAS addresses
        uint32_t tlasGenWork_   = 0;
        bool     tlasGenActive_ = false;// this frame's refit carries GPU instances
        // The CPU/GPU split of the TLAS last built: a different one is a full
        // build, never an update (the instance order changes).
        uint64_t tlasGenPartition_ = 0;
        // Instances the dispatch appends after pendingTlasInstances_.
        uint32_t pendingTlasGpuInstances_ = 0;
        // setGpuDrawGenerationCheck: the CPU's instances for the GPU spans.
        std::vector<VkAccelerationStructureInstanceKHR> tlasGenExpected_;
        uint32_t tlasGenExpectedFrame_ = 0;
        uint32_t tlasGenExpectedBase_  = 0;

        uint32_t instExpandMatrixTotal_ = 0;// matrices in the pool
        uint32_t instExpandWorkTotal_   = 0;// dispatch domain
        // Content serial per GPU-path span, bumped whenever the CPU re-baked
        // that span's entries (EntrySpan::movedThisFrame). The per-fif stamps
        // chase it, so a span that moved on frame N is re-sent to BOTH slots
        // rather than only to the slot that happened to be current — the bug
        // a plain "did it move this frame" gate would have.
        //
        // It also closes a hole the layout diff alone cannot see: two instanced
        // meshes swapping places in entrySpans_ with identical counts produces
        // an IDENTICAL InstExpandSpan list, so the diff would keep the old
        // per-slot stamps while slot k now means a different mesh. Safe because
        // entrySpans_ is only ever rebuilt by the full expansion, and that path
        // sets movedThisFrame on EVERY span — so any rebuild bumps every serial
        // and both slots re-upload.
        std::vector<uint64_t> instExpandSerial_;
        std::array<std::vector<uint64_t>, kFramesInFlight> instExpandFifSerial_;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_GPUDRIVENSTATE_HPP
