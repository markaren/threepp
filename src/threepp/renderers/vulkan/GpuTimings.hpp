// GpuTimings — per-frame GPU timestamp readback + CPU record/frame timing.
//
// Two instruments, one query-pool pair per frame-in-flight each:
//   * the BRACKETS: a fixed slot pair per TimingPass, placed by hand around a
//     dispatch (begin/end), one pool per frame-in-flight;
//   * the PASS TABLE: a slot pair per render-graph pass of the frame, taken in
//     execution order as the graph records (this class is the graph's
//     rg::PassTimer), every view included, in a pool that grows to the
//     frame's pass count. It is what FrameTimings::passes reports.
// After the per-frame fence signals, readBack() reads the previous use of
// that slot from both and populates a VulkanRenderer::FrameTimings struct for
// the public getter.
//
// Extracted from VulkanRenderer.cpp during the file split.

#ifndef THREEPP_VULKAN_GPU_TIMINGS_HPP
#define THREEPP_VULKAN_GPU_TIMINGS_HPP

#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/RenderGraph.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    // One slot per bracketed GPU pass. timingMask records which passes wrote
    // both endpoints so readBack() can skip pools that weren't touched this
    // frame (overlay on no-overlay frames, DoF when disabled, …).
    enum TimingPass : uint32_t {
        TP_RasterGbuf    = 0,
        TP_OverlayDepth  = 1,
        TP_DeferredShade = 2,// internal name for the deferred-shade compute
                             // dispatch timing bracket; bridges to the public
                             // VulkanRenderer::FrameTimings::pathTraceMs field
                             // (kept for API stability — see GpuTimings.cpp).
        TP_Denoise       = 3,
        TP_TAA           = 4,
        TP_OverlayDraw   = 5,
        TP_GbufResolve   = 6,// MSAA dominant-sample resolve (setGbufferMsaa > 1 only)
        TP_ShadeB        = 7,// MSAA dispatch B: per-sample edge shading (setGbufferMsaa > 1 only)
        TP_Dof           = 8,// thin-lens depth of field (setDepthOfField only)
        TP_Froxel        = 9,// froxel volumetrics: inject + integrate (medium-active frames only)
        TP_SensorImage   = 10,// lens distortion + sensor noise (setLensDistortion / setSensorNoise only)
        TP_Splat         = 11,// Gaussian-splat tile rasterizer (scenes with a SplatCloud only)
        // The three stages INSIDE TP_Splat, so the pass can be attributed
        // rather than inferred. They partition TP_Splat and should sum to it
        // bar the barriers between them. Written for the FIRST splat cloud of
        // the frame only: a second cloud would write slots this frame already
        // holds, which is a VUID-vkCmdWriteTimestamp2-None-03864 violation and
        // garbage numbers besides.
        TP_SplatProject  = 12,// per-splat project + cull + prefix sum + expand
        TP_SplatSort     = 13,// the 8 x 4-bit radix passes and their scans
        TP_SplatRaster   = 14,// tile ranges + the tile-local composite
        // The WHOLE submitted command buffer: opened in beginFrame right after
        // the pool reset, closed in endFrame right before vkEndCommandBuffer.
        // Deliberately not a sum of the slots above — it also covers every pass
        // that has no bracket at all (skinned/tet/grass deformers, bloom/post,
        // RCAS, cluster build, cloud march, auto-exposure, particle
        // light, ImGui/present transition) AND every secondary view, whose brackets
        // are suppressed (the pass table has a row for each of these). Read
        // gpuTotalMs - gpuPassSumMs to see how much GPU work
        // is invisible to the bracketed passes. It is a SPAN, not busy time: the
        // TOP_OF_PIPE open is not covered by the imageAvailable wait's stage mask,
        // so a swapchain-acquire stall can land inside it.
        TP_Frame         = 15,
        // GPU per-instance world-matrix expansion (instance_expand.comp).
        // Stage 1 of the GPU-driven instance work: bracketed from the start
        // because "the dispatch is free" is a claim, and the whole project is
        // decided by whether work moved to the GPU costs less there.
        TP_InstanceExpand = 16,
        // ParticleField density scatter (particle_density_scatter.comp): the
        // per-frame clear + splat of every dust field into its world-anchored
        // volume. Bracketed separately from TP_Froxel because §3.3's claim —
        // "the cheapest per-particle representation by an order of magnitude" —
        // is only checkable if the per-particle half and the fixed froxel half
        // are two numbers. Recorded ONCE per frame for all views.
        TP_ParticleDensity = 17,
        // ParticleField DEVICE EMITTER (particle_emit.comp): the closed-form
        // position + prevPosition write for every Ownership::Renderer field.
        // Its own bracket rather than a share of TP_ParticleDensity because the
        // F2 checkpoint is stated as a SPLIT — "emit + scatter < 1 ms, report
        // the two" — and a combined number could hide either half growing.
        // Recorded ONCE per frame for all views.
        TP_ParticleEmit   = 18,
        // Ocean / DisplacedMesh per-frame update, split into the four stages
        // recordDisplacedDeform records back-to-back so the "FFT water is
        // expensive" claim can be attributed rather than inferred: the cascade
        // spectrum+IFFT chains, the water_displace vertex/normal dispatch, the
        // world-foam accumulator, and the in-place BLAS refit/rebuild. Written
        // for the FIRST displaced mesh of the frame only (same single-slot
        // constraint as the splat stages above); the one-shot structural
        // first-build path records untimed.
        TP_OceanFFT       = 19,
        TP_OceanDisplace  = 20,
        TP_OceanFoam      = 21,
        TP_OceanBlas      = 22,
        // Per-frame TLAS refit (recordTlasRefit on the frame cb). Bracketed
        // because it sat on the "invisible to the brackets" list while claims
        // were being made about it.
        TP_TlasRefit      = 23,
        // Per-frame dynamic-geometry refit (recordDynamicGeomRefits on the frame
        // cb): the staging→vertex/normal copies, the prevVertex snapshot and the
        // batched BLAS refit for every graduated CPU deformer and every
        // vertex-interop mesh. Bracketed because the "BLAS refit is cheap" claim
        // was an extrapolation from the ocean's numbers — this work had no
        // bracket at all and landed in the gpuTotal-minus-passSum residual.
        TP_DynGeomRefit   = 24,
        // Half-res RT ambient occlusion + bent normals (rtao.comp). Its own
        // bracket because without one the pass hides inside the unbracketed
        // residual, and the whole Phase-2 case is decided by what it costs.
        TP_Rtao           = 25,
        // World-space probe-GI update (probe_update.comp): the prev-store
        // snapshot copy plus the round-robin probe dispatch. Recorded ONCE per
        // frame for all views (the dispatch is primary-only — see
        // recordSceneDispatch), so this is the whole frame's probe cost, not a
        // per-view share. It sat in the unbracketed residual while the
        // per-view-duplication claim was being made about it.
        TP_ProbeGI        = 26,
        TP_COUNT          = 27,
    };
    inline constexpr uint32_t kTimingSlots = TP_COUNT * 2u;

    class GpuTimings final: public rg::PassTimer {
    public:
        // Probes device timestamp support and creates one VkQueryPool per
        // frame-in-flight. Pools are sized for kTimingSlots = TP_COUNT × 2.
        GpuTimings(VulkanContext& ctx, uint32_t framesInFlight);
        ~GpuTimings() override;
        GpuTimings(const GpuTimings&)            = delete;
        GpuTimings& operator=(const GpuTimings&) = delete;

        // --- per-frame command-buffer operations ---

        // Call at the start of command recording. Captures the CPU record-start
        // time and GPU-resets this frame's query pool slots (brackets and pass
        // table); the pass table's pool is recreated here when an earlier frame
        // had more passes than it could hold.
        void beginFrame(VkCommandBuffer cb, uint32_t frame);

        // rg::PassTimer — the render graph calls these around every pass it
        // records (and its entry/exit barriers) into the frame beginFrame
        // opened. Each span takes the next slot pair of this frame's pass
        // pool; both ends are stamped at ALL_COMMANDS, so a span starts once
        // all earlier work has completed and spans never overlap. Not subject
        // to setSuppressed: a secondary view's passes are rows of their own.
        // Past the pool's capacity the frame's remaining spans go unrecorded
        // and the next use of the slot gets a larger pool.
        void passBegin(VkCommandBuffer cb, const char* name, uint32_t group) override;
        void passEnd(VkCommandBuffer cb) override;

        // GPU timestamp pairs — call begin before a pass, end after it.
        void begin(VkCommandBuffer cb, TimingPass pass, uint32_t frame);
        void end  (VkCommandBuffer cb, TimingPass pass, uint32_t frame);

        // Closes the TP_Frame bracket beginFrame opened. Must be the LAST
        // timestamp recorded into this command buffer (call it immediately before
        // vkEndCommandBuffer). Sets TP_Frame's recorded-mask bit — the bit is set
        // by the END, not the begin, so readBack can never WAIT_BIT on a pair whose
        // second endpoint was never written.
        void endFrameTotal(VkCommandBuffer cb, uint32_t frame);

        // Call at vkEndCommandBuffer to capture cpuRecordMs.
        void finishRecord();

        // --- readback (call after the per-frame fence has signaled) ---

        // Reads GPU timestamps written by the previous use of this frame's
        // pool slot and populates the internal FrameTimings.
        // pendingCpuEnsureMs: the ensureSceneBuilt wall time from the prior
        // render() call (deferred so it appears in the same timing row as the
        // GPU passes for that scene).
        void readBack(uint32_t frame, float pendingCpuEnsureMs);

        // --- out-of-band setters (called outside the cmd-buffer window) ---
        void setCpuFrameMs(float ms) { lastTimings_.cpuFrameMs = ms; }

        // Silence the BRACKETS for a stretch of recording. There is ONE bracket
        // pool per frame-in-flight, with one slot pair per pass, so a second
        // view re-running the same passes into the same command buffer would
        // write timestamps that are already written — every one of them a
        // VUID-vkCmdWriteTimestamp2-None-03864, and the resulting numbers a
        // meaningless mix of two views. Secondary views therefore record
        // their brackets silently and the bracket fields keep meaning "the
        // primary"; the pass table is per pass instance and is not silenced.
        void setSuppressed(bool s) { suppressed_ = s; }

        // --- accessors ---
        [[nodiscard]] VulkanRenderer::FrameTimings timings() const { return lastTimings_; }
        [[nodiscard]] bool supported() const { return timingsSupported_; }

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        std::vector<VkQueryPool> pools_;
        std::vector<uint32_t>   maskRecorded_;
        float timestampPeriodNs_ = 1.0f;
        bool  timingsSupported_  = false;
        bool  suppressed_        = false;

        // Pass table. One pool per frame-in-flight, 2 × passCapacity_ queries;
        // passSlots_ names the spans taken in this use of the slot, in order.
        // A capacity miss raises passWanted_, and the slot's pool is recreated
        // to it at its next beginFrame (its previous use has been read back
        // by then). Pass names are the graph's literals, kept by pointer.
        struct PassSlot {
            const char* name = "";
            uint32_t    group = 0;
        };
        static constexpr uint32_t kPassTableInitial = 256;// passes
        static constexpr uint32_t kPassTableMax     = 8192;
        std::vector<VkQueryPool>           passPools_;
        std::vector<uint32_t>              passCapacity_;
        std::vector<std::vector<PassSlot>> passSlots_;
        std::vector<uint64_t>              passResults_;// readBack scratch
        uint32_t passWanted_ = kPassTableInitial;
        uint32_t recordFrame_ = 0;   // the slot beginFrame opened
        bool     passOpen_    = false;// passBegin took a slot: passEnd stamps
        void ensurePassPool(uint32_t frame);

        VulkanRenderer::FrameTimings                   lastTimings_{};
        std::chrono::high_resolution_clock::time_point recordStartTp_{};
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_GPU_TIMINGS_HPP
