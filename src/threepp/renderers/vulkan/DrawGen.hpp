// DrawGen — the G-buffer draw records of instanced spans, written by compute.
//
// Stage 2 of plans/gpu-driven-instances.md, built the way stage 1
// (InstanceExpand) was: an opt-in producer proven equal to the CPU before
// anything depends on it. buildIndirectDrawData keeps writing every record the
// CPU path writes, except those of an eligible InstancedMesh span: for those
// it reserves the span's records in place in its cull-mode bucket and hands
// draw_gen.comp one SpanDesc, and the shader writes the DrawInfo and the
// indirect command of every instance (frustum-culled ones as vertexCount 0)
// from the per-entry world matrices stage 1 produced.
//
// Frame flow (Impl::buildIndirectDrawData, Impl::prepareDrawGen, the
// "drawGen" frame-graph pass):
//   1. buildIndirectDrawData collects the spans and their record ranges;
//   2. prepareDrawGen (post-fence, pre-record, after prepareInstanceExpansion,
//      which may grow the world buffer this set names) grows this
//      frame-in-flight's span pool, writes it, and rewrites this frame's set;
//   3. record: one dispatch over the summed instance count, after the
//      expansion and before the G-buffer pass reads the records (declare()).
// The outputs are the primary view's per-frame DrawInfo and command buffers
// (host-mapped; the CPU writes its own records into the same buffers at
// disjoint ranges before the submit).
#ifndef THREEPP_VULKAN_DRAW_GEN_HPP
#define THREEPP_VULKAN_DRAW_GEN_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class DrawGen {
    public:
        // Matches draw_gen.comp's DrawSpan (scalar, 120 B).
        struct SpanDesc {
            uint64_t posAddr;
            uint64_t nrmAddr;
            uint64_t uvAddr;
            uint64_t prevPosAddr;
            uint64_t indexAddr;
            uint64_t colorAddr;
            uint32_t firstEntry;
            uint32_t count;
            uint32_t recordBase;
            uint32_t workBase;
            uint32_t flagsBase;
            uint32_t indexed;
            uint32_t stableId;
            uint32_t packedAttrs;
            float    polygonOffset;
            uint32_t vertexCount;
            uint32_t firstVertex;
            uint32_t lodRowBase; // the span's first LodRow, when lodRowCount > 0
            float    localCenter[3];
            float    localRadius;
            uint32_t lodRowCount;// an auto-LOD chain's levels 1..n and the fallback row; 0: no chain
            uint32_t pad0;
        };
        static_assert(sizeof(SpanDesc) == 120, "draw_gen SpanDesc drift");

        // Matches draw_gen.comp's LodRow (scalar, 16 B): what the record of an
        // entry at level k >= 1 draws, selectLodGeom(rec, k) with no drawRange
        // clamp (the range is the base level's). Row n is the fallback a level
        // past the chain gets.
        struct LodRow {
            uint64_t indexAddr;
            uint32_t vertexCount;
            uint32_t indexed;
        };
        static_assert(sizeof(LodRow) == 16, "draw_gen LodRow drift");

        // Matches draw_gen.comp's push constants (scalar, 112 B).
        struct Pc {
            float    planes[24];// six planes: xyz normal, w constant
            uint32_t spanCount;
            uint32_t totalWork;
            uint32_t geomStrideWords;
            uint32_t geomFlagsWord;
        };
        static_assert(sizeof(Pc) == 112, "draw_gen push-constant drift");

        DrawGen(VulkanContext& ctx, uint32_t framesInFlight);
        ~DrawGen();
        DrawGen(const DrawGen&) = delete;
        DrawGen& operator=(const DrawGen&) = delete;

        // Grow this frame's span and LOD row pools and point this frame's set
        // at the buffers it reads and writes. Once per frame, after this
        // slot's fence and before this frame's record() (invariant 4).
        // lodLevels: one byte per entry (MeshEntry::lodLevel), four a word.
        void prepareFrame(uint32_t frame, uint32_t spanCount, uint32_t rowCount, VkBuffer world,
                          VkBuffer geomDescs, VkBuffer drawInfo, VkBuffer cmds, VkBuffer lodFades,
                          VkBuffer lodLevels);
        [[nodiscard]] SpanDesc* spanPtr(uint32_t frame) { return spanPtrs_[frame]; }
        [[nodiscard]] LodRow* rowPtr(uint32_t frame) { return rowPtrs_[frame]; }
        void flushSpans(uint32_t frame, uint32_t spanCount, uint32_t rowCount);

        void record(VkCommandBuffer cb, uint32_t frame, const Pc& pc);
        // record()'s accesses, from the reflected shader.
        void declare(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame) const;

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;
        std::vector<Buffer>    spanBufs_;// [fif] host-mapped SpanDesc[]
        std::vector<SpanDesc*> spanPtrs_;
        std::vector<Buffer>    rowBufs_; // [fif] host-mapped LodRow[]
        std::vector<LodRow*>   rowPtrs_;
        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [fif]

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DRAW_GEN_HPP
