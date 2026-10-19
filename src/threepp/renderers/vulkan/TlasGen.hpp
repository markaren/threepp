// TlasGen — the TLAS instance descriptors of instanced spans, written by
// compute (stage 5 of plans/gpu-driven-instances.md), the way DrawGen writes
// their G-buffer records: an opt-in producer the CPU's output is the reference
// for (setGpuTlasInstances; VulkanRenderer::tlasGenCheck).
//
// The lean path's TLAS fill writes the CPU's instances to the front of the
// frame's instance buffer and plans one SpanDesc per eligible span;
// tlas_gen.comp appends their instances after the CPU's (the TLAS order is
// free: instanceCustomIndex names the entry). Frame flow: the fill
// (ensureSceneBuilt) plans; prepareTlasGen (post-fence, after
// prepareInstanceExpansion) uploads the plan and points this slot's set at the
// world buffer and this slot's instance buffer; the "tlasGen" graph pass runs
// after the expansion and before the "tlas" build that reads it.
#ifndef THREEPP_VULKAN_TLAS_GEN_HPP
#define THREEPP_VULKAN_TLAS_GEN_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class TlasGen {
    public:
        // Matches tlas_gen.comp's TlasSpan (scalar, 40 B).
        struct SpanDesc {
            uint64_t blasAddress;// level 0's
            uint32_t firstEntry;
            uint32_t count;
            uint32_t instBase;
            uint32_t workBase;
            uint32_t maskWord; // mask << 24
            uint32_t flagsWord;// instance flags << 24
            uint32_t lodRowBase; // its first level BLAS address, when lodRowCount > 0
            uint32_t lodRowCount;// an auto-LOD chain's levels 1..n and the fallback; 0: no chain
        };
        static_assert(sizeof(SpanDesc) == 40, "tlas_gen SpanDesc drift");

        struct Pc {
            uint32_t spanCount;
            uint32_t totalWork;
        };

        TlasGen(VulkanContext& ctx, uint32_t framesInFlight);
        ~TlasGen();
        TlasGen(const TlasGen&) = delete;
        TlasGen& operator=(const TlasGen&) = delete;

        // Grow this frame's span pool and point this frame's set at the world
        // buffer and the frame's instance buffer. Post-fence, pre-record.
        // rows: the level BLAS addresses the spans' lodRowBase index (level k
        // >= 1: selectLodGeom(rec, k).asAddress). lodLevels: a byte per entry.
        void prepareFrame(uint32_t frame, uint32_t spanCount, uint32_t rowCount, VkBuffer world,
                          VkBuffer instances, VkBuffer lodFades, VkBuffer lodLevels);
        [[nodiscard]] SpanDesc* spanPtr(uint32_t frame) { return spanPtrs_[frame]; }
        [[nodiscard]] uint64_t* rowPtr(uint32_t frame) { return rowPtrs_[frame]; }
        void flushSpans(uint32_t frame, uint32_t spanCount, uint32_t rowCount);

        void record(VkCommandBuffer cb, uint32_t frame, const Pc& pc);
        void declare(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame) const;

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;
        std::vector<Buffer>    spanBufs_;
        std::vector<SpanDesc*> spanPtrs_;
        std::vector<Buffer>    rowBufs_;
        std::vector<uint64_t*> rowPtrs_;
        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TLAS_GEN_HPP
