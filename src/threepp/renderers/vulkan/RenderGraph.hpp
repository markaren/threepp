// RenderGraph — declared-access pass sequencing for the Vulkan backend.
//
// A pass declares the images and buffers it touches and HOW (stage, access,
// layout) through a PassBuilder; its execute callback then records only the
// work itself. The graph walks the passes in declaration order, tracks the
// state of every resource, and records the pipeline barriers (sync2) and
// layout transitions between passes. Barriers WITHIN a pass — a mip chain, a
// multi-dispatch filter — stay in the pass; the graph orders passes.
//
// Execution order is declaration order. The graph does not reorder or cull.
//
// Boundaries. A graph is built and executed per frame (per view: the renderer
// builds one for the primary camera, from the deformers to the sensor stage,
// and one for each secondary view), with hand-synchronised recording before
// and after it (the frame's head, the post-view tail, the previous frame).
// Resource state at the graph's boundaries is therefore not known to it, and
// it assumes the conservative answer:
//   * entry: one memory barrier (ALL_COMMANDS/MEMORY_WRITE -> the stages the
//     graph's passes declared/MEMORY_READ|MEMORY_WRITE) before the first pass,
//     which also carries each imported image from its declared entry layout
//     to its first-use layout;
//   * exit: the mirror image after the last pass (the declared stages ->
//     ALL_COMMANDS), which carries each image to its declared exit layout
//     (default: the entry layout), so the recording that follows finds each
//     image where it expects it.
// Within the graph every barrier is derived from the declarations.
//
// Planning and recording are separate: compile() fills the barrier plan
// without touching a command buffer (unit-testable, and what dump() prints),
// execute() compiles if needed and records.

#ifndef THREEPP_VULKAN_RENDER_GRAPH_HPP
#define THREEPP_VULKAN_RENDER_GRAPH_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::vulkan::rg {

    // One use of a resource by a pass.
    //
    // Images: `layout` is the layout the pass needs the image in when it
    // starts; the graph transitions to it. UNDEFINED means the pass takes the
    // image in whatever layout it is in and transitions it itself, discarding
    // its contents (a render pass attachment with initialLayout UNDEFINED); the
    // graph then only orders it. `finalLayout` is the layout the pass leaves
    // the image in when it transitions it itself (a render pass finalLayout, or
    // a pass that flips an image and does not flip it back); UNDEFINED means
    // the pass leaves the layout alone.
    struct Access {
        VkPipelineStageFlags2 stages = 0;
        VkAccessFlags2        access = 0;
        VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;// images only
        bool                  write  = false;
        VkImageLayout         finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;// images only
    };

    // ── Common accesses ─────────────────────────────────────────────────────
    // Defaults describe the backend's usual case: compute shaders, storage
    // images in GENERAL. Pass the layout a descriptor was written with — the
    // graph transitions to it, it does not guess it.
    constexpr VkPipelineStageFlags2 kCompute  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr VkPipelineStageFlags2 kFragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    Access sampled(VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VkPipelineStageFlags2 stages = kCompute);
    Access storageRead(VkPipelineStageFlags2 stages = kCompute,
                       VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
    Access storageWrite(VkPipelineStageFlags2 stages = kCompute,
                        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
    Access storageReadWrite(VkPipelineStageFlags2 stages = kCompute,
                            VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
    // A GENERAL-layout image read both by sampler and as a storage image.
    Access generalRead(VkPipelineStageFlags2 stages = kCompute);
    Access uniformRead(VkPipelineStageFlags2 stages = kCompute);
    Access indirectRead();
    Access transferSrc(VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Access transferDst(VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    Access colorAttachment(VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           bool loadOrBlend = true);
    Access depthAttachment(VkImageLayout layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           bool write = true);
    Access accelRead(VkPipelineStageFlags2 stages = kCompute);// ray query / trace

    struct ImageHandle {
        uint32_t index = UINT32_MAX;
        [[nodiscard]] bool valid() const { return index != UINT32_MAX; }
    };
    struct BufferHandle {
        uint32_t index = UINT32_MAX;
        [[nodiscard]] bool valid() const { return index != UINT32_MAX; }
    };

    // A planned VkImageMemoryBarrier2 / VkBufferMemoryBarrier2, kept in plain
    // form so tests and dump() can read it.
    struct PlannedBarrier {
        bool                  isImage = true;
        uint32_t              resource = 0;
        uint32_t              baseMip = 0, mipCount = 1;
        VkPipelineStageFlags2 srcStages = 0, dstStages = 0;
        VkAccessFlags2        srcAccess = 0, dstAccess = 0;
        VkImageLayout         oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout         newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    class RenderGraph;

    class PassBuilder {
    public:
        // mipCount 0 = every mip from baseMip on.
        PassBuilder& use(ImageHandle image, const Access& access,
                         uint32_t baseMip = 0, uint32_t mipCount = 0);
        PassBuilder& use(BufferHandle buffer, const Access& access);

    private:
        friend class RenderGraph;
        PassBuilder(RenderGraph& g, uint32_t pass): graph_(g), pass_(pass) {}
        RenderGraph& graph_;
        uint32_t     pass_;
    };

    class RenderGraph {
    public:
        using ExecuteFn = std::function<void(VkCommandBuffer)>;

        // Forget every pass and resource. The graph object is reused frame to
        // frame: passes and images go to free lists that keep their vectors'
        // capacity, so a steady-state frame builds its graph without allocating.
        void reset();

        // An image owned elsewhere. entryLayout is its layout when the graph
        // starts; exitLayout its layout when the graph ends (UNDEFINED: the
        // entry layout). An image whose contents are dead between frames is
        // imported in UNDEFINED: its first use discards them and it is left
        // in whatever layout its last use put it in. An image imported twice
        // returns the same handle.
        // Names (here and in addPass) are kept by pointer, for dump(): pass
        // string literals or other storage that outlives the graph.
        ImageHandle importImage(const char* name, VkImage image, VkImageAspectFlags aspect,
                                uint32_t mipLevels, VkImageLayout entryLayout,
                                VkImageLayout exitLayout = VK_IMAGE_LAYOUT_UNDEFINED);
        BufferHandle importBuffer(const char* name, VkBuffer buffer);
        // Memory the graph cannot name as one VkBuffer: a set of buffers and
        // images a pass reaches through buffer device addresses or through
        // acceleration-structure traversal (every deformer's vertex output, the
        // BLASes a TLAS references). Declared like a buffer; its barriers are
        // global memory barriers. The same name returns the same handle.
        BufferHandle importMemory(const char* name);

        // Passes run in the order they are added.
        PassBuilder addPass(const char* name, ExecuteFn execute);

        // Plan every barrier. Idempotent; execute() calls it.
        void compile();

        // Record entry barrier, (barrier, pass) pairs, exit barrier.
        void execute(VkCommandBuffer cb);

        // ── Introspection (tests, THREEPP_RG_DUMP) ───────────────────────────
        [[nodiscard]] size_t passCount() const { return passes_.size(); }
        [[nodiscard]] const std::vector<PlannedBarrier>& barriersBefore(size_t pass) const {
            return passes_[pass].barriers;
        }
        [[nodiscard]] const std::vector<PlannedBarrier>& entryBarriers() const { return entry_; }
        [[nodiscard]] const std::vector<PlannedBarrier>& exitBarriers() const { return exit_; }
        [[nodiscard]] bool hasEntryMemoryBarrier() const { return entryMemory_; }
        [[nodiscard]] bool hasExitMemoryBarrier() const { return exitMemory_; }
        [[nodiscard]] std::string dump() const;

    private:
        friend class PassBuilder;

        struct Use {
            bool     isImage = true;
            uint32_t resource = 0;
            uint32_t baseMip = 0, mipCount = 0;
            Access   access;
        };
        struct Pass {
            const char*                 name = "";
            ExecuteFn                   execute;
            std::vector<Use>            uses;
            std::vector<PlannedBarrier> barriers;
        };
        // Per image mip / per buffer. `readers` are the stages that read since
        // the last write (WAR ordering); `visible` lists (stages, access)
        // pairs the last write has already been made visible to.
        struct State {
            bool                  touched = false;// accessed by a pass yet
            VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 writeStages = 0;
            VkAccessFlags2        writeAccess = 0;
            VkPipelineStageFlags2 readers = 0;
            std::vector<std::pair<VkPipelineStageFlags2, VkAccessFlags2>> visible;
        };
        struct Image {
            const char*        name = "";
            VkImage            image = VK_NULL_HANDLE;
            VkImageAspectFlags aspect = 0;
            uint32_t           mipLevels = 1;
            VkImageLayout      entryLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout      exitLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::vector<State> mips;
        };
        // buffer == VK_NULL_HANDLE: a memory resource (importMemory).
        struct Buffer {
            const char* name = "";
            VkBuffer    buffer = VK_NULL_HANDLE;
            State       state;
        };

        std::vector<Pass>   passes_;
        std::vector<Image>  images_;
        std::vector<Buffer> buffers_;
        std::vector<Pass>   freePasses_;
        std::vector<Image>  freeImages_;
        std::unordered_map<VkImage, uint32_t>  imageIndex_;
        std::unordered_map<VkBuffer, uint32_t> bufferIndex_;
        std::vector<uint32_t> memoryIndex_;// buffers_ entries that are memory resources
        mutable std::vector<VkImageMemoryBarrier2>  imageScratch_;// record()
        mutable std::vector<VkBufferMemoryBarrier2> bufferScratch_;
        mutable std::vector<VkMemoryBarrier2>       memoryScratch_;
        std::vector<PlannedBarrier> entry_, exit_;
        bool entryMemory_ = false, exitMemory_ = false;
        bool compiled_ = false;
        VkPipelineStageFlags2 usedStages_ = 0;

        static void plan(State& s, const Access& a, bool isImage, uint32_t resource,
                         uint32_t mip, std::vector<PlannedBarrier>& out);
        void record(VkCommandBuffer cb, const std::vector<PlannedBarrier>& barriers,
                    bool globalMemory, VkPipelineStageFlags2 memSrc,
                    VkPipelineStageFlags2 memDst) const;
    };

}// namespace threepp::vulkan::rg

#endif//THREEPP_VULKAN_RENDER_GRAPH_HPP
