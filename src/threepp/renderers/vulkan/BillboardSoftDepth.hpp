// BillboardSoftDepth — the scene's linear depth for SOFT ParticleField
// billboards (BillboardRepr::softDistance).
//
// ── THE PROBLEM ─────────────────────────────────────────────────────────────
// A billboard is a flat camera-facing quad, depth-tested against the scene. A
// big one that passes through the ground is cut along a straight line, and the
// eye reads that line at once: a dust puff with a ruler edge. The usual cure,
// and the one this implements, is a depth fade: the sprite's coverage ramps to
// zero over `softDistance` metres as it approaches whatever is behind it, so
// the intersection is a gradient rather than a step. Hosts had worked around
// it by lifting every sprite until only its thin outer skirt reached the
// ground, which is what left dust starting visibly above the sand.
//
// The fade needs the scene's depth in METRES at every pixel a sprite covers,
// in the fragment stage of a pipeline that by design binds nothing but its
// texture. This class produces it: one compute dispatch per view per frame
// (particlefield_soft_depth.comp) turns the depth the sprites are tested
// against into one float per pixel, in a buffer whose device address rides the
// per-view billboard record (BillboardViewGpu::softDepthAddr).
//
// ── COST AND THE OFF SWITCH ─────────────────────────────────────────────────
// Nothing exists until a visible billboard field asks for a soft edge: no
// pipeline, no buffer, no pass, no descriptor. With every field at the default
// softDistance 0 the renderer never constructs this class, and the billboard
// fragment stage takes a uniform branch around the whole fade, so the output
// is the bytes it was before the feature. On, the dispatch reads the depth once
// and writes 4 B per pixel (8 MB at 1080p, per view and per frame in flight).
//
// ── DESCRIPTOR DISCIPLINE ───────────────────────────────────────────────────
// One set per (view, frame in flight), written only when the depth image it
// names changes (by imageViewSerial: a reallocated attachment can reuse a
// handle value) and only from prepare(), which runs in the post-fence window of
// that frame-in-flight. Nothing is ever written while a frame naming it is in
// flight (VUID-03047). Buffers are per (view, frame in flight) too, so two
// views never share one inside a frame and a resize retires the old buffer
// through the renderer's frame-serial queue instead of destroying it under a
// frame that may still read it.

#ifndef THREEPP_VULKAN_BILLBOARD_SOFT_DEPTH_HPP
#define THREEPP_VULKAN_BILLBOARD_SOFT_DEPTH_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class BillboardSoftDepth {

    public:
        // Same contract as ParticleFieldPass::RetireBufferFn: a buffer an
        // in-flight frame may still name goes to the frame-serial queue.
        using RetireBufferFn = std::function<void(Buffer&&)>;

        BillboardSoftDepth(VulkanContext& ctx, std::uint32_t framesInFlight, RetireBufferFn retireFn);
        ~BillboardSoftDepth();
        BillboardSoftDepth(const BillboardSoftDepth&) = delete;
        BillboardSoftDepth& operator=(const BillboardSoftDepth&) = delete;

        // What the billboard fragment stage reads: the address of a row-major
        // float array of `width` x `height` linear depths. Zero address = this
        // view has no soft depth this frame; the sprites keep their hard edge.
        struct Target {
            VkBuffer        buffer  = VK_NULL_HANDLE;
            VkDeviceAddress address = 0;
            std::uint32_t   width   = 0;
            std::uint32_t   height  = 0;
        };

        // One view's dispatch, frozen when the frame graph is built so that
        // recording it later depends on nothing that may have moved since.
        struct Dispatch {
            Target          target;
            VkDescriptorSet set          = VK_NULL_HANDLE;
            bool            multisampled = false;
            float           lin[4]{};// viewZ = (lin0 * d + lin1) / (lin2 * d + lin3)
        };

        // Graph-build time, in the post-fence window of `frame`. Sizes this
        // view's buffer for `ext`, points its set at `depthView` and freezes the
        // linearisation from `projInverse` (column-major, the inverse of the
        // projection that depth was rasterized with; xy jitter does not reach
        // its z and w rows). False when anything is missing — the view then
        // publishes no target and its sprites keep the hard edge.
        bool prepare(std::uint32_t viewId, std::uint32_t frame, std::uint64_t serial,
                     VkExtent2D ext, VkImageView depthView, bool multisampled,
                     const float projInverse[16], Dispatch& out);

        // Record time: what prepare() published for this view in this frame,
        // or an empty Target when it published nothing.
        [[nodiscard]] Target target(std::uint32_t viewId, std::uint32_t frame,
                                    std::uint64_t serial) const;

        // The dispatch. The depth must be in DEPTH_STENCIL_READ_ONLY_OPTIMAL —
        // the layout both of its producers leave it in — and the graph orders
        // the reads and the write (declare()).
        void record(VkCommandBuffer cb, const Dispatch& d) const;

        // The depth image's sampled read and the buffer's write, at the
        // compute stage. The billboard passes that follow declare their own
        // fragment-stage read of target().buffer.
        void declare(rg::RenderGraph& g, rg::PassBuilder& pass, VkImage depth, const Dispatch& d) const;

    private:
        struct Slot {
            Buffer          buf{};
            std::uint32_t   width = 0, height = 0;
            VkDescriptorSet set           = VK_NULL_HANDLE;
            std::uint64_t   setViewSerial = 0;// imageViewSerial the set names
            std::uint64_t   preparedSerial = 0;// frame serial of the last prepare()
        };
        struct ViewSlots {
            std::uint32_t     viewId     = 0;
            std::uint64_t     lastSerial = 0;
            std::vector<Slot> slots;// [framesInFlight]
        };

        VulkanContext& ctx_;
        std::uint32_t  framesInFlight_;
        RetireBufferFn retireFn_;

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;// sampler2D
        VkPipeline            pipeMs_     = VK_NULL_HANDLE;// sampler2DMS
        VkSampler             sampler_    = VK_NULL_HANDLE;
        VkDescriptorPool      pool_       = VK_NULL_HANDLE;

        std::vector<ViewSlots> views_;

        // Views that stopped asking (removed, or the scene lost its soft field)
        // give their buffers and sets back once no frame in flight can name
        // them.
        void prune(std::uint64_t serial);
        void release(ViewSlots& v);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_BILLBOARD_SOFT_DEPTH_HPP
