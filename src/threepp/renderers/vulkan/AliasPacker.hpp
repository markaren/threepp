// AliasPacker — TransientPool's placement bookkeeping, without a device.
//
// Blocks of memory; images placed into them by SPAN and SLOTS. A span is a
// bit mask of the stretches of the frame an image is live in (TransientPool's
// phases, one set of bits per view). Two images get disjoint ranges only when
// their spans intersect AND their slot masks intersect: images live in
// different stretches of the frame overlap (that overlap is the aliasing),
// and so do an image used by frame-in-flight slot 0 and one used by slot 1,
// which are never in the same frame. First fit, lowest block first, so every
// stretch packs from the bottom of block 0 and they share as much as their
// sizes allow.

#ifndef THREEPP_VULKAN_ALIAS_PACKER_HPP
#define THREEPP_VULKAN_ALIAS_PACKER_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    // The stretches of one view's passes, in frame order. A pooled image is
    // live in a contiguous run of them (its span); images whose runs do not
    // meet share memory (see TransientPool).
    enum class TransientPhase : uint32_t {
        Gbuffer = 0,// G-buffer raster → MSAA resolve
        Light   = 1,// probes, clusters, cloud shadow, froxels, cloud march, RTAO → shade
        Filter  = 2,// denoise → auto exposure → particle light
        Post    = 3,// splats → dof → bloom → temporal resolve / post
        Tail    = 4,// field glow → overlay → sensor
    };

    // Bit mask of (view, phase) pairs. Each view gets 8 bits; views are
    // folded modulo 8, which can only make two images conflict that need
    // not (never the reverse), so it is safe.
    using TransientSpan = uint64_t;

    [[nodiscard]] constexpr TransientSpan transientSpan(TransientPhase first, TransientPhase last,
                                                        uint32_t view = 0) {
        TransientSpan m = 0;
        for (uint32_t p = static_cast<uint32_t>(first); p <= static_cast<uint32_t>(last); ++p)
            m |= TransientSpan{1} << ((view % 8u) * 8u + p);
        return m;
    }
    // The span of an image live in one phase only.
    [[nodiscard]] constexpr TransientSpan transientGroup(TransientPhase phase, uint32_t view = 0) {
        return transientSpan(phase, phase, view);
    }
    // The slots mask of an image one frame-in-flight slot uses, and of one
    // every slot uses.
    [[nodiscard]] constexpr uint32_t transientSlot(uint32_t slot) { return 1u << slot; }
    constexpr uint32_t kTransientAllSlots = ~0u;

    class AliasPacker {
    public:
        using Span = uint64_t;
        static constexpr uint32_t kNone = UINT32_MAX;
        struct Placement {
            uint32_t     block = kNone;
            VkDeviceSize offset = 0;
        };

        // First fit, in the first live block of a type in `memoryTypeBits`
        // with room clear of every image that `span` and `slots` meet.
        // block == kNone when none has.
        [[nodiscard]] Placement place(Span span, uint32_t slots, VkDeviceSize size,
                                      VkDeviceSize alignment, uint32_t memoryTypeBits) const;
        uint32_t addBlock(VkDeviceSize size, uint32_t memoryType);
        void     occupy(uint32_t block, Span span, uint32_t slots, VkDeviceSize offset, VkDeviceSize size);
        void     release(uint32_t block, Span span, uint32_t slots, VkDeviceSize offset);
        [[nodiscard]] bool blockEmpty(uint32_t block) const;
        void     removeBlock(uint32_t block);// must be empty; its index is never reused
        // No new placements in any current block; they are removed as they
        // empty (a resize: the old blocks are the wrong size).
        void     closeAll();
        [[nodiscard]] VkDeviceSize liveBytes() const;// sum of live block sizes

    private:
        struct Range {
            VkDeviceSize offset, size;
            Span         span;
            uint32_t     slots;
        };
        struct Block {
            VkDeviceSize size = 0;
            uint32_t     memoryType = 0;
            bool         alive = true;
            bool         open = true;// accepts placements
            std::vector<Range> used;// sorted by offset
        };
        std::vector<Block> blocks_;
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_ALIAS_PACKER_HPP
