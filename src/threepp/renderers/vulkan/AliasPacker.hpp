// AliasPacker — TransientPool's placement bookkeeping, without a device.
//
// Blocks of memory; images placed into them by GROUP and SLOTS. Different
// groups are placed independently, so their images overlap (that overlap is
// the aliasing). Within a group, two images get disjoint ranges only when
// their slot masks intersect: an image used by frame-in-flight slot 0 and one
// used by slot 1 are never in the same frame, so they may share memory, while
// either of them and an image every slot uses (mask ~0u) may not. First fit,
// lowest block first, so every group packs from the bottom of block 0 and the
// groups share as much as their sizes allow.

#ifndef THREEPP_VULKAN_ALIAS_PACKER_HPP
#define THREEPP_VULKAN_ALIAS_PACKER_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <vector>

namespace threepp::vulkan {

    class AliasPacker {
    public:
        static constexpr uint32_t kNone = UINT32_MAX;
        struct Placement {
            uint32_t     block = kNone;
            VkDeviceSize offset = 0;
        };

        // First fit, in the first live block of a type in `memoryTypeBits`
        // with room in `group` for `slots`. block == kNone when none has.
        [[nodiscard]] Placement place(uint32_t group, uint32_t slots, VkDeviceSize size,
                                      VkDeviceSize alignment, uint32_t memoryTypeBits) const;
        uint32_t addBlock(VkDeviceSize size, uint32_t memoryType);
        void     occupy(uint32_t block, uint32_t group, uint32_t slots, VkDeviceSize offset, VkDeviceSize size);
        void     release(uint32_t block, uint32_t group, uint32_t slots, VkDeviceSize offset);
        [[nodiscard]] bool blockEmpty(uint32_t block) const;
        void     removeBlock(uint32_t block);// must be empty; its index is never reused
        // No new placements in any current block; they are removed as they
        // empty (a resize: the old blocks are the wrong size).
        void     closeAll();
        [[nodiscard]] VkDeviceSize liveBytes() const;// sum of live block sizes

    private:
        struct Range {
            VkDeviceSize offset, size;
            uint32_t     slots;
        };
        struct Block {
            VkDeviceSize size = 0;
            uint32_t     memoryType = 0;
            bool         alive = true;
            bool         open = true;// accepts placements
            std::map<uint32_t, std::vector<Range>> used;// per group, sorted by offset
        };
        std::vector<Block> blocks_;
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_ALIAS_PACKER_HPP
