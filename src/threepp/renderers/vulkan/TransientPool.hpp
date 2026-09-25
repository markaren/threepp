// TransientPool — device memory shared by images that live within one stretch
// of a frame.
//
// Frame-local scratch (a filter's ping-pong pair, the bloom pyramid, the
// overlay's multisampled target, ...) is written and consumed inside one part
// of the frame and dead outside it, so images the frame uses at different
// times can occupy the same memory. The pool binds each image into shared
// device-memory blocks. Images of different GROUPS may overlap: a group is the
// caller's claim that its images are never in use at the same time as
// another group's within one render graph (the shade's scratch and the post
// chain's). Within a group, images whose SLOTS masks intersect get disjoint
// ranges; an image only frame-in-flight slot 0 uses and one only slot 1 uses
// are never in the same frame, so they may overlap too. declare() hands the
// ranges to the render graph, which checks the claim every frame
// (RenderGraph::aliasErrors) and discards each image's contents at its first
// use in a graph, behind a barrier that waits for the last use of whatever
// held the memory before.
//
// Consequences for callers:
//  * a pooled image's contents never survive a graph: nothing may read it
//    before the graph writes it, or after the graph (host readbacks);
//  * no initial layout transition or clear at creation — the memory may hold
//    another image an in-flight frame is using; the graph's first use in the
//    next frame discards from UNDEFINED;
//  * destroyImage2D releases a pooled image (it recognises it), so owners
//    free pooled and dedicated images the same way.

#ifndef THREEPP_VULKAN_TRANSIENT_POOL_HPP
#define THREEPP_VULKAN_TRANSIENT_POOL_HPP

#include "threepp/renderers/vulkan/AliasPacker.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace threepp::vulkan {

    namespace rg {
        class RenderGraph;
    }

    // Where in the frame a pooled image lives. Images of different phases
    // share memory; so do one view's images of one phase that different
    // frame-in-flight slots use (see TransientPool).
    enum class TransientPhase : uint32_t {
        Shade = 0,// froxels → shade → denoise: the demodulation and filter scratch
        Post  = 1,// dof → bloom → temporal resolve / post: the post chain's scratch
        Tail  = 2,// field glow → overlay → sensor: the tail's scratch
    };

    [[nodiscard]] constexpr uint32_t transientGroup(TransientPhase phase, uint32_t view = 0) {
        return (view << 4) | static_cast<uint32_t>(phase);
    }
    // The slots mask of an image one frame-in-flight slot uses, and of one
    // every slot uses.
    [[nodiscard]] constexpr uint32_t transientSlot(uint32_t slot) { return 1u << slot; }
    constexpr uint32_t kTransientAllSlots = ~0u;

    class TransientPool {
    public:
        // `blockSize`: the smallest block allocated; an image larger than it
        // gets a block of its own size. 0 sizes each block to the image that
        // opens it — right when every group holds the same sizes in the same
        // order (a per-slot set of images: slot 1's then land on slot 0's).
        TransientPool(VmaAllocator allocator, VkDevice device, const char* name,
                      VkDeviceSize blockSize = 0);
        // For blocks allocated from now on. The best size holds the largest
        // group whole, so the other groups fit inside it. A change closes the
        // current blocks to new images (they are freed once their images
        // are), so a resize does not pack the new images into old-size blocks.
        void setBlockSize(VkDeviceSize bytes);
        // What `infos` take packed one after another in one block, without
        // creating them — the block size that holds them as a group.
        [[nodiscard]] VkDeviceSize packedBytes(const std::vector<VkImageCreateInfo>& infos) const;
        ~TransientPool();
        TransientPool(const TransientPool&) = delete;
        TransientPool& operator=(const TransientPool&) = delete;

        // Create an image of `info` bound into the pool in `group`, used by
        // the frame-in-flight slots in `slots`. VK_NULL_HANDLE on failure (the
        // caller may fall back to a dedicated allocation).
        VkImage createImage(const VkImageCreateInfo& info, uint32_t group, uint32_t slots);

        // Declare every pooled image's memory range to `graph` (images it has
        // not imported are skipped). Call after the graph's passes are added.
        void declare(rg::RenderGraph& graph) const;

        // Device memory the pool holds, and what its images would take with
        // an allocation each.
        [[nodiscard]] VkDeviceSize reservedBytes() const;
        [[nodiscard]] VkDeviceSize requestedBytes() const;

        // Destroy `image` and release its range if a pool created it. False
        // otherwise. destroyImage2D calls this for every image.
        static bool release(VkImage image);
        // A pool created `image`.
        [[nodiscard]] static bool pooled(VkImage image);

    private:
        struct Entry {
            uint32_t     block, group, slots;
            VkDeviceSize offset, size;
        };
        bool releaseImpl(VkImage image);

        VmaAllocator             allocator_;
        VkDevice                 device_;
        const char*              name_;
        VkDeviceSize             blockSize_;
        AliasPacker              packer_;
        std::vector<VmaAllocation> blocks_;// by packer block index; null once freed
        std::unordered_map<VkImage, Entry> images_;
        mutable std::mutex       mtx_;
    };

    // Create `info`'s image in `pool`/`group`, or — with no pool, or when the
    // pool cannot place it — with an allocation of its own (`*alloc`; null for
    // a pooled image). Either way destroyImage2D frees it.
    VkResult createImageMaybePooled(VmaAllocator allocator, TransientPool* pool, uint32_t group, uint32_t slots,
                                    const VkImageCreateInfo& info, VkImage* image, VmaAllocation* alloc);

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TRANSIENT_POOL_HPP
