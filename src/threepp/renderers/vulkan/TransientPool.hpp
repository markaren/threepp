// TransientPool — device memory shared by images that live within part of a
// frame.
//
// Frame-local images (a filter's ping-pong pair, the bloom pyramid, the
// overlay's multisampled target, the scene HDR target the shade writes and the
// post chain consumes, ...) are written and consumed inside part of the frame
// and dead outside it, so images the frame uses at different times can occupy
// the same memory. The pool binds each image into shared device-memory
// blocks. Each image carries a SPAN: the phases of the frame (per view) it is
// live in, the caller's claim that it is in use nowhere else in the frame's
// render graph. Images whose spans do not intersect may overlap (the shade's
// filter scratch and the post chain's). Images whose spans intersect get
// disjoint ranges unless their SLOTS masks do not intersect: an image only
// frame-in-flight slot 0 uses and one only slot 1 uses are never in the same
// frame, so they may overlap too. declare() hands the ranges to the render
// graph, which checks the claim every frame (RenderGraph::aliasErrors) and
// discards each image's contents at its first use in a graph, behind a
// barrier that waits for the last use of whatever held the memory before.
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


    class TransientPool {
    public:
        // `blockSize`: the smallest block allocated; an image larger than it
        // gets a block of its own size. 0 sizes each block to the image that
        // opens it — right when every group holds the same sizes in the same
        // order (a per-slot set of images: slot 1's then land on slot 0's).
        TransientPool(VmaAllocator allocator, VkDevice device, const char* name,
                      VkDeviceSize blockSize = 0);
        // For blocks allocated from now on. The best size holds a view's
        // images whole (packedBytes), so they share one block. A change closes
        // the current blocks to new images (they are freed once their images
        // are), so a resize does not pack the new images into old-size blocks.
        void setBlockSize(VkDeviceSize bytes);
        // An image to be created, for packedBytes.
        struct Request {
            VkImageCreateInfo info{};
            TransientSpan     span  = 0;
            uint32_t          slots = kTransientAllSlots;
        };
        // What `requests` take when placed in that order into one block, as
        // createImage would place them, without creating them — the block
        // size that holds them.
        [[nodiscard]] VkDeviceSize packedBytes(const std::vector<Request>& requests) const;
        ~TransientPool();
        TransientPool(const TransientPool&) = delete;
        TransientPool& operator=(const TransientPool&) = delete;

        // Create an image of `info` bound into the pool, live in `span`, used
        // by the frame-in-flight slots in `slots`. VK_NULL_HANDLE on failure
        // (the caller may fall back to a dedicated allocation).
        VkImage createImage(const VkImageCreateInfo& info, TransientSpan span, uint32_t slots);

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
            uint32_t      block;
            TransientSpan span;
            uint32_t      slots;
            VkDeviceSize  offset, size;
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

    // Create `info`'s image in `pool` with `span`, or — with no pool, or when
    // the pool cannot place it — with an allocation of its own (`*alloc`; null
    // for a pooled image). Either way destroyImage2D frees it.
    VkResult createImageMaybePooled(VmaAllocator allocator, TransientPool* pool, TransientSpan span, uint32_t slots,
                                    const VkImageCreateInfo& info, VkImage* image, VmaAllocation* alloc);

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TRANSIENT_POOL_HPP
