// Frame-serial deferred-deletion (retire) queue for the Vulkan deferred
// renderer. Replaces staleness-triggered vkDeviceWaitIdle full-device stalls:
// a GPU resource whose LAST referencing frame is known is handed to the queue
// (stamped with that frame's serial) instead of being destroyed inline, then
// reclaimed a few frames later once its fence has provably signaled — no stall.
//
// ── Correctness invariant ────────────────────────────────────────────────────
// The renderer submits to a SINGLE graphics queue, in submission order, with
// `kFramesInFlight` per-slot fences. A frame recorded as monotonic serial S
// uses slot S % kFramesInFlight. At the start of the frame being recorded as
// serial S we have just returned from vkWaitForFences(inFlight[S %
// kFramesInFlight]); that fence was last signaled by the submission of frame
// S - kFramesInFlight (the previous occupant of that slot). Because the queue
// drains in order, EVERY frame with serial <= S - kFramesInFlight is therefore
// GPU-complete, and any resource whose last referencing frame's serial R
// satisfies R <= S - kFramesInFlight can no longer be read by the GPU and is
// safe to destroy. drain(S) enforces exactly `R + kFramesInFlight <= S`.
//
// Because descriptorBindingUpdateAfterBind / partiallyBound are NOT enabled on
// this device (VulkanContext.cpp), a descriptor SET referencing a resource may
// not be updated while a command buffer using it is in flight either; the
// retire queue only covers RESOURCE lifetime — descriptor-set rewrites are
// deferred separately via per-frame dirty flags in CoreImpl.
//
// ── Shutdown / teardown ──────────────────────────────────────────────────────
// Every remaining vkDeviceWaitIdle in the renderer (shutdown, swapchain
// teardown, resize realloc) must call flushAll() immediately after the wait:
// once the whole device is idle every retired resource is unreferenced, and
// leaving them queued leaks them at device destroy (VUID-vkDestroyDevice-
// device-05137, the class of bug that previously bit lineGeomCache_).

#ifndef THREEPP_VULKAN_RETIRE_QUEUE_HPP
#define THREEPP_VULKAN_RETIRE_QUEUE_HPP

#include "VulkanContext.hpp"
#include "VulkanResources.hpp"

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class RetireQueue {
    public:
        // Hand over a resource, stamped with `serial` = the serial of the frame
        // currently being recorded (its last referencing frame). The source is
        // zeroed so the caller's handle can't double-free. No-op on null handles.
        void retire(Buffer&& b, uint64_t serial) {
            if (b.handle == VK_NULL_HANDLE) return;
            buffers_.push_back({b, serial});
            b = Buffer{};
        }
        void retire(Image2D&& img, uint64_t serial) {
            if (img.image == VK_NULL_HANDLE) return;
            images_.push_back({img, serial});
            img = Image2D{};
        }
        void retireAS(VkAccelerationStructureKHR as, uint64_t serial) {
            if (as == VK_NULL_HANDLE) return;
            structures_.push_back({as, serial});
        }

        // Destroy everything whose last referencing frame has provably retired.
        // Call at frame start, right AFTER vkWaitForFences(inFlight[currentFrame]),
        // passing the serial of the frame about to be recorded and kFramesInFlight.
        void drain(VulkanContext& ctx, uint64_t currentSerial, uint32_t framesInFlight) {
            const VkDevice     d = ctx.device();
            const VmaAllocator a = ctx.allocator();
            const auto& rt = ctx.rt();
            auto ready = [&](uint64_t s) { return s + framesInFlight <= currentSerial; };

            reclaim(buffers_, ready, [&](Buffer& r) { destroyBuffer(a, r); });
            reclaim(images_, ready, [&](Image2D& r) { destroyImage2D(a, d, r); });
            reclaim(structures_, ready, [&](VkAccelerationStructureKHR& r) {
                if (rt.destroyAccelerationStructure) rt.destroyAccelerationStructure(d, r, nullptr);
            });
        }

        // Unconditionally destroy everything queued. ONLY legal after a
        // vkDeviceWaitIdle (whole device idle ⇒ nothing references these).
        void flushAll(VulkanContext& ctx) {
            const VkDevice     d = ctx.device();
            const VmaAllocator a = ctx.allocator();
            const auto& rt = ctx.rt();
            for (auto& e : buffers_) destroyBuffer(a, e.res);
            for (auto& e : images_) destroyImage2D(a, d, e.res);
            for (auto& e : structures_)
                if (rt.destroyAccelerationStructure) rt.destroyAccelerationStructure(d, e.res, nullptr);
            buffers_.clear();
            images_.clear();
            structures_.clear();
        }

        bool empty() const {
            return buffers_.empty() && images_.empty() && structures_.empty();
        }

    private:
        template<class T>
        struct Entry {
            T        res;
            uint64_t serial;
        };

        // Destroy every ready entry (destroy() zeroes the resource) and compact
        // the vector, preserving the still-pending tail.
        template<class T, class Ready, class Destroy>
        static void reclaim(std::vector<Entry<T>>& v, Ready ready, Destroy destroy) {
            size_t keep = 0;
            for (size_t i = 0; i < v.size(); ++i) {
                if (ready(v[i].serial)) {
                    destroy(v[i].res);
                } else {
                    if (keep != i) v[keep] = v[i];
                    ++keep;
                }
            }
            v.resize(keep);
        }

        std::vector<Entry<Buffer>>                        buffers_;
        std::vector<Entry<Image2D>>                       images_;
        std::vector<Entry<VkAccelerationStructureKHR>>    structures_;
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_RETIRE_QUEUE_HPP
