// VulkanResources — small POD wrappers + RAII-style helper functions for
// the buffer / image plumbing shared across the Vulkan renderer code.
//
// Extracted from VulkanRenderer.cpp during the incremental file split (the
// monolith was 11k+ lines and these helpers had no dependency on the
// renderer's per-frame state, so they're the lowest-risk piece to pull
// first). All functions are free functions taking the allocator/device
// they need explicitly — no global state.

#ifndef THREEPP_VULKAN_RESOURCES_HPP
#define THREEPP_VULKAN_RESOURCES_HPP

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <source_location>

namespace threepp::vulkan {

    // Throws std::runtime_error with a descriptive prefix when `r != VK_SUCCESS`.
    // Used pervasively as `check(vkXxx(...), "vkXxx")` to keep call sites short.
    void check(VkResult r, const char* what);

    // A command buffer outside any frame, for resource setup (a fresh image's
    // first layout transition, a one-off clear or upload): allocated from
    // `pool`, recorded by `record`, submitted to `queue`, waited for with
    // vkQueueWaitIdle and freed. `what` names the caller in a failure's
    // message ("alloc one-shot cb(what)", ...).
    void submitOneShot(VkDevice device, VkQueue queue, VkCommandPool pool, const char* what,
                       const std::function<void(VkCommandBuffer)>& record);

    // A shader module from SPIR-V words (`bytes` long). A failure throws
    // through check() with `what` as its label.
    VkShaderModule createShaderModule(VkDevice device, const uint32_t* spv, size_t bytes, const char* what);

    // A compute pipeline from one SPIR-V module: the module is created, the
    // pipeline built on `layout` with entry point "main", and the module
    // destroyed again whether or not the pipeline came to be. `cache` may be
    // VK_NULL_HANDLE. `spec`: the stage's specialization constants, or null.
    // `what` names the pipeline in the error a failure throws through check():
    // "vkCreateShaderModule(what)" or "vkCreateComputePipelines(what)". The
    // module goes through the backend's vkCreateShaderModule
    // (VulkanContext.hpp), so the NoContraction option reaches it.
    [[nodiscard]] VkPipeline createComputePipeline(VkDevice device, VkPipelineCache cache, VkPipelineLayout layout,
                                                   const uint32_t* spv, size_t spvBytes, const char* what,
                                                   const VkSpecializationInfo* spec = nullptr);

    // Round x up to the nearest multiple of `align` (align must be a power of two).
    [[nodiscard]] inline uint32_t alignUp(uint32_t x, uint32_t align) {
        return (x + align - 1) & ~(align - 1);
    }

    // Minimal Vulkan buffer record: handle + allocation + size + (optional)
    // device address. .address is populated when the buffer was created with
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
    struct Buffer {
        VkBuffer        handle  = VK_NULL_HANDLE;
        VmaAllocation   alloc   = VK_NULL_HANDLE;
        VkDeviceSize    size    = 0;
        VkDeviceAddress address = 0;
    };

    // Allocate a buffer. If usage includes SHADER_DEVICE_ADDRESS_BIT,
    // populates the .address field via vkGetBufferDeviceAddress. `where` is
    // the caller's source line, kept for the memory dump (see
    // forEachLiveBuffer); callers never pass it.
    Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VmaMemoryUsage memoryUsage,
                        VmaAllocationCreateFlags flags = 0,
                        std::source_location where = std::source_location::current());

    // Free a buffer (idempotent — safe to call on a zero-initialized Buffer).
    // Zeroes the struct on exit so subsequent calls are no-ops.
    void destroyBuffer(VmaAllocator alloc, Buffer& b);

    // THREEPP_VK_MEMDUMP=1 (read once) keeps a registry of every live buffer
    // createBuffer and createAsScratchBuffer made: its creating source line,
    // usage and size. Without the variable nothing is recorded and the walk
    // below visits nothing. Lets the memory dump say WHICH buffers sit in
    // the BAR window or in system memory.
    struct LiveBufferSite {
        const char*        file  = nullptr;// the creating call's source file
        uint32_t           line  = 0;
        VkBufferUsageFlags usage = 0;
        VkDeviceSize       size  = 0;
    };
    void forEachLiveBuffer(const std::function<void(VmaAllocation, const LiveBufferSite&)>& fn);

    // ── Host-visible memory traffic (portable flush/invalidate) ─────────
    // Desktop GPUs expose HOST_COHERENT on every host-visible heap, so raw
    // map→memcpy→unmap "just works" there — but the Vulkan spec doesn't
    // guarantee coherent host-visible memory, and on non-coherent heaps
    // host writes need vmaFlushAllocation before the GPU reads them (and
    // host reads need vmaInvalidateAllocation after the GPU writes). Both
    // are no-ops on coherent memory, so routing ALL host-visible traffic
    // through these helpers costs nothing on current hardware and removes
    // the portability hole. VMA rounds the ranges to nonCoherentAtomSize
    // internally. The VkResults are check()ed — these calls only fail on
    // invalid arguments (a bug), never at runtime on a healthy device.

    // Whole-blob host→GPU upload: map → memcpy → flush the written range →
    // unmap. The standard replacement for the map/memcpy/unmap trio.
    inline void uploadHostVisible(VmaAllocator alloc, const Buffer& b,
                                  const void* src, VkDeviceSize bytes,
                                  VkDeviceSize dstOffset = 0) {
        void* mapped = nullptr;
        check(vmaMapMemory(alloc, b.alloc, &mapped), "vmaMapMemory(uploadHostVisible)");
        std::memcpy(static_cast<uint8_t*>(mapped) + dstOffset, src, bytes);
        check(vmaFlushAllocation(alloc, b.alloc, dstOffset, bytes),
              "vmaFlushAllocation(uploadHostVisible)");
        vmaUnmapMemory(alloc, b.alloc);
    }

    // Flush after STRUCTURED writes through a mapped pointer (field-by-field
    // stores, loops, persistently-mapped per-frame batches): call once after
    // the write batch, before the buffer is consumed. Use the defaults
    // unless the exact written range is trivially at hand.
    inline void flushHostWrites(VmaAllocator alloc, VmaAllocation allocation,
                                VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) {
        check(vmaFlushAllocation(alloc, allocation, offset, size),
              "vmaFlushAllocation(flushHostWrites)");
    }

    // Mirror direction: invalidate BEFORE reading GPU-written data through a
    // mapped pointer (readbacks: pixels, lidar returns, event streams, ...).
    inline void invalidateHostReads(VmaAllocator alloc, VmaAllocation allocation,
                                    VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) {
        check(vmaInvalidateAllocation(alloc, allocation, offset, size),
              "vmaInvalidateAllocation(invalidateHostReads)");
    }

    // Dedicated EXPORTABLE device-local buffer for cross-API interop (CUDA).
    // Allocated outside VMA: export requires VkExportMemoryAllocateInfo +
    // dedicated allocation, which VMA pools don't carry. `osHandle` is the
    // exported OS handle — a Win32 NT handle on Windows (owned by us; closed
    // in destroyExternalBuffer — CUDA's import duplicates it), a POSIX fd
    // elsewhere (ownership transfers to CUDA on successful import, so it is
    // set to null here once handed out and NOT closed by destroy).
    //
    // `size` is the ALLOCATION size (VkMemoryRequirements::size), NOT the size
    // the caller asked for: the driver rounds allocations up, and CUDA's
    // cuImportExternalMemory for a DEDICATED allocation wants the size of the
    // VkDeviceMemory, not of the buffer bound to it. Importing with the smaller
    // number succeeds and maps a SHORT range — writes past it are out of bounds
    // with no error anywhere in the chain, which is why this field carries the
    // full allocation size. Mapping the whole allocation from
    // CUDA is legal; the extra tail is simply unused by Vulkan.
    struct ExternalBuffer {
        VkBuffer       handle   = VK_NULL_HANDLE;
        VkDeviceMemory memory   = VK_NULL_HANDLE;
        VkDeviceSize   size     = 0;
        void*          osHandle = nullptr;
    };

    // Create an exportable dedicated buffer. Requires the platform external-
    // memory extension on the device (VulkanContext::externalMemorySupported()).
    // Throws on failure (same contract as the other helpers).
    ExternalBuffer createExternalBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                                        VkDeviceSize size, VkBufferUsageFlags usage);

    // Hand the OS handle to an importer (CUDA). ALWAYS go through this rather
    // than reading `b.osHandle` directly at a hand-out site, because the two
    // platforms have opposite ownership rules:
    //
    //   Windows — the NT handle stays OURS. CUDA's import duplicates it, so the
    //   same value may be handed out any number of times and destroyExternalBuffer
    //   closes it exactly once. Returns b.osHandle unchanged.
    //
    //   POSIX — the fd's ownership TRANSFERS to the importer, which closes it.
    //   Handing the same fd out twice therefore gives the second importer an fd
    //   the first one already closed (or, worse, one the process has since
    //   recycled for an unrelated file). So the stored fd is taken and nulled on
    //   the first hand-out, and every later hand-out MINTS A FRESH fd from the
    //   same VkDeviceMemory — each independently owned by its importer. That
    //   keeps the "re-enable returns a usable handle" contract the interop
    //   entry points advertise, which a bare null-and-return would break.
    //
    // Returns nullptr if the buffer has no memory (default-constructed record).
    void* takeOsHandle(VkDevice device, ExternalBuffer& b);

    // Free an external buffer (idempotent; closes the Win32 handle).
    void destroyExternalBuffer(VkDevice device, ExternalBuffer& b);

    // Acceleration-structure scratch buffer with the alignment the spec demands:
    // VkAccelerationStructureBuildGeometryInfoKHR::scratchData::deviceAddress
    // must be aligned to
    // VkPhysicalDeviceAccelerationStructurePropertiesKHR::
    // minAccelerationStructureScratchOffsetAlignment.
    // NVIDIA reports 128B, AMD 256B, Intel up to 1024B. The plain createBuffer
    // path falls through to whatever alignment VMA picks (typically 64B from
    // VkMemoryRequirements), which on AMD/Intel produces a misaligned address
    // and the build faults with VK_ERROR_DEVICE_LOST. 256B is conservative for
    // desktop GPUs; if a future device demands more, upgrade to a property-query.
    Buffer createAsScratchBuffer(VmaAllocator alloc, VkDevice device, VkDeviceSize size,
                                 std::source_location where = std::source_location::current());

    // Minimal sampled-image record. Used both for textures (env, materials,
    // blue-noise tile) and for storage images (accumulator, gbuf, à-trous
    // ping-pong). Empty `sampler` slot for storage-only
    // images; .mipLevels > 1 for prefiltered cube/equirect (PMREM).
    struct Image2D {
        VkImage       image     = VK_NULL_HANDLE;
        VmaAllocation alloc     = VK_NULL_HANDLE;
        VkImageView   view      = VK_NULL_HANDLE;
        VkSampler     sampler   = VK_NULL_HANDLE;
        uint32_t      width     = 0;
        uint32_t      height    = 0;
        uint32_t      mipLevels = 1;
        VkFormat      format    = VK_FORMAT_UNDEFINED;
    };

    // Free an image (idempotent). Destroys the sampler + view + image in
    // order and zeroes the struct.
    void destroyImage2D(VmaAllocator alloc, VkDevice device, Image2D& img);

    // The renderer's frame-serial retire queue (RetireQueue, through
    // Impl::retire) as the passes that free resources mid-record see it: the
    // resource is destroyed once every frame in flight that may still name it
    // has finished, with no device-wide wait. It zeroes what it takes. The
    // passes that take one require it.
    using RetireBufferFn = std::function<void(Buffer&&)>;
    using RetireImageFn  = std::function<void(Image2D&&)>;

    // Moves `image` (one mip, one layer) from oldLayout to newLayout behind an
    // all-commands, all-memory barrier: the upscalers' once-per-frame borrow
    // and restore of their inputs around a vendor SDK (NGX, FidelityFX) that
    // records its own work in between. One barrier per resource per frame, so
    // broad beats exact. A restore passes UNDEFINED as oldLayout: the inputs'
    // contents are not needed after the upscale (colour and depth are
    // regenerated next frame), so nothing depends on whatever working layout
    // the SDK left them in.
    void broadTransition(VkCommandBuffer cb, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                         VkImageAspectFlags aspect);

    // vkCreateImageView, registering the view's image and subresource range
    // for the render graph's descriptor-derived declarations (see
    // DescriptorShadow.hpp). Every view the backend creates goes through this.
    VkResult createImageView(VkDevice device, const VkImageViewCreateInfo* info,
                             const VkAllocationCallbacks* allocator, VkImageView* view);

    // A number for THIS view object: assigned when createImageView registers
    // it and never handed out again. A handle VALUE is not an identity — a
    // driver gives a destroyed view's value to the next one created at the
    // same size, and the validation layers (which hand out unique handles)
    // hide it — so "does my descriptor set already name this view" compares
    // serials, never handles. 0 for a null view. A view that never went through
    // createImageView gets a new number on every call: a cache keyed on it
    // always misses, which is slow and visible instead of stale and silent.
    [[nodiscard]] uint64_t imageViewSerial(VkImageView view);

    // vkUpdateDescriptorSets, recording each write in the descriptor shadow
    // (DescriptorShadow.hpp). Every descriptor update the backend makes goes
    // through this.
    void updateDescriptorSets(VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* writes,
                              uint32_t copyCount, const VkCopyDescriptorSet* copies);

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_RESOURCES_HPP
