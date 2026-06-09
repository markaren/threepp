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

namespace threepp::vulkan {

    // Throws std::runtime_error with a descriptive prefix when `r != VK_SUCCESS`.
    // Used pervasively as `check(vkXxx(...), "vkXxx")` to keep call sites short.
    void check(VkResult r, const char* what);

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
    // populates the .address field via vkGetBufferDeviceAddress.
    Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VmaMemoryUsage memoryUsage,
                        VmaAllocationCreateFlags flags = 0);

    // Free a buffer (idempotent — safe to call on a zero-initialized Buffer).
    // Zeroes the struct on exit so subsequent calls are no-ops.
    void destroyBuffer(VmaAllocator alloc, Buffer& b);

    // Dedicated EXPORTABLE device-local buffer for cross-API interop (CUDA).
    // Allocated outside VMA: export requires VkExportMemoryAllocateInfo +
    // dedicated allocation, which VMA pools don't carry. `osHandle` is the
    // exported OS handle — a Win32 NT handle on Windows (owned by us; closed
    // in destroyExternalBuffer — CUDA's import duplicates it), a POSIX fd
    // elsewhere (ownership transfers to CUDA on successful import, so it is
    // set to null here once handed out and NOT closed by destroy).
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
    Buffer createAsScratchBuffer(VmaAllocator alloc, VkDevice device, VkDeviceSize size);

    // Minimal sampled-image record. Used both for textures (env, materials,
    // blue-noise tile) and for storage images (accumulator, gbuf, ReSTIR DI
    // reservoirs, à-trous ping-pong). Empty `sampler` slot for storage-only
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

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_RESOURCES_HPP
