#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

namespace threepp::vulkan {

    void check(VkResult r, const char* what) {
        if (r != VK_SUCCESS) {
            throw std::runtime_error(std::string("[VulkanRenderer] ") + what + " failed: " + std::to_string(r));
        }
    }

    Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VmaMemoryUsage memoryUsage,
                        VmaAllocationCreateFlags flags) {
        Buffer b{};
        b.size = size;

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage = memoryUsage;
        aci.flags = flags;

        check(vmaCreateBuffer(alloc, &bci, &aci, &b.handle, &b.alloc, nullptr),
              "vmaCreateBuffer");

        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            VkBufferDeviceAddressInfo dai{};
            dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            dai.buffer = b.handle;
            b.address = vkGetBufferDeviceAddress(device, &dai);
        }
        return b;
    }

    void destroyBuffer(VmaAllocator alloc, Buffer& b) {
        if (b.handle) vmaDestroyBuffer(alloc, b.handle, b.alloc);
        b = {};
    }

    Buffer createAsScratchBuffer(VmaAllocator alloc, VkDevice device, VkDeviceSize size) {
        Buffer b{};
        b.size = size;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateBufferWithAlignment(alloc, &bci, &aci,
                                            /*minAlignment*/ 256,
                                            &b.handle, &b.alloc, nullptr),
              "vmaCreateBufferWithAlignment(scratch)");
        VkBufferDeviceAddressInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        dai.buffer = b.handle;
        b.address = vkGetBufferDeviceAddress(device, &dai);
        return b;
    }

    ExternalBuffer createExternalBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                                        VkDeviceSize size, VkBufferUsageFlags usage) {
#ifdef _WIN32
        constexpr VkExternalMemoryHandleTypeFlagBits kHandleType =
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        constexpr VkExternalMemoryHandleTypeFlagBits kHandleType =
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        ExternalBuffer b{};
        b.size = size;

        VkExternalMemoryBufferCreateInfo emb{};
        emb.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        emb.handleTypes = kHandleType;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &emb;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &bci, nullptr, &b.handle), "vkCreateBuffer(external)");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b.handle, &req);

        // Device-local memory type from the requirement mask. CUDA's import maps
        // the same physical pages, so device-local is both the fast and the
        // correct choice (no host round-trip anywhere in the chain).
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        uint32_t typeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if (!(req.memoryTypeBits & (1u << i))) continue;
            if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                typeIndex = i;
                break;
            }
        }
        if (typeIndex == UINT32_MAX) {
            vkDestroyBuffer(device, b.handle, nullptr);
            throw std::runtime_error("[VulkanRenderer] no device-local memory type for external buffer");
        }

        // Dedicated allocation: CUDA's OPAQUE_WIN32/FD import requires (and the
        // CUDA_EXTERNAL_MEMORY_DEDICATED flag on the import side declares) a
        // dedicated VkDeviceMemory backing exactly this buffer.
        VkMemoryDedicatedAllocateInfo ded{};
        ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ded.buffer = b.handle;
        VkExportMemoryAllocateInfo exp{};
        exp.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exp.pNext = &ded;
        exp.handleTypes = kHandleType;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &exp;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = typeIndex;
        check(vkAllocateMemory(device, &mai, nullptr, &b.memory), "vkAllocateMemory(external)");
        check(vkBindBufferMemory(device, b.handle, b.memory, 0), "vkBindBufferMemory(external)");

#ifdef _WIN32
        auto getHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
        if (!getHandle) {
            destroyExternalBuffer(device, b);
            throw std::runtime_error("[VulkanRenderer] vkGetMemoryWin32HandleKHR not available");
        }
        VkMemoryGetWin32HandleInfoKHR ghi{};
        ghi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        ghi.memory = b.memory;
        ghi.handleType = kHandleType;
        HANDLE h = nullptr;
        check(getHandle(device, &ghi, &h), "vkGetMemoryWin32HandleKHR");
        b.osHandle = h;
#else
        auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        if (!getFd) {
            destroyExternalBuffer(device, b);
            throw std::runtime_error("[VulkanRenderer] vkGetMemoryFdKHR not available");
        }
        VkMemoryGetFdInfoKHR gfi{};
        gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gfi.memory = b.memory;
        gfi.handleType = kHandleType;
        int fd = -1;
        check(getFd(device, &gfi, &fd), "vkGetMemoryFdKHR");
        b.osHandle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif
        return b;
    }

    void destroyExternalBuffer(VkDevice device, ExternalBuffer& b) {
#ifdef _WIN32
        if (b.osHandle) CloseHandle(static_cast<HANDLE>(b.osHandle));
#endif
        // POSIX fd: ownership transferred to the importer (CUDA) — never closed here.
        if (b.handle) vkDestroyBuffer(device, b.handle, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }

    void destroyImage2D(VmaAllocator alloc, VkDevice device, Image2D& img) {
        if (img.sampler) vkDestroySampler(device, img.sampler, nullptr);
        if (img.view)    vkDestroyImageView(device, img.view, nullptr);
        if (img.image)   vmaDestroyImage(alloc, img.image, img.alloc);
        img = {};
    }

}// namespace threepp::vulkan
