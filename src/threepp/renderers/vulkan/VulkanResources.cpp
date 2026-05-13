#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <stdexcept>
#include <string>

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

    void destroyImage2D(VmaAllocator alloc, VkDevice device, Image2D& img) {
        if (img.sampler) vkDestroySampler(device, img.sampler, nullptr);
        if (img.view)    vkDestroyImageView(device, img.view, nullptr);
        if (img.image)   vmaDestroyImage(alloc, img.image, img.alloc);
        img = {};
    }

}// namespace threepp::vulkan
