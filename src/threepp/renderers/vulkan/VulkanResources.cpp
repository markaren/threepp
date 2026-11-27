#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/TransientPool.hpp"
#include "threepp/renderers/vulkan/EnvKnob.hpp"
#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"// routes vkCreateShaderModule (NoContraction)

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

    VkShaderModule createShaderModule(VkDevice device, const uint32_t* spv, size_t bytes, const char* what) {
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = bytes;
        smci.pCode    = spv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device, &smci, nullptr, &mod), what);
        return mod;
    }

    void submitOneShot(VkDevice device, VkQueue queue, VkCommandPool pool, const char* what,
                       const std::function<void(VkCommandBuffer)>& record) {
        const auto label = [what](const char* step) { return std::string(step) + "(" + what + ")"; };

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(device, &ai, &cb), label("alloc one-shot cb").c_str());

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), label("begin one-shot cb").c_str());
        record(cb);
        check(vkEndCommandBuffer(cb), label("end one-shot cb").c_str());

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), label("submit one-shot").c_str());
        check(vkQueueWaitIdle(queue), label("wait one-shot").c_str());
        vkFreeCommandBuffers(device, pool, 1, &cb);
    }

    VkPipeline createComputePipeline(VkDevice device, VkPipelineCache cache, VkPipelineLayout layout,
                                     const uint32_t* spv, size_t spvBytes, const char* what,
                                     const VkSpecializationInfo* spec) {
        const auto checkNamed = [what](VkResult r, const char* call) {
            if (r != VK_SUCCESS) check(r, (std::string(call) + "(" + what + ")").c_str());
        };

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spvBytes;
        smci.pCode    = spv;
        VkShaderModule mod = VK_NULL_HANDLE;
        checkNamed(vkCreateShaderModule(device, &smci, nullptr, &mod), "vkCreateShaderModule");

        VkComputePipelineCreateInfo cpci{};
        cpci.sType                     = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module              = mod;
        cpci.stage.pName               = "main";
        cpci.stage.pSpecializationInfo = spec;
        cpci.layout                    = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult r = vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipe);
        // The module is not needed past this call, and is released before a
        // failure throws.
        vkDestroyShaderModule(device, mod, nullptr);
        checkNamed(r, "vkCreateComputePipelines");
        return pipe;
    }

    namespace {
        // The live-buffer registry behind forEachLiveBuffer (memory dump only).
        bool registryOn() {
            static const bool on = env::flag("THREEPP_VK_MEMDUMP");
            return on;
        }
        std::mutex& registryMutex() {
            static std::mutex m;
            return m;
        }
        std::unordered_map<VmaAllocation, LiveBufferSite>& registry() {
            static std::unordered_map<VmaAllocation, LiveBufferSite> r;
            return r;
        }
        void registerBuffer(VmaAllocation a, const std::source_location& where,
                            VkBufferUsageFlags usage, VkDeviceSize size) {
            if (!registryOn() || a == VK_NULL_HANDLE) return;
            std::lock_guard<std::mutex> lk(registryMutex());
            registry()[a] = {where.file_name(), static_cast<uint32_t>(where.line()), usage, size};
        }
    }// namespace

    void forEachLiveBuffer(const std::function<void(VmaAllocation, const LiveBufferSite&)>& fn) {
        if (!registryOn()) return;
        std::lock_guard<std::mutex> lk(registryMutex());
        for (const auto& [a, site] : registry()) fn(a, site);
    }

    Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VmaMemoryUsage memoryUsage,
                        VmaAllocationCreateFlags flags,
                        std::source_location where) {
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
        registerBuffer(b.alloc, where, usage, size);

        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            VkBufferDeviceAddressInfo dai{};
            dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            dai.buffer = b.handle;
            b.address = vkGetBufferDeviceAddress(device, &dai);
        }
        return b;
    }

    void destroyBuffer(VmaAllocator alloc, Buffer& b) {
        if (b.alloc && registryOn()) {
            std::lock_guard<std::mutex> lk(registryMutex());
            registry().erase(b.alloc);
        }
        if (b.handle) vmaDestroyBuffer(alloc, b.handle, b.alloc);
        b = {};
    }

    Buffer createAsScratchBuffer(VmaAllocator alloc, VkDevice device, VkDeviceSize size,
                                 std::source_location where) {
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
        registerBuffer(b.alloc, where, bci.usage, size);
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
        // The ALLOCATION size, not the requested one — see the ExternalBuffer
        // comment. mai.allocationSize below is req.size, and that is the number
        // a dedicated cuImportExternalMemory has to be told about; recording the
        // caller's (smaller) `size` here is how an importer silently ends up
        // mapping a short range.
        b.size = req.size;

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

    void* takeOsHandle(VkDevice device, ExternalBuffer& b) {
        if (b.memory == VK_NULL_HANDLE) return nullptr;
#ifdef _WIN32
        (void) device;// the NT handle is ours to keep — hand out the same value
        return b.osHandle;
#else
        if (b.osHandle) {
            void* h = b.osHandle;
            b.osHandle = nullptr;// ownership transferred to the importer
            return h;
        }
        // Second and later hand-outs: the previous fd belongs to (and has
        // probably been closed by) an earlier importer, so mint a new one.
        // vkGetMemoryFdKHR is documented to return a NEW fd on every call.
        auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        if (!getFd) return nullptr;
        VkMemoryGetFdInfoKHR gfi{};
        gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gfi.memory = b.memory;
        gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (getFd(device, &gfi, &fd) != VK_SUCCESS) return nullptr;
        return reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif
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
        if (img.view) {
            unregisterImageView(img.view);
            vkDestroyImageView(device, img.view, nullptr);
        }
        // A TransientPool image shares its memory; the pool destroys it and
        // releases its range. Anything else owns a VMA allocation.
        if (img.image && !TransientPool::release(img.image)) vmaDestroyImage(alloc, img.image, img.alloc);
        img = {};
    }

}// namespace threepp::vulkan
