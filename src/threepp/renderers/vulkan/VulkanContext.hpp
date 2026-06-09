// VulkanContext — owns the Vulkan handles shared by every Vulkan-backed
// renderer / pipeline in threepp. Holds the instance, surface, physical
// device, logical device, queues, swapchain, and VMA allocator.
//
// Created once per VulkanRenderer; passed by reference into pipeline
// constructors. Recreated on swapchain-out-of-date (resize, minimize, etc.)
// via `recreateSwapchain()` which rebuilds chain + image views in place.

#ifndef THREEPP_VULKAN_CONTEXT_HPP
#define THREEPP_VULKAN_CONTEXT_HPP

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

struct GLFWwindow;

namespace threepp::vulkan {

    struct QueueFamilies {
        uint32_t graphics = UINT32_MAX;
        uint32_t present  = UINT32_MAX;
        uint32_t compute  = UINT32_MAX;
    };

    class VulkanContext {

    public:
        explicit VulkanContext(GLFWwindow* window, bool enableRayTracing, bool vsync = true);
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        // Recreate swapchain and views (on resize / out-of-date).
        // Idle-waits on the device first; safe to call any time.
        void recreateSwapchain();

        // Accessors --------------------------------------------------------
        VkInstance       instance() const { return instance_; }
        VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
        VkDevice         device() const { return device_; }
        VmaAllocator     allocator() const { return allocator_; }

        VkSurfaceKHR surface() const { return surface_; }
        VkQueue graphicsQueue() const { return graphicsQueue_; }
        VkQueue presentQueue() const { return presentQueue_; }
        VkQueue computeQueue() const { return computeQueue_; }
        const QueueFamilies& queueFamilies() const { return queueFamilies_; }

        VkSwapchainKHR     swapchain() const { return swapchain_; }
        VkFormat           swapchainFormat() const { return swapchainFormat_; }
        VkExtent2D         swapchainExtent() const { return swapchainExtent_; }
        const std::vector<VkImage>&     swapchainImages() const { return swapchainImages_; }
        const std::vector<VkImageView>& swapchainImageViews() const { return swapchainImageViews_; }

        bool rayTracingEnabled() const { return rayTracingEnabled_; }

        // NVIDIA Shader Execution Reordering (VK_NV_ray_tracing_invocation_reorder).
        // When true, the raygen pipeline loads the SER variant of the raygen
        // SPV (hitObjectNV / reorderThreadNV / hitObjectExecuteShaderNV around
        // the bounce trace) which warp-reorders threads by hit material before
        // the closest_hit invocation — typically a 10-30% speed-up on incoherent
        // diffuse-bounce paths on Ada / Ampere. Falls back to plain traceRayEXT
        // when unsupported (AMD, Intel, older NVIDIA, software fallback).
        bool rayTracingInvocationReorderSupported() const {
            return rayTracingInvocationReorderSupported_;
        }

        // VK_KHR_ray_query — inline ray tracing from any stage (compute). Used
        // by the raster-first deferred shading pass for hard shadow rays.
        // Optional; ReferencePT works without it.
        bool rayQuerySupported() const {
            return rayQuerySupported_;
        }

        // Attach a debug-utils name to a Vulkan object so validation messages
        // and RenderDoc / Nsight reports identify it by label instead of by
        // raw uint64 handle. No-op when validation is off (the EXT extension
        // isn't loaded). Cheap; safe to call from any image-creation helper.
        void setObjectName(VkImage image, const char* name) const;
        void setObjectName(VkImageView view, const char* name) const;
        void setObjectName(VkBuffer buffer, const char* name) const;

        // Ray-tracing pipeline properties. Only valid when rayTracingEnabled().
        // Needed by callers building Shader Binding Tables.
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtPipelineProperties() const {
            return rtPipelineProperties_;
        }

        // KHR ray-tracing entry points loaded via vkGetDeviceProcAddr at
        // device creation. All members are non-null only when
        // rayTracingEnabled() is true.
        struct RtFunctions {
            PFN_vkCreateAccelerationStructureKHR        createAccelerationStructure        = nullptr;
            PFN_vkDestroyAccelerationStructureKHR       destroyAccelerationStructure       = nullptr;
            PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizes = nullptr;
            PFN_vkCmdBuildAccelerationStructuresKHR     cmdBuildAccelerationStructures     = nullptr;
            PFN_vkGetAccelerationStructureDeviceAddressKHR
                                                        getAccelerationStructureDeviceAddress = nullptr;
            PFN_vkCreateRayTracingPipelinesKHR          createRayTracingPipelines          = nullptr;
            PFN_vkGetRayTracingShaderGroupHandlesKHR    getRayTracingShaderGroupHandles    = nullptr;
            PFN_vkCmdTraceRaysKHR                       cmdTraceRays                       = nullptr;
        };
        const RtFunctions& rt() const { return rt_; }

    private:
        GLFWwindow* window_ = nullptr;

        VkInstance               instance_       = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
        VkSurfaceKHR             surface_        = VK_NULL_HANDLE;
        VkPhysicalDevice         physicalDevice_ = VK_NULL_HANDLE;
        VkDevice                 device_         = VK_NULL_HANDLE;
        VmaAllocator             allocator_      = VK_NULL_HANDLE;

        QueueFamilies queueFamilies_{};
        VkQueue graphicsQueue_ = VK_NULL_HANDLE;
        VkQueue presentQueue_  = VK_NULL_HANDLE;
        VkQueue computeQueue_  = VK_NULL_HANDLE;

        VkSwapchainKHR           swapchain_       = VK_NULL_HANDLE;
        VkFormat                 swapchainFormat_ = VK_FORMAT_UNDEFINED;
        VkExtent2D               swapchainExtent_{};
        std::vector<VkImage>     swapchainImages_;
        std::vector<VkImageView> swapchainImageViews_;

        bool vsync_ = true;// FIFO when true, else MAILBOX/IMMEDIATE (see createSwapchain)
        bool rayTracingEnabled_ = false;
        bool rayTracingInvocationReorderSupported_ = false;
        bool rayQuerySupported_ = false;
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties_{};
        RtFunctions rt_{};
        // Non-null only when VK_EXT_debug_utils is enabled (i.e. validation
        // is on). Loaded via vkGetDeviceProcAddr at device creation.
        PFN_vkSetDebugUtilsObjectNameEXT setObjectNameFn_ = nullptr;

        void createInstance(bool enableValidation);
        void createDebugMessenger();
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createAllocator();
        void createSwapchain();
        void createSwapchainImageViews();
        void destroySwapchainResources();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_CONTEXT_HPP
