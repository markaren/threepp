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
        explicit VulkanContext(GLFWwindow* window, bool enableRayTracing, bool vsync = true,
                               bool preferHeadlessSurface = false);
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

        // Shared pipeline cache, loaded from disk at construction and saved at
        // destruction. Pass to every vkCreate{Graphics,Compute,RayTracing}
        // Pipelines call so repeat launches skip the (multi-second, esp. for
        // the RT megakernel) cold pipeline compile. Purely an optimization —
        // the driver validates the on-disk blob against the device and it is
        // discarded if incompatible, so it can never affect correctness.
        VkPipelineCache pipelineCache() const { return pipelineCache_; }

        // Persist the cache during the run, not only at destruction. The
        // destructor save is the one a kill by PID never reaches, and killing
        // by PID is how this project measures, so a 95 s cold compile was
        // routinely paid and then thrown away. Called once per frame from the
        // renderer and throttled here: a size-only vkGetPipelineCacheData is
        // cheap, and the write happens only when that size has moved since the
        // last save, i.e. when a pipeline was genuinely compiled in this
        // process. The first check runs after the first frame, which is where
        // the startup compiles have all just landed.
        void savePipelineCacheIfChanged();

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
        // True when the surface allowed TRANSFER_SRC swapchain usage — the
        // precondition for every path that copies the presented image out
        // (readRGBPixels, scene capture). See createSwapchain.
        bool swapchainSupportsTransferSrc() const { return swapchainTransferSrc_; }
        // True when shaderStorageImageReadWithoutFormat was available and
        // enabled on the device — the precondition for a compute pass
        // imageLoad()-ing the BGRA8 swapchain through a format-less storage
        // image (GLSL has no bgra8 qualifier; rgba8 mismatches the view). The
        // event camera's Final source needs it; universal on desktop GPUs.
        bool storageImageReadWithoutFormat() const { return storageImageReadWithoutFormat_; }

        bool rayTracingEnabled() const { return rayTracingEnabled_; }

        // True when the surface is a VK_EXT_headless_surface (headless canvas
        // on a supporting ICD). The swapchain machinery runs unchanged, but
        // presentation is a no-op sink — nothing is displayed and no
        // window-system connection is needed, which is what lets the renderer
        // run on display-less machines (cloud GPU instances).
        bool headlessSurface() const { return headlessSurface_; }

        // True when this swapchain must never be presented to: a headless canvas
        // (nothing is displayed) that additionally asked for it via
        // THREEPP_VULKAN_SUPPRESS_PRESENT=1. Presenting to a hidden window costs
        // a host-side wait for the presented image's rendering to complete on
        // Windows/NVIDIA — 5.5-10.6 ms per frame, exactly the GPU frame
        // duration, so the CPU can never get a frame ahead. Off by default
        // because removing that stall measured SLOWER wherever GPU compute is
        // co-resident; see the derivation and the numbers at the assignment in
        // the constructor, and acquireOrReuseSwapchainImage (VulkanCoreFrame)
        // for what replaces the acquire/present cycle when it is on.
        bool presentSuppressed() const { return presentSuppressed_; }

        // VK_KHR_ray_query — inline ray tracing from any stage (compute). Used
        // by the raster-first deferred shading pass for hard shadow rays.
        // Optional; ReferencePT works without it.
        bool rayQuerySupported() const {
            return rayQuerySupported_;
        }

        // Exportable external memory (VK_KHR_external_memory_win32 / _fd) —
        // lets device-local buffers be shared zero-copy with CUDA (PhysX
        // soft-body tet positions). Optional; everything falls back to the
        // host-visible upload path without it.
        bool externalMemorySupported() const {
            return externalMemorySupported_;
        }

        // VK_KHR_pipeline_executable_properties — the DRIVER's own report of a
        // pipeline's register count / scratch (spill) usage / occupancy, which
        // is the only trustworthy way to tell whether a shader change moved
        // occupancy rather than just wall-clock. Opt-in via
        // THREEPP_VULKAN_PIPELINE_STATS=1 because capturing statistics forces
        // pipelines to be compiled with an extra flag (and defeats the pipeline
        // cache), so it is a diagnostic mode, not something to leave on.
        bool pipelineStatsEnabled() const {
            return pipelineStatsEnabled_;
        }

        // Print every executable's statistics for `pipe` to stdout, tagged with
        // `label`. No-op unless pipelineStatsEnabled(). Call right after
        // pipeline creation.
        void dumpPipelineStats(VkPipeline pipe, const char* label) const;

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
        bool                     swapchainTransferSrc_ = false;
        bool                     storageImageReadWithoutFormat_ = false;

        bool vsync_ = true;// FIFO when true, else MAILBOX/IMMEDIATE (see createSwapchain)
        bool headlessSurface_ = false;// VK_EXT_headless_surface instead of a window surface
        bool presentSuppressed_ = false;// opt-in headless mode: acquire once per slot, never present
        bool rayTracingEnabled_ = false;
        bool rayQuerySupported_ = false;
        bool externalMemorySupported_ = false;
        bool pipelineStatsEnabled_ = false;
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties_{};
        RtFunctions rt_{};
        // Non-null only when VK_EXT_debug_utils is enabled (i.e. validation
        // is on). Loaded via vkGetDeviceProcAddr at device creation.
        PFN_vkSetDebugUtilsObjectNameEXT setObjectNameFn_ = nullptr;

        VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
        // Bytes on disk after the last successful save (or keyed load), so a
        // save is skipped when the driver reports the same size back.
        size_t savedPipelineCacheBytes_ = 0;
        uint32_t pipelineCacheTick_ = 0;

        void createInstance(bool enableValidation);
        void createDebugMessenger();
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createAllocator();
        // Load the on-disk pipeline cache (validated against this device) at
        // startup; persist it back to disk at shutdown.
        void createPipelineCache();
        void savePipelineCache();
        void createSwapchain();
        void createSwapchainImageViews();
        void destroySwapchainResources();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_CONTEXT_HPP
