#include "VulkanContext.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace threepp::vulkan {

    namespace {

        constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

        constexpr std::array<const char*, 1> kBaseDeviceExtensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        constexpr std::array<const char*, 4> kRayTracingExtensions{
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        };

        bool hasInstanceLayer(const char* name) {
            uint32_t n = 0;
            vkEnumerateInstanceLayerProperties(&n, nullptr);
            std::vector<VkLayerProperties> layers(n);
            vkEnumerateInstanceLayerProperties(&n, layers.data());
            for (const auto& l : layers) {
                if (std::strcmp(l.layerName, name) == 0) return true;
            }
            return false;
        }

        std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice dev) {
            uint32_t n = 0;
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> exts(n);
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data());
            return exts;
        }

        bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
            for (const auto& e : exts) {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT* data,
                void*) {
            if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                std::cerr << "[Vulkan] " << data->pMessage << "\n";
            }
            return VK_FALSE;
        }

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("[VulkanContext] ") + what + " failed: " + std::to_string(r));
            }
        }

    }// namespace

    VulkanContext::VulkanContext(GLFWwindow* window, bool enableRayTracing, bool vsync)
        : window_(window), vsync_(vsync) {

#ifndef NDEBUG
        const bool enableValidation = hasInstanceLayer(kValidationLayer);
        if (!enableValidation) {
            std::cerr << "[VulkanContext] " << kValidationLayer
                      << " not found; running without validation.\n";
        }
#else
        const bool enableValidation = false;
#endif
        rayTracingEnabled_ = enableRayTracing;

        createInstance(enableValidation);
        if (enableValidation) createDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createAllocator();
        createSwapchain();
        createSwapchainImageViews();
    }

    VulkanContext::~VulkanContext() {
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);

        destroySwapchainResources();

        if (allocator_ != VK_NULL_HANDLE) vmaDestroyAllocator(allocator_);
        if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
        if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);

        if (debugMessenger_ != VK_NULL_HANDLE) {
            auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
                    instance_, "vkDestroyDebugUtilsMessengerEXT");
            if (fn) fn(instance_, debugMessenger_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

    void VulkanContext::createInstance(bool enableValidation) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "threepp";
        app.pEngineName = "threepp";
        app.apiVersion = VK_API_VERSION_1_3;

        uint32_t glfwExtCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
        if (enableValidation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        std::vector<const char*> layers;
        if (enableValidation) layers.push_back(kValidationLayer);

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ci.ppEnabledLayerNames = layers.data();

        check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
    }

    void VulkanContext::createDebugMessenger() {
        VkDebugUtilsMessengerCreateInfoEXT ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        ci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        ci.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        ci.pfnUserCallback = debugCallback;

        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
                instance_, "vkCreateDebugUtilsMessengerEXT");
        if (fn) check(fn(instance_, &ci, nullptr, &debugMessenger_), "vkCreateDebugUtilsMessengerEXT");
    }

    void VulkanContext::createSurface() {
        check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
              "glfwCreateWindowSurface");
    }

    void VulkanContext::pickPhysicalDevice() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (n == 0) throw std::runtime_error("[VulkanContext] no Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance_, &n, devs.data());

        // Prefer discrete GPU with required extensions.
        auto deviceScore = [&](VkPhysicalDevice d) -> int {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);

            const auto exts = deviceExtensions(d);
            for (const auto* base : kBaseDeviceExtensions) {
                if (!hasExtension(exts, base)) return -1;
            }
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
            if (rayTracingEnabled_) {
                bool allRT = true;
                for (const auto* rt : kRayTracingExtensions) {
                    if (!hasExtension(exts, rt)) { allRT = false; break; }
                }
                if (!allRT) return -1;
                score += 100;
            }
            return score;
        };

        int bestScore = -1;
        for (auto d : devs) {
            int s = deviceScore(d);
            if (s > bestScore) {
                bestScore = s;
                physicalDevice_ = d;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error(rayTracingEnabled_
                ? "[VulkanContext] no GPU with ray-tracing extensions found"
                : "[VulkanContext] no GPU with swapchain support found");
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        std::cerr << "[VulkanContext] picked GPU: " << props.deviceName << "\n";

        if (rayTracingEnabled_) {
            rtPipelineProperties_.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &rtPipelineProperties_;
            vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

            // Probe for VK_NV_ray_tracing_invocation_reorder (SER). Extension
            // presence alone isn't enough — also need the feature's
            // rayTracingInvocationReorder bit set on this device. Both are
            // queried here so the rest of the renderer can branch on a single
            // boolean. Non-NVIDIA GPUs and pre-Ampere NVIDIA fall through to
            // the fallback raygen SPV.
            const auto pickedExts = deviceExtensions(physicalDevice_);
            if (hasExtension(pickedExts, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)) {
                VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeat{};
                reorderFeat.sType =
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
                VkPhysicalDeviceFeatures2 feat2{};
                feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                feat2.pNext = &reorderFeat;
                vkGetPhysicalDeviceFeatures2(physicalDevice_, &feat2);
                rayTracingInvocationReorderSupported_ =
                        reorderFeat.rayTracingInvocationReorder == VK_TRUE;
            }
            std::cerr << "[VulkanContext] SER (VK_NV_ray_tracing_invocation_reorder): "
                      << (rayTracingInvocationReorderSupported_ ? "enabled" : "fallback")
                      << "\n";
        }

        // Find queue families.
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qn, qprops.data());
        for (uint32_t i = 0; i < qn; ++i) {
            const auto& q = qprops[i];
            if ((q.queueFlags & VK_QUEUE_GRAPHICS_BIT) && queueFamilies_.graphics == UINT32_MAX) {
                queueFamilies_.graphics = i;
            }
            if ((q.queueFlags & VK_QUEUE_COMPUTE_BIT) && queueFamilies_.compute == UINT32_MAX) {
                queueFamilies_.compute = i;
            }
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, surface_, &presentSupport);
            if (presentSupport && queueFamilies_.present == UINT32_MAX) {
                queueFamilies_.present = i;
            }
        }
        if (queueFamilies_.graphics == UINT32_MAX || queueFamilies_.present == UINT32_MAX) {
            throw std::runtime_error("[VulkanContext] required queue families not present on GPU");
        }
    }

    void VulkanContext::createLogicalDevice() {
        const float prio = 1.0f;
        std::set<uint32_t> uniqueFams{queueFamilies_.graphics,
                                     queueFamilies_.present,
                                     queueFamilies_.compute};
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t fam : uniqueFams) {
            if (fam == UINT32_MAX) continue;
            VkDeviceQueueCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            ci.queueFamilyIndex = fam;
            ci.queueCount = 1;
            ci.pQueuePriorities = &prio;
            qcis.push_back(ci);
        }

        std::vector<const char*> extensions(kBaseDeviceExtensions.begin(), kBaseDeviceExtensions.end());
        if (rayTracingEnabled_) {
            extensions.insert(extensions.end(), kRayTracingExtensions.begin(), kRayTracingExtensions.end());
            if (rayTracingInvocationReorderSupported_) {
                extensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
            }
        }

        // Required core 1.2 / 1.3 features (BDA, dynamic rendering, sync2).
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.descriptorIndexing = VK_TRUE;
        f12.runtimeDescriptorArray = VK_TRUE;
        f12.scalarBlockLayout = VK_TRUE;// closest-hit reads GeometryDesc[] / normals via scalar layout
        f12.pNext = &f13;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR fAS{};
        fAS.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        fAS.accelerationStructure = VK_TRUE;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR fRT{};
        fRT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        fRT.rayTracingPipeline = VK_TRUE;

        // SER feature struct — only chained when both extension and feature
        // probed positive at pickPhysicalDevice. Driver rejects vkCreateDevice
        // if we chain it without the extension being in the enabled-extension
        // list above (the conditional `extensions.push_back(...)` handles that).
        VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV fReorder{};
        fReorder.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
        fReorder.rayTracingInvocationReorder = VK_TRUE;

        if (rayTracingEnabled_) {
            fRT.pNext = &f12;
            fAS.pNext = &fRT;
            if (rayTracingInvocationReorderSupported_) {
                // Splice fReorder at the head of the chain so it precedes fAS.
                fReorder.pNext = &fAS;
            }
        }

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.features.shaderInt64 = VK_TRUE;// uint64_t buffer-reference addresses in rchit
        features2.features.samplerAnisotropy = VK_TRUE;// material-tex sampler enables aniso filtering
        // Hybrid gbuf indirect-drawing path encodes the per-draw DrawInfo
        // index into VkDrawIndirectCommand::firstInstance; the VS reads
        // it back as gl_InstanceIndex. Without this feature, firstInstance
        // must be 0 and the trick collapses.
        features2.features.drawIndirectFirstInstance = VK_TRUE;
        // gl_DrawIDARB / gl_BaseInstanceARB / gl_BaseVertexARB — not strictly
        // required by the current gbuf shader (it uses gl_InstanceIndex), but
        // pulling the feature in keeps the door open for future per-draw
        // pulls and is universally supported on RT-capable hardware.
        features2.features.multiDrawIndirect = VK_TRUE;
        // VK_POLYGON_MODE_LINE for the wireframe overlay pipeline.
        features2.features.fillModeNonSolid = VK_TRUE;
        // Storage-image writes without a format qualifier — lets shaders write
        // to BGRA8 swap targets without declaring rgba8 (which would mismatch
        // the underlying VkImageView format and produce a validation warning).
        features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        if (rayTracingEnabled_) {
            features2.pNext = rayTracingInvocationReorderSupported_
                                      ? static_cast<void*>(&fReorder)
                                      : static_cast<void*>(&fAS);
        } else {
            features2.pNext = &f12;
        }

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &features2;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();

        check(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_), "vkCreateDevice");

        vkGetDeviceQueue(device_, queueFamilies_.graphics, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueFamilies_.present, 0, &presentQueue_);
        if (queueFamilies_.compute != UINT32_MAX) {
            vkGetDeviceQueue(device_, queueFamilies_.compute, 0, &computeQueue_);
        } else {
            computeQueue_ = graphicsQueue_;
        }

        // EXT_debug_utils object-name function — loaded once for the whole
        // app, used by setObjectName helpers. Null when the extension isn't
        // enabled (validation off); the helpers detect that and no-op.
        setObjectNameFn_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));

        if (rayTracingEnabled_) {
            auto load = [this](const char* name, void** dst) {
                *dst = reinterpret_cast<void*>(vkGetDeviceProcAddr(device_, name));
                if (!*dst) {
                    throw std::runtime_error(std::string("[VulkanContext] vkGetDeviceProcAddr(") +
                                             name + ") returned null");
                }
            };
            load("vkCreateAccelerationStructureKHR",        reinterpret_cast<void**>(&rt_.createAccelerationStructure));
            load("vkDestroyAccelerationStructureKHR",       reinterpret_cast<void**>(&rt_.destroyAccelerationStructure));
            load("vkGetAccelerationStructureBuildSizesKHR", reinterpret_cast<void**>(&rt_.getAccelerationStructureBuildSizes));
            load("vkCmdBuildAccelerationStructuresKHR",     reinterpret_cast<void**>(&rt_.cmdBuildAccelerationStructures));
            load("vkGetAccelerationStructureDeviceAddressKHR",
                 reinterpret_cast<void**>(&rt_.getAccelerationStructureDeviceAddress));
            load("vkCreateRayTracingPipelinesKHR",       reinterpret_cast<void**>(&rt_.createRayTracingPipelines));
            load("vkGetRayTracingShaderGroupHandlesKHR", reinterpret_cast<void**>(&rt_.getRayTracingShaderGroupHandles));
            load("vkCmdTraceRaysKHR",                    reinterpret_cast<void**>(&rt_.cmdTraceRays));
        }
    }

    namespace {
        // Shared body of the three setObjectName overloads — same call shape,
        // only objectType + handle differ. Validation-off path early-outs at
        // the null function-pointer check.
        void setObjectNameImpl(PFN_vkSetDebugUtilsObjectNameEXT fn,
                               VkDevice device, VkObjectType type,
                               uint64_t handle, const char* name) {
            if (!fn || handle == 0 || !name) return;
            VkDebugUtilsObjectNameInfoEXT info{};
            info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            info.objectType   = type;
            info.objectHandle = handle;
            info.pObjectName  = name;
            fn(device, &info);
        }
    }

    void VulkanContext::setObjectName(VkImage image, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_IMAGE,
                          reinterpret_cast<uint64_t>(image), name);
    }

    void VulkanContext::setObjectName(VkImageView view, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_IMAGE_VIEW,
                          reinterpret_cast<uint64_t>(view), name);
    }

    void VulkanContext::setObjectName(VkBuffer buffer, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_BUFFER,
                          reinterpret_cast<uint64_t>(buffer), name);
    }

    void VulkanContext::createAllocator() {
        VmaAllocatorCreateInfo ci{};
        ci.physicalDevice = physicalDevice_;
        ci.device = device_;
        ci.instance = instance_;
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        if (rayTracingEnabled_) {
            ci.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }
        check(vmaCreateAllocator(&ci, &allocator_), "vmaCreateAllocator");
    }

    void VulkanContext::createSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &caps);

        uint32_t fmtN = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &fmtN, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(fmtN);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &fmtN, formats.data());

        VkSurfaceFormatKHR chosenFmt = formats[0];
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosenFmt = f;
                break;
            }
        }

        uint32_t pmN = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmN, nullptr);
        std::vector<VkPresentModeKHR> presentModes(pmN);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmN, presentModes.data());
        // Honor the canvas vsync flag. vsync (default) -> FIFO: present blocks on
        // the display refresh, capping the render loop to the monitor rate instead
        // of spinning the GPU at 100% (matches the WGPU backend, and avoids starving
        // co-resident compute such as on-device inference). vsync off -> prefer
        // MAILBOX (uncapped, no tearing) then IMMEDIATE, for lowest latency / fastest
        // progressive path-tracer convergence.
        VkPresentModeKHR chosenMode = VK_PRESENT_MODE_FIFO_KHR;// guaranteed; vsync-capped
        if (!vsync_) {
            bool hasMailbox = false, hasImmediate = false;
            for (auto m : presentModes) {
                if (m == VK_PRESENT_MODE_MAILBOX_KHR) hasMailbox = true;
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) hasImmediate = true;
            }
            if (hasMailbox) chosenMode = VK_PRESENT_MODE_MAILBOX_KHR;
            else if (hasImmediate) chosenMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        std::cout << "[VulkanContext] present mode: "
                  << (chosenMode == VK_PRESENT_MODE_FIFO_KHR      ? "FIFO (vsync)"
                      : chosenMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX"
                                                                  : "IMMEDIATE")
                  << "\n";

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX) {
            int w, h;
            glfwGetFramebufferSize(window_, &w, &h);
            extent.width = std::clamp<uint32_t>(static_cast<uint32_t>(w),
                                                 caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp<uint32_t>(static_cast<uint32_t>(h),
                                                  caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
            imageCount = caps.maxImageCount;
        }

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imageCount;
        ci.imageFormat = chosenFmt.format;
        ci.imageColorSpace = chosenFmt.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = chosenMode;
        ci.clipped = VK_TRUE;

        const uint32_t fams[] = {queueFamilies_.graphics, queueFamilies_.present};
        if (queueFamilies_.graphics != queueFamilies_.present) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = fams;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        swapchainFormat_ = chosenFmt.format;
        swapchainExtent_ = extent;

        uint32_t imN = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &imN, nullptr);
        swapchainImages_.resize(imN);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imN, swapchainImages_.data());
    }

    void VulkanContext::createSwapchainImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ci.subresourceRange.baseMipLevel = 0;
            ci.subresourceRange.levelCount = 1;
            ci.subresourceRange.baseArrayLayer = 0;
            ci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]), "vkCreateImageView");
        }
    }

    void VulkanContext::destroySwapchainResources() {
        for (auto v : swapchainImageViews_) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device_, v, nullptr);
        }
        swapchainImageViews_.clear();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchainImages_.clear();
    }

    void VulkanContext::recreateSwapchain() {
        // Spin while minimised.
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        destroySwapchainResources();
        createSwapchain();
        createSwapchainImageViews();
    }

}// namespace threepp::vulkan
