
#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#ifdef THREEPP_WITH_WGPU
#include <imgui_impl_wgpu.h>
#include <threepp/renderers/WgpuRenderer.hpp>
#endif

#ifdef THREEPP_WITH_VULKAN
#include <imgui_impl_vulkan.h>
#include <threepp/renderers/VulkanRenderer.hpp>
#endif

#include <functional>
#include <iostream>

#include <threepp/canvas/Canvas.hpp>
#include <threepp/canvas/Monitor.hpp>
#include <threepp/renderers/Renderer.hpp>

class ImguiContext {

public:
    explicit ImguiContext(void* window, bool useOpenGL = true) {
        ImGui::CreateContext();
        if (useOpenGL) {
            ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(window), true);
#ifdef __EMSCRIPTEN__
            ImGui_ImplOpenGL3_Init("#version 300 es");
#else
            ImGui_ImplOpenGL3_Init("#version 330 core");
#endif
            glInitialized_ = true;
        } else {
            ImGui_ImplGlfw_InitForOther(static_cast<GLFWwindow*>(window), true);
        }

        setFontScale(threepp::monitor::contentScale().first);
    }

    explicit ImguiContext(const threepp::Canvas& canvas)
        : ImguiContext(canvas.windowPtr(), canvas.graphicsApi() != threepp::GraphicsAPI::WebGPU) {
        canvas.onMonitorChange([this](int monitor) {
            setFontScale(threepp::monitor::contentScale(monitor).first);
        });
    }

    ImguiContext(const threepp::Canvas& canvas, threepp::Renderer& renderer)
        : ImguiContext(canvas.windowPtr(), false) {

#ifdef THREEPP_WITH_VULKAN
        if (canvas.graphicsApi() == threepp::GraphicsAPI::Vulkan) {
            vulkanRenderer_ = dynamic_cast<threepp::VulkanRenderer*>(&renderer);
            if (vulkanRenderer_) {
                auto device = static_cast<VkDevice>(vulkanRenderer_->nativeDevice());

                // Small dedicated pool: a single combined-image-sampler is
                // enough for the font atlas; ImGui_ImplVulkan_AddTexture grows
                // it lazily for user textures.
                VkDescriptorPoolSize poolSize{};
                poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                poolSize.descriptorCount = 16;

                VkDescriptorPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                poolInfo.maxSets = 16;
                poolInfo.poolSizeCount = 1;
                poolInfo.pPoolSizes = &poolSize;
                vkCreateDescriptorPool(device, &poolInfo, nullptr, &vulkanDescriptorPool_);

                VkFormat colorFormat = static_cast<VkFormat>(
                        vulkanRenderer_->nativeSwapchainFormat());

                VkPipelineRenderingCreateInfoKHR prCi{};
                prCi.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
                prCi.colorAttachmentCount = 1;
                prCi.pColorAttachmentFormats = &colorFormat;

                ImGui_ImplVulkan_InitInfo initInfo{};
                initInfo.ApiVersion = VK_API_VERSION_1_3;
                initInfo.Instance = static_cast<VkInstance>(vulkanRenderer_->nativeInstance());
                initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(vulkanRenderer_->nativePhysicalDevice());
                initInfo.Device = device;
                initInfo.QueueFamily = vulkanRenderer_->graphicsQueueFamily();
                initInfo.Queue = static_cast<VkQueue>(vulkanRenderer_->nativeGraphicsQueue());
                initInfo.DescriptorPool = vulkanDescriptorPool_;
                initInfo.MinImageCount = 2;
                initInfo.ImageCount = vulkanRenderer_->imageCount();
                initInfo.UseDynamicRendering = true;
                initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
                initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = prCi;

                ImGui_ImplVulkan_Init(&initInfo);

                vulkanRenderer_->setOverlayCallback([this](void* commandBuffer) {
                    if (pendingDrawData_) {
                        ImGui_ImplVulkan_RenderDrawData(
                                pendingDrawData_,
                                static_cast<VkCommandBuffer>(commandBuffer));
                    }
                });

                vulkanInitialized_ = true;
            }
        } else
#endif
        if (canvas.graphicsApi() != threepp::GraphicsAPI::WebGPU) {
            // GL path — reinitialize for OpenGL (undo the InitForOther from delegated ctor)
            ImGui_ImplGlfw_Shutdown();
            ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(canvas.windowPtr()), true);
#ifdef __EMSCRIPTEN__
            ImGui_ImplOpenGL3_Init("#version 300 es");
#else
            ImGui_ImplOpenGL3_Init("#version 330 core");
#endif
            glInitialized_ = true;
        }
#ifdef THREEPP_WITH_WGPU
        else {
            wgpuRenderer_ = dynamic_cast<threepp::WgpuRenderer*>(&renderer);
            if (wgpuRenderer_) {
                ImGui_ImplWGPU_InitInfo initInfo{};
                initInfo.Device = static_cast<WGPUDevice>(wgpuRenderer_->nativeDevice());
                initInfo.RenderTargetFormat = static_cast<WGPUTextureFormat>(wgpuRenderer_->nativeSurfaceFormat());
                initInfo.DepthStencilFormat = WGPUTextureFormat_Depth24Plus;
                initInfo.PipelineMultisampleState.count = 1; // overlay renders to resolved (non-MSAA) surface

                ImGui_ImplWGPU_Init(&initInfo);

                wgpuRenderer_->setOverlayCallback([this](void* passEncoder) {
                    if (pendingDrawData_) {
                        // Override the draw data's display size to match the
                        // current renderer size. During a live window resize,
                        // the draw data may have been generated with a stale
                        // display size (from the previous frame's ui.render()),
                        // causing ImGui's scissor rects to exceed the actual
                        // render pass attachment dimensions.
                        auto sz = wgpuRenderer_->size();
                        pendingDrawData_->DisplaySize = ImVec2(
                            static_cast<float>(sz.width()),
                            static_cast<float>(sz.height()));
                        ImGui_ImplWGPU_RenderDrawData(pendingDrawData_, static_cast<WGPURenderPassEncoder>(passEncoder));
                    }
                });

                wgpuInitialized_ = true;
            }
        }
#endif

        canvas.onMonitorChange([this](int monitor) {
            setFontScale(threepp::monitor::contentScale(monitor).first);
        });
    }

    ImguiContext(ImguiContext&&) = delete;
    ImguiContext(const ImguiContext&) = delete;
    ImguiContext& operator=(const ImguiContext&) = delete;

    void render() {
        if (!glInitialized_ && !wgpuInitialized_ && !vulkanInitialized_) return;

        if (!dpiAwareIsConfigured_) {

            ImGuiStyle& style = ImGui::GetStyle();
            style = ImGuiStyle();
            style.FontScaleDpi = dpiScale_;
            style.ScaleAllSizes(dpiScale_);

            dpiAwareIsConfigured_ = true;
        }

        if (glInitialized_) ImGui_ImplOpenGL3_NewFrame();
#ifdef THREEPP_WITH_WGPU
        if (wgpuInitialized_) ImGui_ImplWGPU_NewFrame();
#endif
#ifdef THREEPP_WITH_VULKAN
        if (vulkanInitialized_) ImGui_ImplVulkan_NewFrame();
#endif
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        onRender();

        ImGui::Render();

        if (glInitialized_) {
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
#ifdef THREEPP_WITH_WGPU
        if (wgpuInitialized_) {
            pendingDrawData_ = ImGui::GetDrawData();
        }
#endif
#ifdef THREEPP_WITH_VULKAN
        if (vulkanInitialized_) {
            pendingDrawData_ = ImGui::GetDrawData();
        }
#endif
    }

    virtual ~ImguiContext() {
        if (glInitialized_) ImGui_ImplOpenGL3_Shutdown();
#ifdef THREEPP_WITH_WGPU
        if (wgpuInitialized_) {
            if (wgpuRenderer_) wgpuRenderer_->setOverlayCallback(nullptr);
            ImGui_ImplWGPU_Shutdown();
        }
#endif
#ifdef THREEPP_WITH_VULKAN
        if (vulkanInitialized_) {
            if (vulkanRenderer_) vulkanRenderer_->setOverlayCallback(nullptr);
            // Drain any pending GPU work before tearing down ImGui's
            // descriptor sets / pipelines.
            vkDeviceWaitIdle(static_cast<VkDevice>(vulkanRenderer_->nativeDevice()));
            ImGui_ImplVulkan_Shutdown();
            if (vulkanDescriptorPool_) {
                vkDestroyDescriptorPool(
                        static_cast<VkDevice>(vulkanRenderer_->nativeDevice()),
                        vulkanDescriptorPool_, nullptr);
            }
        }
#endif
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void setFontScale(float scale) {
        dpiAwareIsConfigured_ = false;
        dpiScale_ = scale;
    }

    void makeDpiAware() {

        std::cerr << "Deprecated function. Use setFontScale instead." << std::endl;
    }

    [[nodiscard]] float dpiScale() const {
        return dpiScale_;
    }

protected:
    virtual void onRender() = 0;

private:
    bool glInitialized_ = false;
    bool wgpuInitialized_ = false;
    bool vulkanInitialized_ = false;
    bool dpiAwareIsConfigured_ = true;
    float dpiScale_ = 1.f;
    ImDrawData* pendingDrawData_ = nullptr;
#ifdef THREEPP_WITH_WGPU
    threepp::WgpuRenderer* wgpuRenderer_ = nullptr;
#endif
#ifdef THREEPP_WITH_VULKAN
    threepp::VulkanRenderer* vulkanRenderer_ = nullptr;
    VkDescriptorPool vulkanDescriptorPool_ = VK_NULL_HANDLE;
#endif
};

class ImguiFunctionalContext: public ImguiContext {

public:
    explicit ImguiFunctionalContext(void* window, std::function<void()> f)
        : ImguiContext(window),
          f_(std::move(f)) {}

    explicit ImguiFunctionalContext(const threepp::Canvas& canvas, std::function<void()> f)
        : ImguiContext(canvas),
          f_(std::move(f)) {}

    ImguiFunctionalContext(const threepp::Canvas& canvas, threepp::Renderer& renderer, std::function<void()> f)
        : ImguiContext(canvas, renderer),
          f_(std::move(f)) {}

protected:
    void onRender() override {
        f_();
    }

private:
    std::function<void()> f_;
};

#endif//THREEPP_IMGUI_HELPER_HPP
