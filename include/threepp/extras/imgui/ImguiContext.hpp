
#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <threepp/canvas/Monitor.hpp>

class ImguiContext {

public:
    explicit ImguiContext(void* window, bool dpiAware = false) {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(window), true);
#if EMSCRIPTEN
        ImGui_ImplOpenGL3_Init("#version 300 es");
#else
        ImGui_ImplOpenGL3_Init("#version 330 core");
#endif

        if (dpiAware) {
            const auto [dpiScaleX, _] = threepp::monitor::contentScale();

            ImGuiIO& io = ImGui::GetIO();
            io.FontGlobalScale = dpiScaleX;// Assuming dpiScaleX = dpiScaleY

            ImGuiStyle& style = ImGui::GetStyle();
            style.ScaleAllSizes(dpiScaleX);
        }
    }

    ImguiContext(ImguiContext&&) = delete;
    ImguiContext(const ImguiContext&) = delete;
    ImguiContext& operator=(const ImguiContext&) = delete;

    void render() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        onRender();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    virtual ~ImguiContext() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

protected:
    virtual void onRender() = 0;
};

class ImguiFunctionalContext: public ImguiContext {

public:
    explicit ImguiFunctionalContext(void* window, std::function<void()> f)
        : ImguiContext(window),
          f_(std::move(f)) {}


protected:
    void onRender() override {
        f_();
    }

private:
    std::function<void()> f_;
};

#endif//THREEPP_IMGUI_HELPER_HPP
