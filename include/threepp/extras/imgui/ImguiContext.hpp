

#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <functional>
#include <utility>

class ImguiContext {

public:
    explicit ImguiContext(void* window) {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*) window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
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

    ~ImguiContext() {
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
