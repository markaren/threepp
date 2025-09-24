
#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include "threepp/canvas/Canvas.hpp"


#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <functional>
#include <iostream>

#include <threepp/canvas/Canvas.hpp>
#include <threepp/canvas/Monitor.hpp>

class ImguiContext {

public:
    explicit ImguiContext(void* window) {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(static_cast<GLFWwindow*>(window), true);
#if EMSCRIPTEN
        ImGui_ImplOpenGL3_Init("#version 300 es");
#else
        ImGui_ImplOpenGL3_Init("#version 330 core");
#endif

        setFontScale(threepp::monitor::contentScale().first);
    }

    explicit ImguiContext(const threepp::Canvas& canvas): ImguiContext(canvas.windowPtr()) {
        canvas.onMonitorChange([this](int monitor) {
            setFontScale(threepp::monitor::contentScale(monitor).first);
        });
    }

    ImguiContext(ImguiContext&&) = delete;
    ImguiContext(const ImguiContext&) = delete;
    ImguiContext& operator=(const ImguiContext&) = delete;

    void render() {
        if (!dpiAwareIsConfigured_) {

            ImGuiStyle& style = ImGui::GetStyle();
            style = ImGuiStyle();
            style.FontScaleDpi = dpiScale_;
            style.ScaleAllSizes(dpiScale_);

            dpiAwareIsConfigured_ = true;
        }

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
    bool dpiAwareIsConfigured_ = true;
    float dpiScale_ = 1.f;
};

class ImguiFunctionalContext: public ImguiContext {

public:
    explicit ImguiFunctionalContext(void* window, std::function<void()> f)
        : ImguiContext(window),
          f_(std::move(f)) {}

    explicit ImguiFunctionalContext(const threepp::Canvas& canvas, std::function<void()> f)
        : ImguiContext(canvas),
          f_(std::move(f)) {}

protected:
    void onRender() override {
        f_();
    }

private:
    std::function<void()> f_;
};

#endif//THREEPP_IMGUI_HELPER_HPP
