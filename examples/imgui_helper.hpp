

#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include "threepp/Canvas.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class imggui_helper {

public:
    threepp::Canvas& canvas;

    explicit imggui_helper(threepp::Canvas& canvas): canvas(canvas) {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*) canvas.window_ptr(), true);
        ImGui_ImplOpenGL3_Init("#version 330 core");
    }

    void render() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        onRender();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    ~imggui_helper() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

protected:
    virtual void onRender() = 0;

};

#endif//THREEPP_IMGUI_HELPER_HPP
