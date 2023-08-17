

#ifndef THREEPP_IMGUI_HELPER_HPP
#define THREEPP_IMGUI_HELPER_HPP

#include "threepp/canvas/Canvas.hpp"

#include "imgui.h"
#include "imgui_impl_opengl3.h"

#include <functional>
#include <utility>

namespace threepp {

    class ImguiContext {

    public:
        explicit ImguiContext(CanvasBase& canvas): canvas(canvas) {
            ImGui::CreateContext();
            canvas.initImguiContext();
            ImGui_ImplOpenGL3_Init("#version 330");
        }

        ImguiContext(ImguiContext&&) = delete;
        ImguiContext(const ImguiContext&) = delete;
        ImguiContext& operator=(const ImguiContext&) = delete;

        void render() {
            ImGui_ImplOpenGL3_NewFrame();
            canvas.newImguiFrame();
            ImGui::NewFrame();

            onRender();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        ~ImguiContext() {
            ImGui_ImplOpenGL3_Shutdown();
            canvas.destroyImguiContext();
            ImGui::DestroyContext();
        }

    protected:
        virtual void onRender() = 0;

    private:
        CanvasBase& canvas;
    };

    class ImguiFunctionalContext: public ImguiContext {

    public:
        explicit ImguiFunctionalContext(CanvasBase& canvas, std::function<void()> f)
            : ImguiContext(canvas),
              f_(std::move(f)) {}


    protected:
        void onRender() override {
            f_();
        }

    private:
        std::function<void()> f_;
    };

}

#endif//THREEPP_IMGUI_HELPER_HPP
