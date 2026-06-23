// Dear ImGui immediate-mode UI for the Python bindings.
//
// Built on threepp's own ImguiContext, which handles both the OpenGL backend
// and the Vulkan backend (it registers an overlay callback on the VulkanRenderer
// to record ImGui draw data into the frame after the scene). So the same panel
// works on a GLRenderer or a VulkanRenderer window.
//
//   ui = tp.ImguiContext(canvas, renderer)
//   def draw():
//       tp.imgui.begin("Controls"); ...; tp.imgui.end()
//   def animate():
//       controls.update(); renderer.render(scene, camera); ui.render(draw)
//   canvas.animate(animate)
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_IMGUI

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "threepp/canvas/Canvas.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace threepp;

namespace {

    // threepp's ImguiContext drives a virtual onRender(); here it forwards to a
    // transient Python callback set just before each frame, so the UI code can
    // change every call (unlike ImguiFunctionalContext, which fixes it at ctor).
    class PyImgui : public ImguiContext {
    public:
        explicit PyImgui(const Canvas& canvas) : ImguiContext(canvas) { setup(); }
        PyImgui(const Canvas& canvas, Renderer& renderer) : ImguiContext(canvas, renderer) { setup(); }

        void render_with(const std::function<void()>& draw) {
            current_ = &draw;
            ImguiContext::render();
            current_ = nullptr;
        }

        [[nodiscard]] bool want_capture_mouse() const { return ImGui::GetIO().WantCaptureMouse; }
        [[nodiscard]] bool want_capture_keyboard() const { return ImGui::GetIO().WantCaptureKeyboard; }

    protected:
        void onRender() override {
            if (current_ && *current_) (*current_)();
        }

    private:
        void setup() {
            ImGui::StyleColorsDark();
            ImGui::GetIO().IniFilename = nullptr;// don't litter an imgui.ini next to scripts
            // contentScale() can report 0 for a headless/hidden window, which
            // builds a zero-size font atlas — fine for the legacy GL path but it
            // crashes imgui's dynamic-texture font path (the Vulkan backend).
            // Clamp to a sane minimum.
            if (dpiScale() <= 0.f) setFontScale(1.0f);
        }
        const std::function<void()>* current_ = nullptr;
    };

}// namespace

namespace threepp_py {

    void init_imgui(py::module_& m) {
        m.attr("HAS_IMGUI") = true;

        py::class_<PyImgui>(m, "ImguiContext")
                .def(py::init([](Canvas& canvas, const py::object& renderer) -> std::unique_ptr<PyImgui> {
                    // The ImGui backend follows the renderer/canvas: GLRenderer → GL,
                    // VulkanRenderer → Vulkan overlay (threepp's ImguiContext records
                    // ImGui into the deferred frame after the scene).
                    if (auto* vk = renderer.is_none() ? nullptr : py_vulkan_native_renderer(renderer)) {
                        return std::make_unique<PyImgui>(canvas, *vk);
                    }
                    if (canvas.graphicsApi() == GraphicsAPI::Vulkan) {
                        throw std::invalid_argument(
                                "ImguiContext: a Vulkan Canvas needs its VulkanRenderer passed as the "
                                "`renderer` argument.");
                    }
                    if (renderer.is_none()) return std::make_unique<PyImgui>(canvas);// GL backend
                    if (py::isinstance<GLRenderer>(renderer)) {
                        return std::make_unique<PyImgui>(canvas, *renderer.cast<GLRenderer*>());
                    }
                    throw std::invalid_argument(
                            "ImguiContext: renderer must be a GLRenderer or VulkanRenderer (or omitted "
                            "for the GL backend)");
                }),
                     py::arg("canvas"), py::arg("renderer") = py::none(),
                     py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
                     "Dear ImGui UI. Pass the renderer (GLRenderer or VulkanRenderer; omit for GL). "
                     "Create it AFTER the renderer. Works on both backends.")
                .def("render", &PyImgui::render_with, py::arg("draw"),
                     "Build + draw one UI frame; call inside animate() after renderer.render().")
                .def_property_readonly("want_capture_mouse", &PyImgui::want_capture_mouse,
                                       "True when the pointer is over UI — gate OrbitControls on `not ui.want_capture_mouse`.")
                .def_property_readonly("want_capture_keyboard", &PyImgui::want_capture_keyboard);

        // ---- curated immediate-mode widget API -------------------------------
        auto im = m.def_submodule("imgui", "Dear ImGui immediate-mode widgets (call inside ImguiContext.render's draw callback).");

        im.def("begin", [](const std::string& name) { return ImGui::Begin(name.c_str()); }, py::arg("name"),
               "Start a window; returns False if collapsed. Pair with end().");
        im.def("end", [] { ImGui::End(); });
        im.def("set_next_window_pos", [](float x, float y) { ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver); }, py::arg("x"), py::arg("y"));
        im.def("set_next_window_size", [](float w, float h) { ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver); }, py::arg("width"), py::arg("height"));
        im.def("separator", [] { ImGui::Separator(); });
        im.def("spacing", [] { ImGui::Spacing(); });
        im.def("same_line", [] { ImGui::SameLine(); });
        im.def("collapsing_header", [](const std::string& label) { return ImGui::CollapsingHeader(label.c_str()); }, py::arg("label"));
        im.def("tree_node", [](const std::string& label) { return ImGui::TreeNode(label.c_str()); }, py::arg("label"));
        im.def("tree_pop", [] { ImGui::TreePop(); });

        im.def("text", [](const std::string& s) { ImGui::TextUnformatted(s.c_str()); }, py::arg("text"));
        im.def("bullet_text", [](const std::string& s) { ImGui::BulletText("%s", s.c_str()); }, py::arg("text"));

        im.def("button", [](const std::string& label) { return ImGui::Button(label.c_str()); }, py::arg("label"),
               "Returns True on the frame the button is clicked.");
        im.def("checkbox", [](const std::string& label, bool v) {
            bool b = v;
            bool changed = ImGui::Checkbox(label.c_str(), &b);
            return py::make_tuple(changed, b);
        }, py::arg("label"), py::arg("value"));
        im.def("slider_float", [](const std::string& label, float v, float lo, float hi) {
            float f = v;
            bool changed = ImGui::SliderFloat(label.c_str(), &f, lo, hi);
            return py::make_tuple(changed, f);
        }, py::arg("label"), py::arg("value"), py::arg("min"), py::arg("max"));
        im.def("slider_int", [](const std::string& label, int v, int lo, int hi) {
            int i = v;
            bool changed = ImGui::SliderInt(label.c_str(), &i, lo, hi);
            return py::make_tuple(changed, i);
        }, py::arg("label"), py::arg("value"), py::arg("min"), py::arg("max"));
        im.def("drag_float", [](const std::string& label, float v, float speed, float lo, float hi) {
            float f = v;
            bool changed = ImGui::DragFloat(label.c_str(), &f, speed, lo, hi);
            return py::make_tuple(changed, f);
        }, py::arg("label"), py::arg("value"), py::arg("speed") = 1.0f, py::arg("min") = 0.0f, py::arg("max") = 0.0f);
        im.def("input_float", [](const std::string& label, float v) {
            float f = v;
            bool changed = ImGui::InputFloat(label.c_str(), &f);
            return py::make_tuple(changed, f);
        }, py::arg("label"), py::arg("value"));
        im.def("color_edit3", [](const std::string& label, std::array<float, 3> rgb) {
            bool changed = ImGui::ColorEdit3(label.c_str(), rgb.data());
            return py::make_tuple(changed, py::make_tuple(rgb[0], rgb[1], rgb[2]));
        }, py::arg("label"), py::arg("rgb"));
        im.def("combo", [](const std::string& label, int current, const std::vector<std::string>& items) {
            std::vector<const char*> ptrs;
            ptrs.reserve(items.size());
            for (const auto& s : items) ptrs.push_back(s.c_str());
            int idx = current;
            bool changed = ImGui::Combo(label.c_str(), &idx, ptrs.data(), static_cast<int>(ptrs.size()));
            return py::make_tuple(changed, idx);
        }, py::arg("label"), py::arg("current"), py::arg("items"));

        im.def("show_demo_window", [] { ImGui::ShowDemoWindow(); },
               "Show the built-in ImGui demo window (a gallery of every widget).");
        im.def("get_framerate", [] { return ImGui::GetIO().Framerate; });
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_IMGUI not defined

namespace threepp_py {

    void init_imgui(py::module_& m) {
        m.attr("HAS_IMGUI") = false;
    }

}// namespace threepp_py

#endif
