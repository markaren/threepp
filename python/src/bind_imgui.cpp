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
#include <cfloat>
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
                    // A renderer is required on every backend so the same construction
                    // call is portable. The Vulkan backend needs it to attach the ImGui
                    // overlay; we require it on GL too rather than offer a GL-only
                    // shortcut that breaks the moment you switch to a VulkanRenderer.
                    if (renderer.is_none()) {
                        throw std::invalid_argument(
                                "ImguiContext requires a renderer — pass your GLRenderer or "
                                "VulkanRenderer (create the UI after the renderer). The same "
                                "call then runs on both backends.");
                    }
                    if (auto* vk = py_vulkan_native_renderer(renderer)) {
                        return std::make_unique<PyImgui>(canvas, *vk);// Vulkan overlay
                    }
                    if (py::isinstance<GLRenderer>(renderer)) {
                        return std::make_unique<PyImgui>(canvas, *renderer.cast<GLRenderer*>());
                    }
                    throw std::invalid_argument(
                            "ImguiContext: renderer must be a GLRenderer or VulkanRenderer.");
                }),
                     py::arg("canvas"), py::arg("renderer"),
                     py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
                     "Dear ImGui UI. Pass the renderer (GLRenderer or VulkanRenderer); create "
                     "the context AFTER the renderer. The same call works on both backends.")
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

        // ---- plots and free drawing ------------------------------------------
        // A panel that has to show a CURVE -- a propeller's open-water diagram, a
        // convergence history, a spectrum -- had nothing here but text. plot_lines
        // is ImGui's own sparkline; the draw_* calls below are the escape hatch for
        // the cases it cannot serve (several curves on one pair of axes, an
        // operating point marked on top of them), and together they are the whole
        // of what a read-only diagram needs.
        im.def("plot_lines", [](const std::string& label, const std::vector<float>& values,
                                const std::string& overlay, float scale_min, float scale_max,
                                float width, float height) {
            ImGui::PlotLines(label.c_str(), values.empty() ? nullptr : values.data(),
                             static_cast<int>(values.size()), 0,
                             overlay.empty() ? nullptr : overlay.c_str(),
                             scale_min, scale_max, ImVec2(width, height));
        }, py::arg("label"), py::arg("values"), py::arg("overlay") = "",
           py::arg("scale_min") = FLT_MAX, py::arg("scale_max") = FLT_MAX,
           py::arg("width") = 0.0f, py::arg("height") = 0.0f,
           "A sparkline of `values`. scale_min/max default to the data's own range.");

        // Reserve a rectangle in the layout and hand back where it landed, so the
        // draw_* calls below have somewhere to put a diagram that ImGui itself has
        // no widget for. Coordinates are SCREEN space, which is what the draw list
        // takes -- a panel that moves takes its diagram with it.
        im.def("dummy", [](float width, float height) { ImGui::Dummy(ImVec2(width, height)); },
               py::arg("width"), py::arg("height"),
               "Reserve an empty rect in the layout; pair with item_rect().");
        im.def("item_rect", [] {
            const ImVec2 a = ImGui::GetItemRectMin();
            const ImVec2 b = ImGui::GetItemRectMax();
            return py::make_tuple(a.x, a.y, b.x, b.y);
        }, "(x0, y0, x1, y1) of the LAST item, in screen coordinates.");

        // The draw list is the current window's, so everything below is clipped to
        // the panel and scrolls with it.
        im.def("draw_line", [](float x0, float y0, float x1, float y1,
                               std::array<float, 4> rgba, float thickness) {
            ImGui::GetWindowDrawList()->AddLine(ImVec2(x0, y0), ImVec2(x1, y1),
                                                ImGui::GetColorU32(ImVec4(rgba[0], rgba[1], rgba[2], rgba[3])),
                                                thickness);
        }, py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
           py::arg("rgba"), py::arg("thickness") = 1.0f);
        im.def("draw_polyline", [](const std::vector<std::array<float, 2>>& pts,
                                   std::array<float, 4> rgba, float thickness) {
            if (pts.size() < 2) return;
            std::vector<ImVec2> v;
            v.reserve(pts.size());
            for (const auto& p : pts) v.emplace_back(p[0], p[1]);
            ImGui::GetWindowDrawList()->AddPolyline(v.data(), static_cast<int>(v.size()),
                                                    ImGui::GetColorU32(ImVec4(rgba[0], rgba[1], rgba[2], rgba[3])),
                                                    0, thickness);
        }, py::arg("points"), py::arg("rgba"), py::arg("thickness") = 1.0f,
           "One open polyline through screen-space (x, y) points.");
        im.def("draw_rect", [](float x0, float y0, float x1, float y1,
                               std::array<float, 4> rgba, float thickness, bool filled) {
            auto* dl = ImGui::GetWindowDrawList();
            const ImU32 c = ImGui::GetColorU32(ImVec4(rgba[0], rgba[1], rgba[2], rgba[3]));
            if (filled) dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), c);
            else dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), c, 0.0f, 0, thickness);
        }, py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
           py::arg("rgba"), py::arg("thickness") = 1.0f, py::arg("filled") = false);
        im.def("draw_circle", [](float x, float y, float radius,
                                 std::array<float, 4> rgba, float thickness, bool filled) {
            auto* dl = ImGui::GetWindowDrawList();
            const ImU32 c = ImGui::GetColorU32(ImVec4(rgba[0], rgba[1], rgba[2], rgba[3]));
            if (filled) dl->AddCircleFilled(ImVec2(x, y), radius, c, 0);
            else dl->AddCircle(ImVec2(x, y), radius, c, 0, thickness);
        }, py::arg("x"), py::arg("y"), py::arg("radius"),
           py::arg("rgba"), py::arg("thickness") = 1.0f, py::arg("filled") = false);
        im.def("draw_text", [](float x, float y, const std::string& s,
                               std::array<float, 4> rgba) {
            ImGui::GetWindowDrawList()->AddText(ImVec2(x, y),
                                                ImGui::GetColorU32(ImVec4(rgba[0], rgba[1], rgba[2], rgba[3])),
                                                s.c_str());
        }, py::arg("x"), py::arg("y"), py::arg("text"), py::arg("rgba"));

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
