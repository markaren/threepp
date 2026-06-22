// The rendering layer: Canvas (window / headless surface), GLRenderer (draws a
// scene and reads pixels back as numpy), and OrbitControls.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/constants.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#include <cctype>
#include <cstring>
#include <string>

using namespace threepp;

namespace threepp_py {

    // Map a friendly key name ('W', 'a', 'SPACE', 'UP', ...) to threepp's Key enum.
    // Letters/digits exploit the enum's contiguous A-Z and NUM_0-NUM_9 ranges.
    static Key keyFromName(std::string n) {
        for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (n.size() == 1 && n[0] >= 'A' && n[0] <= 'Z')
            return static_cast<Key>(static_cast<int>(Key::A) + (n[0] - 'A'));
        if (n.size() == 1 && n[0] >= '0' && n[0] <= '9')
            return static_cast<Key>(static_cast<int>(Key::NUM_0) + (n[0] - '0'));
        if (n == "SPACE") return Key::SPACE;
        if (n == "UP") return Key::UP;
        if (n == "DOWN") return Key::DOWN;
        if (n == "LEFT") return Key::LEFT;
        if (n == "RIGHT") return Key::RIGHT;
        if (n == "ESCAPE" || n == "ESC") return Key::ESCAPE;
        if (n == "ENTER") return Key::ENTER;
        if (n == "TAB") return Key::TAB;
        if (n == "SHIFT") return Key::LEFT_SHIFT;
        if (n == "CTRL" || n == "CONTROL") return Key::LEFT_CONTROL;
        return Key::UNKNOWN;
    }

    void init_render(py::module_& m) {

        // Tone-mapping operator (maps HDR/linear radiance to displayable range).
        // Default is NoToneMapping (clips >1); ACESFilmic/Neutral give a filmic
        // roll-off that keeps HDR highlights and IBL reflections from blowing out.
        py::enum_<ToneMapping>(m, "ToneMapping")
                .value("NoToneMapping", ToneMapping::None)
                .value("Linear", ToneMapping::Linear)
                .value("Reinhard", ToneMapping::Reinhard)
                .value("Cineon", ToneMapping::Cineon)
                .value("ACESFilmic", ToneMapping::ACESFilmic)
                .value("Neutral", ToneMapping::Neutral);

        // ---- Canvas ----------------------------------------------------------
        // A GLFW window (or a hidden surface when headless=True). Construction is
        // exposed as keyword arguments rather than the fluent Parameters builder.
        py::class_<Canvas>(m, "Canvas")
                .def(py::init([](const std::string& title, int width, int height, int antialiasing,
                                 bool vsync, bool resizable, bool headless) {
                    Canvas::Parameters p;
                    p.title(title);
                    if (width > 0 && height > 0) p.size(width, height);
                    p.antialiasing(antialiasing).vsync(vsync).resizable(resizable).headless(headless);
                    return std::make_unique<Canvas>(p);
                }),
                     py::arg("title") = "threepp", py::arg("width") = -1, py::arg("height") = -1,
                     py::arg("antialiasing") = 4, py::arg("vsync") = true,
                     py::arg("resizable") = true, py::arg("headless") = false)
                .def("animate", &Canvas::animate, py::arg("callback"),
                     "Run the render loop, calling callback() every frame until the window closes.")
                .def("animate_once", &Canvas::animateOnce, py::arg("callback"),
                     "Render a single frame; returns False when the app should quit.")
                .def("size", [](const Canvas& c) {
                    auto s = c.size();
                    return py::make_tuple(s.width(), s.height());
                })
                .def("aspect", &Canvas::aspect)
                .def_property_readonly("graphics_api", [](const Canvas& c) {
                    switch (c.graphicsApi()) {
                        case GraphicsAPI::OpenGL: return "OpenGL";
                        case GraphicsAPI::Vulkan: return "Vulkan";
                        case GraphicsAPI::WebGPU: return "WebGPU";
                        default: return "Cross";
                    }
                })
                .def("set_size", [](Canvas& c, int w, int h) { c.setSize({w, h}); }, py::arg("width"), py::arg("height"))
                .def("on_window_resize", [](Canvas& c, const std::function<void(int, int)>& cb) {
                    c.onWindowResize([cb](WindowSize s) { cb(s.width(), s.height()); });
                }, py::arg("callback"),
                   "Register callback(width, height), called when the window is resized. Use it to "
                   "update the camera aspect (+ update_projection_matrix) and the renderer size.")
                .def("is_key_down", [](const Canvas& c, const std::string& key) { return c.isKeyDown(keyFromName(key)); },
                     py::arg("key"),
                     "Poll whether a key is currently held — e.g. 'W','A','S','D','SPACE','UP','LEFT'. "
                     "Query per-frame for continuous controls (WASD driving); never sticks.")
                .def("is_open", &Canvas::isOpen)
                .def("close", &Canvas::close);

        // ---- GLRenderer ------------------------------------------------------
        py::class_<GLRenderer>(m, "GLRenderer")
                .def(py::init([](Canvas& canvas) { return std::make_unique<GLRenderer>(canvas); }),
                     py::arg("canvas"), py::keep_alive<1, 2>())
                .def("render", &GLRenderer::render, py::arg("scene"), py::arg("camera"))
                .def("set_size", [](GLRenderer& r, int w, int h) { r.setSize({w, h}); }, py::arg("width"), py::arg("height"))
                .def("set_pixel_ratio", &GLRenderer::setPixelRatio, py::arg("value"))
                .def("set_clear_color", &GLRenderer::setClearColor, py::arg("color"), py::arg("alpha") = 1.f)
                .def("clear", &GLRenderer::clear, py::arg("color") = true, py::arg("depth") = true, py::arg("stencil") = true)
                .def_readwrite("auto_clear", &GLRenderer::autoClear)
                .def_readwrite("sort_objects", &GLRenderer::sortObjects)
                .def_readwrite("check_shader_errors", &GLRenderer::checkShaderErrors)
                .def_readwrite("tone_mapping", &GLRenderer::toneMapping)
                .def_readwrite("tone_mapping_exposure", &GLRenderer::toneMappingExposure)
                .def_property("shadow_map_enabled",
                              [](GLRenderer& r) { return r.shadowMap().enabled; },
                              [](GLRenderer& r, bool v) { r.shadowMap().enabled = v; })
                .def("size", [](const GLRenderer& r) {
                    auto s = r.size();
                    return py::make_tuple(s.width(), s.height());
                })
                // Read the current framebuffer as a (H, W, 3) uint8 numpy array.
                // OpenGL reads bottom-up; flip=True (default) returns the usual
                // top-down image orientation.
                .def("read_pixels", [](GLRenderer& r, bool flip) {
                    auto s = r.size();
                    const int w = s.width(), h = s.height();
                    std::vector<unsigned char> buf = r.readRGBPixels();
                    py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h),
                                              static_cast<py::ssize_t>(w),
                                              static_cast<py::ssize_t>(3)});
                    const size_t rowBytes = static_cast<size_t>(w) * 3;
                    auto* dst = arr.mutable_data();
                    if (buf.size() >= rowBytes * static_cast<size_t>(h)) {
                        for (int row = 0; row < h; ++row) {
                            const auto srcRow = static_cast<size_t>(flip ? (h - 1 - row) : row);
                            std::memcpy(dst + static_cast<size_t>(row) * rowBytes,
                                        buf.data() + srcRow * rowBytes, rowBytes);
                        }
                    }
                    return arr;
                }, py::arg("flip") = true)
                .def("save_frame", [](GLRenderer& r, const std::string& path) { r.writeFramebuffer(path); }, py::arg("path"));

        // ---- OrbitControls ---------------------------------------------------
        py::class_<OrbitControls>(m, "OrbitControls")
                .def(py::init([](Camera& camera, Canvas& canvas) {
                    return std::make_unique<OrbitControls>(camera, canvas);
                }),
                     py::arg("camera"), py::arg("canvas"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
                .def_readwrite("enabled", &OrbitControls::enabled)
                .def_readwrite("target", &OrbitControls::target)
                .def_readwrite("min_distance", &OrbitControls::minDistance)
                .def_readwrite("max_distance", &OrbitControls::maxDistance)
                .def_readwrite("enable_damping", &OrbitControls::enableDamping)
                .def_readwrite("damping_factor", &OrbitControls::dampingFactor)
                .def_readwrite("enable_zoom", &OrbitControls::enableZoom)
                .def_readwrite("zoom_speed", &OrbitControls::zoomSpeed)
                .def_readwrite("enable_rotate", &OrbitControls::enableRotate)
                .def_readwrite("rotate_speed", &OrbitControls::rotateSpeed)
                .def_readwrite("enable_pan", &OrbitControls::enablePan)
                .def_readwrite("auto_rotate", &OrbitControls::autoRotate)
                .def_readwrite("auto_rotate_speed", &OrbitControls::autoRotateSpeed)
                .def("update", &OrbitControls::update);
    }

}// namespace threepp_py
