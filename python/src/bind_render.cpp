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
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/helpers/LidarModel.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cctype>
#include <cstring>
#include <sstream>
#include <string>

using namespace threepp;

namespace threepp_py {

    // Resolve a Python renderer object to the backend-neutral threepp::Renderer&.
    // Accepts the GLRenderer binding or the Vulkan facade (whose underlying native
    // renderer is recovered via py_vulkan_native_renderer); raises a clear TypeError
    // for anything else. Lets renderer-taking helpers (DepthSensor) be backend-neutral.
    static Renderer& as_renderer(const py::handle& h) {
        if (Renderer* vk = py_vulkan_native_renderer(h)) return *vk;
        return h.cast<GLRenderer&>();
    }

    // Map a friendly key name ('W', 'a', 'SPACE', 'UP', ...) to threepp's Key enum.
    // Letters/digits exploit the enum's contiguous A-Z and NUM_0-NUM_9 ranges.
    static Key keyFromName(std::string n) {
        for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (n.size() == 1 && n[0] >= 'A' && n[0] <= 'Z')
            return static_cast<Key>(static_cast<int>(Key::A) + (n[0] - 'A'));
        if (n.size() == 1 && n[0] >= '0' && n[0] <= '9')
            return static_cast<Key>(static_cast<int>(Key::NUM_0) + (n[0] - '0'));
        // Numpad keys: "KP8" / "NUM8" / "NUMPAD8" -> Key::KP_8 (distinct from the
        // top-row digit "8" -> Key::NUM_8).
        for (const std::string& pre : {std::string("KP"), std::string("NUMPAD"), std::string("NUM")}) {
            if (n.size() == pre.size() + 1 && n.compare(0, pre.size(), pre) == 0 &&
                n.back() >= '0' && n.back() <= '9')
                return static_cast<Key>(static_cast<int>(Key::KP_0) + (n.back() - '0'));
        }
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

        // ---- LIDAR value types (helpers/LidarTypes.hpp + LidarModel.hpp) -----
        // Pure data structs shared by the GL LidarSensor and the Vulkan
        // PathTracedLidarSensor. Bound here (always-compiled, GL-safe) as plain
        // value types; the scan itself is a separate, renderer-side step.
        py::class_<LidarBeam>(m, "LidarBeam")
                .def(py::init([](const Vector3& origin, const Vector3& direction) {
                    return LidarBeam{origin, direction};
                }), py::arg("origin") = Vector3(), py::arg("direction") = Vector3())
                .def_readwrite("origin", &LidarBeam::origin)
                .def_readwrite("direction", &LidarBeam::direction);

        py::class_<LidarReturn>(m, "LidarReturn")
                .def(py::init<>())
                .def_readwrite("position", &LidarReturn::position)
                .def_readwrite("normal", &LidarReturn::normal)
                .def_readwrite("distance", &LidarReturn::distance)
                .def_readwrite("intensity", &LidarReturn::intensity)
                .def_readwrite("hit_instance_id", &LidarReturn::hitInstanceId)
                .def_readwrite("return_no", &LidarReturn::returnNo)
                .def("__repr__", [](const LidarReturn& r) {
                    std::ostringstream o;
                    o << "LidarReturn(distance=" << r.distance << ", intensity=" << r.intensity
                      << ", hit_instance_id=" << r.hitInstanceId << ", return_no=" << r.returnNo << ")";
                    return o.str();
                });

        py::class_<LidarParams>(m, "LidarParams")
                .def(py::init<>())
                .def_readwrite("max_range", &LidarParams::maxRange)
                .def_readwrite("laser_power", &LidarParams::laserPower)
                .def_readwrite("reference_range", &LidarParams::referenceRange)
                .def_readwrite("atmospheric_extinction", &LidarParams::atmosphericExtinction)
                .def_readwrite("detector_threshold", &LidarParams::detectorThreshold)
                .def_readwrite("max_returns", &LidarParams::maxReturns)
                .def_readwrite("samples_per_beam", &LidarParams::samplesPerBeam)
                .def_readwrite("beam_divergence_mrad", &LidarParams::beamDivergenceMrad)
                .def_readwrite("medium_surface_y", &LidarParams::mediumSurfaceY)
                .def_readwrite("medium_extinction", &LidarParams::mediumExtinction)
                .def_readwrite("medium_albedo", &LidarParams::mediumAlbedo)
                .def_readwrite("medium_anisotropy", &LidarParams::mediumAnisotropy);

        py::class_<LidarModel>(m, "LidarModel")
                .def(py::init<>())
                .def_readwrite("elevation_angles", &LidarModel::elevationAngles)
                .def_readwrite("azimuth_resolution", &LidarModel::azimuthResolution)
                .def_readwrite("azimuth_min", &LidarModel::azimuthMin)
                .def_readwrite("azimuth_max", &LidarModel::azimuthMax)
                .def_static("vlp16", &LidarModel::VLP16, "Velodyne VLP-16: 16 beams, +/-15deg elevation.")
                .def_static("hdl32e", &LidarModel::HDL32E, "Velodyne HDL-32E: 32 beams, -30.67..+10.67deg.")
                .def_static("os1_64", &LidarModel::OS1_64, "Ouster OS1-64: 64 beams, +/-22.5deg elevation.")
                .def_static("os0_128", &LidarModel::OS0_128, "Ouster OS0-128: 128 beams, +/-45deg elevation.")
                .def("__repr__", [](const LidarModel& lm) {
                    std::ostringstream o;
                    o << "LidarModel(beams=" << lm.elevationAngles.size()
                      << ", azimuth_resolution=" << lm.azimuthResolution << ")";
                    return o.str();
                });

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
                .def("set_viewport", [](GLRenderer& r, int x, int y, int width, int height) { r.setViewport(x, y, width, height); },
                     py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
                .def("set_scissor", [](GLRenderer& r, int x, int y, int width, int height) { r.setScissor(x, y, width, height); },
                     py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
                .def("set_scissor_test", &GLRenderer::setScissorTest, py::arg("enabled"))
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

        // ---- DepthSensor (helpers/DepthSensor.hpp) ---------------------------
        // A GPU depth-render sensor: renders the scene from its own viewpoint, linearizes
        // depth, reads it back, and reprojects to a world-space point cloud (optionally with
        // per-point sRGB color). It is an Object3D — aim it with position/rotation/quaternion/
        // look_at, then scan() with a GLRenderer. scan() refreshes the sensor's world matrix
        // first, so it works whether or not the sensor was added to the scene.
        auto pts_to_numpy = [](const std::vector<Vector3>& cloud) {
            py::array_t<float> pts({static_cast<py::ssize_t>(cloud.size()), static_cast<py::ssize_t>(3)});
            auto* d = pts.mutable_data();
            for (size_t i = 0; i < cloud.size(); ++i) {
                d[i * 3 + 0] = cloud[i].x; d[i * 3 + 1] = cloud[i].y; d[i * 3 + 2] = cloud[i].z;
            }
            return pts;
        };
        py::class_<DepthSensor, Object3D, std::shared_ptr<DepthSensor>>(m, "DepthSensor")
                .def(py::init([](float fov_y, unsigned int width, unsigned int height, float near, float far) {
                    return std::make_shared<DepthSensor>(fov_y, width, height, near, far);
                }), py::arg("fov_y"), py::arg("width"), py::arg("height"),
                    py::arg("near") = 0.1f, py::arg("far") = 100.f,
                    "Depth sensor with a vertical FOV (deg), output resolution, and near/far clip (m).")
                .def_readwrite("range_noise", &DepthSensor::rangeNoise,
                               "Gaussian range-noise std-dev in metres (0 = perfect sensor).")
                .def_property_readonly("width", &DepthSensor::width)
                .def_property_readonly("height", &DepthSensor::height)
                .def_property_readonly("fov", &DepthSensor::fov)
                .def_property_readonly("near", &DepthSensor::near)
                .def_property_readonly("far", &DepthSensor::far)
                .def("scan", [pts_to_numpy](DepthSensor& self, const py::object& renderer, Scene& scene) {
                    self.updateWorldMatrix(true, true);            // sync sensor + child camera pose
                    std::vector<Vector3> cloud;
                    self.scan(as_renderer(renderer), scene, cloud);
                    return pts_to_numpy(cloud);
                }, py::arg("renderer"), py::arg("scene"),
                   "Depth scan -> (N,3) float32 world-space hit points (N = points that hit within far). "
                   "Works with a GLRenderer (raster depth) or a VulkanRenderer (path-traced through the "
                   "renderer's acceleration structure -- render() the scene at least once first).")
                .def("scan_rgbd", [pts_to_numpy](DepthSensor& self, const py::object& renderer, Scene& scene) {
                    self.updateWorldMatrix(true, true);
                    std::vector<Vector3> cloud;
                    std::vector<Color> colors;
                    self.scan(as_renderer(renderer), scene, cloud, colors);
                    py::array_t<float> col({static_cast<py::ssize_t>(colors.size()), static_cast<py::ssize_t>(3)});
                    auto* c = col.mutable_data();
                    for (size_t i = 0; i < colors.size(); ++i) {
                        c[i * 3 + 0] = colors[i].r; c[i * 3 + 1] = colors[i].g; c[i * 3 + 2] = colors[i].b;
                    }
                    return py::make_tuple(pts_to_numpy(cloud), col);
                }, py::arg("renderer"), py::arg("scene"),
                   "RGB-D scan -> (points (N,3) float32 world-space, colors (N,3) float32 in [0,1]). On GL the "
                   "colors are sampled sRGB; on Vulkan they are LIDAR intensity as greyscale.");

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
