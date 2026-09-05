// The rendering layer: Canvas (window / headless surface), GLRenderer (draws a
// scene and reads pixels back as numpy), OrbitControls, and TransformControls.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/constants.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/input/KeyFromName.hpp"
#include "threepp/helpers/LidarModel.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/helpers/SonarModel.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cctype>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

    // keyFromName lives in the library (threepp/input/KeyFromName.hpp) - one
    // mapping shared with the player's keyboard provider, so the names a script
    // was written against mean the same key on every surface.

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
                .value("Neutral", ToneMapping::Neutral)
                // AgX (Sobotka / three.js AgXToneMapping): the gentlest
                // highlight roll-off of the set. Vulkan renderers only.
                .value("AgX", ToneMapping::AgX);

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
                // 0 = surface, 1 = volume scatter (fog / dust / snow).
                .def_readwrite("return_kind", &LidarReturn::returnKind)
                .def("__repr__", [](const LidarReturn& r) {
                    std::ostringstream o;
                    o << "LidarReturn(distance=" << r.distance << ", intensity=" << r.intensity
                      << ", hit_instance_id=" << r.hitInstanceId << ", return_no=" << r.returnNo << ")";
                    return o.str();
                });

        py::class_<LidarParams>(m, "LidarParams")
                .def(py::init<>())
                .def_readwrite("max_range", &LidarParams::maxRange)
                .def_readwrite("min_range", &LidarParams::minRange)
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
                .def_readwrite("medium_anisotropy", &LidarParams::mediumAnisotropy)
                .def_readwrite("paired_clean_trace", &LidarParams::pairedCleanTrace,
                               "Trace every beam twice in one dispatch -- as-is, and with the "
                               "ParticleField density medium off, same beams and RNG keys. The "
                               "difference IS the degradation. PathTracedLidarSensor.scan() "
                               "returns the clean leg under the 'clean' key.");

        py::class_<LidarModel>(m, "LidarModel")
                .def(py::init<>())
                .def_readwrite("elevation_angles", &LidarModel::elevationAngles)
                .def_readwrite("azimuth_resolution", &LidarModel::azimuthResolution)
                .def_readwrite("azimuth_min", &LidarModel::azimuthMin)
                .def_readwrite("azimuth_max", &LidarModel::azimuthMax)
                .def_static("vlp16", &LidarModel::VLP16, "Velodyne VLP-16: 16 beams, +/-15deg elevation.")
                .def_static("hdl32e", &LidarModel::HDL32E, "Velodyne HDL-32E: 32 beams, -30.67..+10.67deg.")
                .def_static("os1_64", &LidarModel::OS1_64, "Ouster OS1-64: 64 beams, +/-22.5deg elevation, 1024 columns/rev.")
                .def_static("os0_128", &LidarModel::OS0_128, "Ouster OS0-128: 128 beams, +/-45deg elevation, 1024 columns/rev.")
                .def("__repr__", [](const LidarModel& lm) {
                    std::ostringstream o;
                    o << "LidarModel(beams=" << lm.elevationAngles.size()
                      << ", azimuth_resolution=" << lm.azimuthResolution << ")";
                    return o.str();
                });

        // ---- Imaging sonar value types (helpers/SonarModel.hpp) --------------
        // The renderer-free half of SonarSensor: fan geometry, the echogram,
        // and per-target reflectivity. The sensor itself is Vulkan-side.
        py::class_<SonarModel>(m, "SonarModel",
                               "Fan geometry + echo model of an imaging sonar. Angles in the LIDAR frame "
                               "(azimuth 0 = local -Z, positive toward +X, elevation up). Presets carry "
                               "swath / beams / aperture / range from the datasheet; no beam pattern.")
                .def(py::init<>())
                .def_readwrite("horizontal_fov", &SonarModel::horizontalFov, "Full swath, degrees.")
                .def_readwrite("beams", &SonarModel::beams, "Azimuth beams across the swath.")
                .def_readwrite("vertical_aperture", &SonarModel::verticalAperture, "Full vertical aperture, degrees.")
                .def_readwrite("vertical_samples", &SonarModel::verticalSamples, "Rays per beam through the aperture.")
                .def_readwrite("max_range", &SonarModel::maxRange)
                .def_readwrite("min_range", &SonarModel::minRange, "Blind zone: closer surfaces are traced through.")
                .def_readwrite("range_bins", &SonarModel::rangeBins)
                .def_readwrite("attenuation", &SonarModel::attenuation,
                               "One-way amplitude attenuation, 1/m, applied over the two-way path.")
                .def_readwrite("incidence_floor", &SonarModel::incidenceFloor,
                               "strength *= floor + (1 - floor) * |n.d|; 0 = Lambertian.")
                .def_property_readonly("ray_count", &SonarModel::rayCount)
                .def_property_readonly("bin_width", &SonarModel::binWidth)
                .def_static("wide130", &SonarModel::Wide130, "130 x 20 deg, 256 beams, 20 m (the default).")
                .def_static("oculus_m750d", &SonarModel::OculusM750d, "Blueprint Oculus M750d, 750 kHz: 130 x 20 deg, 512 beams, 120 m.")
                .def_static("blueview_m900", &SonarModel::BlueViewM900, "Teledyne BlueView M900-130: 130 x 20 deg, 768 beams, 100 m.")
                .def_static("gemini_720is", &SonarModel::Gemini720is, "Tritech Gemini 720is: 120 x 20 deg, 512 beams, 120 m.")
                .def("__repr__", [](const SonarModel& s) {
                    std::ostringstream o;
                    o << "SonarModel(fov=" << s.horizontalFov << ", beams=" << s.beams
                      << ", aperture=" << s.verticalAperture << ", samples=" << s.verticalSamples
                      << ", max_range=" << s.maxRange << ", bins=" << s.rangeBins << ")";
                    return o.str();
                });

        py::class_<SonarImage>(m, "SonarImage",
                               "One sonar frame: echo strength in [0, 1] per (beam, range bin).")
                .def(py::init<>())
                .def_readonly("beams", &SonarImage::beams)
                .def_readonly("bins", &SonarImage::bins)
                .def_readonly("max_range", &SonarImage::maxRange)
                .def_readonly("time", &SonarImage::time, "Sim time the rays were fired at.")
                .def_property_readonly("intensity", [](const SonarImage& img) {
                    // A (beams, bins) float32 copy: the image is rewritten by the
                    // next scan, and a view into it would change under the caller.
                    py::array_t<float> a({static_cast<py::ssize_t>(img.beams), static_cast<py::ssize_t>(img.bins)});
                    if (!img.intensity.empty())
                        std::memcpy(a.mutable_data(), img.intensity.data(), img.intensity.size() * sizeof(float));
                    return a;
                }, "Echo strength as a (beams, bins) float32 array, beam 0 = left-most.")
                .def("range_of_bin", &SonarImage::rangeOfBin, py::arg("bin"));

        py::class_<SonarReflectivity>(m, "SonarReflectivity",
                                      "Echo strength per target, keyed on the stable instance id "
                                      "(renderer.set_object_instance_id). Unlisted surfaces echo at "
                                      "default_value; volume-scatter returns at volume.")
                .def(py::init<>())
                .def_readwrite("default_value", &SonarReflectivity::defaultValue)
                .def_readwrite("volume", &SonarReflectivity::volume)
                .def("set", &SonarReflectivity::set, py::arg("instance_id"), py::arg("reflectivity"))
                .def("get", [](const SonarReflectivity& r, std::int32_t id) {
                    const auto it = r.byInstance.find(id);
                    return it == r.byInstance.end() ? r.defaultValue : it->second;
                }, py::arg("instance_id"))
                .def("clear", [](SonarReflectivity& r) { r.byInstance.clear(); });

        m.def("sonar_ray_directions", [](const SonarModel& model) {
            const auto dirs = sonarRayDirections(model);
            py::array_t<float> a({static_cast<py::ssize_t>(dirs.size()), static_cast<py::ssize_t>(3)});
            auto* p = a.mutable_data();
            for (std::size_t i = 0; i < dirs.size(); ++i) {
                p[3 * i + 0] = dirs[i].x;
                p[3 * i + 1] = dirs[i].y;
                p[3 * i + 2] = dirs[i].z;
            }
            return a;
        }, py::arg("model"),
           "Sensor-local unit ray directions, (beams * vertical_samples, 3), beam-major. The table "
           "a SonarSensor traces; also what to feed renderer.scan_lidar for a hand-rolled fan.");

        // ---- Canvas ----------------------------------------------------------
        // A GLFW window (or a hidden surface when headless=True). Construction is
        // exposed as keyword arguments rather than the fluent Parameters builder.
        py::class_<Canvas>(m, "Canvas")
                .def(py::init([](const std::string& title, int width, int height, int antialiasing,
                                 bool vsync, bool resizable, bool headless, bool fullscreen,
                                 std::optional<std::pair<int, int>> position) {
                    Canvas::Parameters p;
                    p.title(title);
                    if (width > 0 && height > 0) p.size(width, height);
                    if (position) p.position(position->first, position->second);
                    p.antialiasing(antialiasing).vsync(vsync).resizable(resizable).headless(headless).fullscreen(fullscreen);
                    return std::make_unique<Canvas>(p);
                }),
                     py::arg("title") = "threepp", py::arg("width") = -1, py::arg("height") = -1,
                     py::arg("antialiasing") = 4, py::arg("vsync") = true,
                     py::arg("resizable") = true, py::arg("headless") = false,
                     py::arg("fullscreen") = false, py::arg("position") = py::none(),
                     "A window (or a hidden surface when headless=True). width/height default to half "
                     "the primary monitor. fullscreen=True gives BORDERLESS windowed fullscreen: an "
                     "undecorated, non-resizable window covering the primary monitor, which ignores "
                     "width/height and resizable. It never changes the display mode, so alt-tab behaves "
                     "like any other window. headless=True wins over it (there is no window to show). "
                     "position=(x, y) places the window in screen coordinates instead of letting the OS "
                     "choose (ignored with fullscreen=True; no-op on Wayland).")
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
                        case GraphicsAPI::Vulkan: return "Vulkan";
                        default: return "OpenGL";
                    }
                })
                .def("set_size", [](Canvas& c, int w, int h) { c.setSize({w, h}); }, py::arg("width"), py::arg("height"))
                .def("set_position", [](Canvas& c, int x, int y) { c.setPosition({x, y}); }, py::arg("x"), py::arg("y"),
                     "Move the window's outer top-left corner (title bar included) to (x, y) in screen "
                     "coordinates — (0, 0) keeps the whole window on screen. No-op on Wayland.")
                .def("position", [](const Canvas& c) {
                    auto p = c.position();
                    return py::make_tuple(p.first, p.second);
                })
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
                    std::vector<unsigned char> buf;
                    {
                        // glReadPixels drains the GL pipeline — GIL released
                        // for the stall (numpy below is built with it held).
                        py::gil_scoped_release release;
                        buf = r.readRGBPixels();
                    }
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
                .def("save_frame", [](GLRenderer& r, const std::string& path) { r.writeFramebuffer(path); }, py::arg("path"))
                // GL buffer object id (GLuint) backing a geometry attribute, or None
                // until the first render uploads it. Exists for zero-copy CUDA/GL
                // interop: an external GPU simulation (e.g. NVIDIA Warp) registers
                // the VBO and writes vertex data in place. The attribute's CPU-side
                // array goes stale by design — don't mix with update_attribute on
                // the same attribute.
                .def("gl_buffer_id", [](GLRenderer& r, BufferGeometry& g, const std::string& name) -> py::object {
                    auto* attr = g.getAttribute(name);
                    if (!attr) return py::none();
                    try {
                        const auto id = r.getGlBufferId(*attr);
                        if (!id || *id == 0) return py::none();
                        return py::int_(*id);
                    } catch (const std::out_of_range&) {
                        return py::none();
                    }
                }, py::arg("geometry"), py::arg("attribute"));

        // ---- DepthSensor (helpers/DepthSensor.hpp) ---------------------------
        // A GPU depth-render sensor: renders the scene from its own viewpoint, linearizes
        // depth, reads it back, and reprojects to a world-space point cloud (optionally with
        // per-point sRGB color). It is an Object3D — aim it with position/rotation/quaternion
        // (NOT look_at: beams go down local -Z but the non-camera look_at convention points
        // +Z at the target, so look_at aims it exactly backwards — see the class docstring),
        // then scan() with a GLRenderer. scan() refreshes the sensor's world matrix first,
        // so it works whether or not the sensor was added to the scene.
        auto pts_to_numpy = [](const std::vector<Vector3>& cloud) {
            py::array_t<float> pts({static_cast<py::ssize_t>(cloud.size()), static_cast<py::ssize_t>(3)});
            auto* d = pts.mutable_data();
            for (size_t i = 0; i < cloud.size(); ++i) {
                d[i * 3 + 0] = cloud[i].x; d[i * 3 + 1] = cloud[i].y; d[i * 3 + 2] = cloud[i].z;
            }
            return pts;
        };
        // The cloud a scan_begin() fires into and a later scan_collect() takes
        // delivery of. It has to be ONE vector across the two calls: on a
        // raster backend scan_begin does the whole scan and fills it there and
        // then, and scan_collect merely hands it over — a fresh vector per call
        // would silently return nothing on GL. Python has nowhere to park it
        // (DepthSensor is bound with an existing shared_ptr holder, so
        // py::dynamic_attr is not available), so the binding keeps it here,
        // keyed by sensor. Bounded: at most one vector per sensor with an
        // uncollected scan_begin outstanding, since scan_collect erases the
        // entry — a sensor that is always collected holds nothing between
        // frames, and one that is fired and abandoned holds exactly one cloud
        // until its next collect. The mutex is belt-and-braces (the GIL already
        // serializes these calls) and is never held across a call into Python.
        struct PendingClouds {
            std::mutex mtx;
            std::unordered_map<const DepthSensor*, std::vector<Vector3>> clouds;
        };
        static PendingClouds pending;
        py::class_<DepthSensor, Object3D, Sensor, std::shared_ptr<DepthSensor>>(
                m, "DepthSensor",
                "GPU depth sensor: scans the scene from its own pose and returns a world-space "
                "point cloud.\n\n"
                "Beam convention: beams are cast down the sensor's LOCAL -Z axis (camera "
                "convention), and look_at() honours it -- look_at(target) turns the beams "
                "toward the target, exactly as it would for a camera. (Older releases gave "
                "the sensor the plain-Object3D convention instead, turning local +Z toward "
                "the target -- beams aimed exactly backwards, every scan an empty (0, 3) "
                "cloud -- and callers compensated by aiming at the mirror point "
                "2*position - target. Such call sites now aim backwards: pass the target "
                "itself.)\n")
                .def(py::init([](float fov_y, unsigned int width, unsigned int height, float near, float far) {
                    return std::make_shared<DepthSensor>(fov_y, width, height, near, far);
                }), py::arg("fov_y"), py::arg("width"), py::arg("height"),
                    py::arg("near") = 0.1f, py::arg("far") = 100.f,
                    "Depth sensor with a vertical FOV (deg), output resolution, and the near/far "
                    "RANGE bounds (m) it reports in -- a blind sphere of radius near out to a max "
                    "range of far, inclusive at both ends and identical on GL and Vulkan. These are "
                    "ranges, not view-space clip planes: an off-axis surface is judged by its "
                    "distance, not by its depth along the view axis.")
                // `range_noise` stays a float — the sigma is the knob every
                // caller actually turns (sliders, episode randomization) — with
                // the full seeded model beside it as `noise`.
                .def_property(
                        "range_noise",
                        [](const DepthSensor& s) { return s.rangeNoise.stddev; },
                        [](DepthSensor& s, float v) { s.rangeNoise.stddev = v; },
                        "Gaussian range-noise std-dev in metres (0 = perfect sensor). "
                        "Shorthand for noise.stddev.")
                .def_readwrite("noise", &DepthSensor::rangeNoise,
                               "The full RangeNoiseModel (stddev, stddev_per_metre, bias, seed). "
                               "Set `seed` to make a captured dataset replayable; the stream re-seeds "
                               "on the next scan when the seed changes.")
                .def_property_readonly("last_scan_time", &DepthSensor::lastScanTime,
                                       "Sim time (s) stamped on the most recent scan — the timestamp to "
                                       "record alongside the cloud. Register the sensor with a PhysxWorld "
                                       "or drive `sim_time` yourself; see threepp.Sensor.")
                .def_property_readonly("scan_due", &DepthSensor::scanDue,
                                       "True when the rate gate says a scan is due (always true unless "
                                       "rate_hz is set and the sensor is registered with a PhysxWorld).")
                .def("reset_noise", &DepthSensor::resetNoise,
                     "Re-seed the noise stream from noise.seed and clear last_scan_time — call between "
                     "episodes so two runs with the same seed produce the same clouds.")
                .def_property_readonly("width", &DepthSensor::width)
                .def_property_readonly("height", &DepthSensor::height)
                .def_property_readonly("fov", &DepthSensor::fov)
                .def_property_readonly("near", &DepthSensor::near)
                .def_property_readonly("far", &DepthSensor::far)
                .def("scan", [pts_to_numpy](DepthSensor& self, const py::object& renderer, Scene& scene) {
                    self.updateWorldMatrix(true, true);            // sync sensor + child camera pose
                    Renderer& r = as_renderer(renderer);           // resolve the py handle with the GIL held
                    std::vector<Vector3> cloud;
                    {
                        py::gil_scoped_release release;// blocks on the depth render + readback
                        self.scan(r, scene, cloud);
                    }
                    return pts_to_numpy(cloud);
                }, py::arg("renderer"), py::arg("scene"),
                   "Depth scan -> (N,3) float32 world-space hit points (N = points that hit within far). "
                   "Works with a GLRenderer (raster depth) or a VulkanRenderer (path-traced through the "
                   "renderer's acceleration structure -- render() the scene at least once first). "
                   "Beams go down the sensor's local -Z; aim them with look_at(target) "
                   "(see the class docstring).")
                // ---- pipelined scan: fire on one frame, collect on a later one ----
                .def("scan_begin", [](DepthSensor& self, const py::object& renderer, Scene& scene) {
                    self.updateWorldMatrix(true, true);            // sync sensor + child camera pose
                    Renderer& r = as_renderer(renderer);
                    // The raster backend does the ENTIRE scan inside scanBegin,
                    // so the GIL is released for it — into a LOCAL cloud first:
                    // pending.mtx must only ever be held with the GIL held, or
                    // a second thread blocking on the mutex while holding the
                    // GIL deadlocks against our re-acquire.
                    std::vector<Vector3> cloud;
                    bool immediate;
                    {
                        py::gil_scoped_release release;
                        immediate = self.scanBegin(r, scene, cloud);
                    }
                    std::lock_guard<std::mutex> lock(pending.mtx);
                    pending.clouds[&self] = std::move(cloud);
                    return immediate;
                }, py::arg("renderer"), py::arg("scene"),
                   "Fire a scan without waiting for it. Call it AFTER render() on the frame you want "
                   "sampled: the beams snapshot the sensor's pose (and stamp last_scan_time) here, not "
                   "at scan_collect. Take delivery with scan_collect on a later frame — on Vulkan a "
                   "collect with at least one intervening render() is essentially free, whereas scan() "
                   "blocks on the readback and so pays for every frame already queued behind the fence.\n\n"
                   "    if sensor.scan_due and not sensor.scan_pending:\n"
                   "        sensor.scan_begin(renderer, scene)\n"
                   "    if sensor.scan_ready(renderer):\n"
                   "        pts = sensor.scan_collect(renderer)\n\n"
                   "Returns True when the cloud is ALREADY complete — the raster (GLRenderer) path has "
                   "nothing to pipeline, so it does the whole scan here. On Vulkan it returns False and "
                   "the cloud arrives at a later scan_collect. Either way scan_collect is what hands the "
                   "points over, so the loop above is correct on both backends; only the frame the cloud "
                   "lands on differs.")
                .def("scan_ready", [](const DepthSensor& self, const py::object& renderer) {
                    return self.scanReady(as_renderer(renderer));
                }, py::arg("renderer"),
                   "True when a fired scan can be collected without waiting. A poll, never a wait. "
                   "False when no scan is outstanding. Raster: True as soon as scan_begin has run.")
                .def_property_readonly("scan_pending", &DepthSensor::scanPending,
                                       "True between scan_begin and its scan_collect. Firing again while "
                                       "one is outstanding throws the earlier scan away, so a driver on a "
                                       "rate gate should skip a due scan while this is True.")
                .def("scan_collect", [pts_to_numpy](DepthSensor& self, const py::object& renderer) {
                    Renderer& r = as_renderer(renderer);
                    // Take the entry out under the lock (held with the GIL, see
                    // scan_begin), then collect with the GIL released. No entry
                    // means no scan_begin was ever fired on this sensor;
                    // scanCollect then reports nothing outstanding and the
                    // local cloud stays empty.
                    std::vector<Vector3> cloud;
                    {
                        std::lock_guard<std::mutex> lock(pending.mtx);
                        const auto it = pending.clouds.find(&self);
                        if (it != pending.clouds.end()) {
                            cloud = std::move(it->second);
                            pending.clouds.erase(it);
                        }
                    }
                    bool delivered;
                    {
                        py::gil_scoped_release release;// fence wait + readback
                        delivered = self.scanCollect(r, cloud);
                    }
                    return delivered ? pts_to_numpy(cloud) : pts_to_numpy(std::vector<Vector3>{});
                }, py::arg("renderer"),
                   "Take delivery of a scan_begin -> (N,3) float32 world-space hit points, exactly like "
                   "scan(). Returns an EMPTY (0,3) array when there was nothing to deliver: no scan "
                   "outstanding, or a scan_begin the backend refused because too many traces were "
                   "already in flight. Check scan_ready first (or accept the empty array as 'not yet').")
                .def("scan_rgbd", [pts_to_numpy](DepthSensor& self, const py::object& renderer, Scene& scene) {
                    self.updateWorldMatrix(true, true);
                    Renderer& r = as_renderer(renderer);
                    std::vector<Vector3> cloud;
                    std::vector<Color> colors;
                    {
                        py::gil_scoped_release release;// blocks on depth + colour readbacks
                        self.scan(r, scene, cloud, colors);
                    }
                    py::array_t<float> col({static_cast<py::ssize_t>(colors.size()), static_cast<py::ssize_t>(3)});
                    auto* c = col.mutable_data();
                    for (size_t i = 0; i < colors.size(); ++i) {
                        c[i * 3 + 0] = colors[i].r; c[i * 3 + 1] = colors[i].g; c[i * 3 + 2] = colors[i].b;
                    }
                    return py::make_tuple(pts_to_numpy(cloud), col);
                }, py::arg("renderer"), py::arg("scene"),
                   "RGB-D scan -> (points (N,3) float32 world-space, colors (N,3) float32 in [0,1]). On GL the "
                   "colors are sampled sRGB; on Vulkan they are LIDAR intensity as greyscale.");

        // ---- TransformControls -----------------------------------------------
        py::class_<TransformControls, Object3D, std::shared_ptr<TransformControls>>(m, "TransformControls")
                .def(py::init([](Camera& camera, Canvas& canvas) {
                         return std::make_shared<TransformControls>(camera, canvas);
                     }),
                     py::arg("camera"), py::arg("canvas"),
                     py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
                .def_readwrite("enabled", &TransformControls::enabled)
                .def_readwrite("show_x", &TransformControls::showX)
                .def_readwrite("show_y", &TransformControls::showY)
                .def_readwrite("show_z", &TransformControls::showZ)
                .def("set_mode", &TransformControls::setMode, py::arg("mode"),
                     "Mode: 'translate' | 'rotate' | 'scale'")
                .def("set_space", &TransformControls::setSpace, py::arg("space"),
                     "Space: 'world' | 'local'")
                .def("get_space", &TransformControls::getSpace)
                .def("set_size", &TransformControls::setSize, py::arg("size"))
                .def("set_translation_snap", &TransformControls::setTranslationSnap, py::arg("snap"))
                .def("set_rotation_snap", &TransformControls::setRotationSnap, py::arg("snap"))
                .def("set_scale_snap", &TransformControls::setScaleSnap, py::arg("snap"))
                .def_property_readonly("dragging", &TransformControls::isDragging,
                     "True while the user is actively dragging the gizmo.")
                .def("attach", [](TransformControls& self, const py::handle& obj) -> TransformControls& {
                         return self.attach(*as_object3d(obj));
                     }, py::arg("object"), py::return_value_policy::reference,
                     "Attach the gizmo to an Object3D. Add the TransformControls itself to the scene.")
                .def("detach", &TransformControls::detach, py::return_value_policy::reference);

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
