// threepp.editor, camera half — the one sensor a script can read PIXELS from.
//
// camera_from_object(obj) hands back the colour camera the play session is
// running on that object, and `image` is its most recent frame. That closes the
// gap this file exists for: everything else a script could reach was a number
// or a pose, so a robot-mounted camera — the sensor most perception work
// actually starts from — could only be simulated by standing up a SECOND
// threepp process with its own window, its own copy of the scene and its own
// idea of what time it is. Now it is a checkbox on the link and four lines of
// script.
//
// Separate from bind_editor_sensors.cpp, and deliberately NOT gated on the
// PhysX SDK. That file is gated for what it means (no PhysX, no bodies, so no
// body sensors to read), and the same reasoning cuts the other way here: a
// picture is a renderer product start to finish. A build without the SDK still
// authors, plays, aims and reads a camera, so the name has to exist there.
//
// The contract is the sensor handles' contract, and is stated in full in
// bind_editor_sensors.cpp: a handle lives only for the Play that made it,
// raises rather than reading freed memory afterwards, and resolves by walking
// UP the scene graph so a script on a child of the instrumented link still
// finds the camera bolted to it.

#include "bindings.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"
#include "threepp/helpers/CameraSensor.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

using namespace threepp;

namespace {

    namespace py = pybind11;

    using Session = editor::SensorPlaySession;
    using Entry = Session::Entry;

    // The same weak token the physics and sensor handles carry. Duplicated
    // rather than shared for the same reason they duplicate it between
    // themselves: each lives in its own translation unit's anonymous namespace.
    struct Lifetime {

        std::weak_ptr<const void> token;

        [[nodiscard]] bool alive() const { return !token.expired(); }

        void require() const {

            if (!alive()) {
                throw std::runtime_error("this camera belongs to a play session that has "
                                         "stopped - ask for it again after Play starts");
            }
        }
    };


    class CameraHandle {

    public:
        CameraHandle(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime)
            : object_(std::move(object)), entry_(entry), lifetime_(std::move(lifetime)),
              label_(entry->label) {}

        [[nodiscard]] bool valid() const { return lifetime_.alive(); }

        [[nodiscard]] std::shared_ptr<Object3D> object() const { return object_; }

        [[nodiscard]] unsigned int width() const { return sensor().width(); }
        [[nodiscard]] unsigned int height() const { return sensor().height(); }

        // Frames captured since Play, and the sim time of the newest. `frames`
        // is how a script tells a NEW picture from the one it already published
        // — comparing the bytes would be both slower and wrong (a static scene
        // legitimately produces identical frames).
        [[nodiscard]] std::size_t frames() const { return sensor().frames(); }
        [[nodiscard]] double time() const { return sensor().lastCaptureTime(); }

        [[nodiscard]] py::bytes image() const {

            const auto& pixels = sensor().image();
            return {reinterpret_cast<const char*>(pixels.data()), pixels.size()};
        }

        [[nodiscard]] bool save(const std::string& path) const {

            return sensor().writeImage(std::filesystem::path(path));
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<Camera (session stopped)>";
            const auto& cam = sensor();
            return "<Camera '" + label_ + "' " + std::to_string(cam.width()) + "x" +
                   std::to_string(cam.height()) + " frames=" + std::to_string(cam.frames()) + ">";
        }

    private:
        // `entry_` points into the session's own list, cleared by stop(); the
        // lifetime gate is what makes dereferencing it safe. A live entry always
        // has the camera — findSensors only answers with entries that came up.
        [[nodiscard]] const CameraSensor& sensor() const {

            lifetime_.require();
            return *entry_->camera;
        }

        std::shared_ptr<Object3D> object_;
        const Entry* entry_;
        Lifetime lifetime_;
        std::string label_;// repr() has to work after the entry is gone
    };

}// namespace

namespace threepp_py {

    void init_editor_camera(py::module_& m) {

        // bind_editor.cpp made the submodule; this adds the camera to it.
        auto sub = m.attr("editor").cast<py::module_>();

        py::class_<CameraHandle, std::shared_ptr<CameraHandle>>(
                sub, "Camera",
                "The live colour camera authored on an object - what the robot sees.\n\n"
                "Exists only during Play, and only while the session that made it is running. "
                "Read `frames` to tell a new picture from one already handled.")
                .def_property_readonly("object", &CameraHandle::object,
                                       "The scene object this camera was authored on.")
                .def_property_readonly("valid", &CameraHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("width", &CameraHandle::width, "Image width in pixels.")
                .def_property_readonly("height", &CameraHandle::height, "Image height in pixels.")
                .def_property_readonly("frames", &CameraHandle::frames,
                                       "Frames captured since Play began. Compare against the value "
                                       "you last handled to detect a new one; two consecutive frames "
                                       "of a still scene are legitimately identical bytes.")
                .def_property_readonly("time", &CameraHandle::time,
                                       "Sim time the newest frame was captured at (s), on the same "
                                       "clock as every other sensor's measurements.")
                .def_property_readonly(
                        "image", &CameraHandle::image,
                        "The newest frame as bytes: tightly packed RGB8, row-major, TOP-LEFT "
                        "origin (row 0 is the top), length width*height*3. Empty before the "
                        "first capture.\n\n"
                        "A copy, so keeping it is safe - the sensor overwrites its own buffer on "
                        "the next capture. This is the layout a ROS sensor_msgs/Image with "
                        "encoding \"rgb8\" wants, with no conversion.")
                .def("save", &CameraHandle::save, py::arg("path"),
                     "Write the newest frame to a .png/.jpg/.bmp, creating parent directories. "
                     "False when there is no frame yet or the extension is not one of those.")
                .def("__repr__", &CameraHandle::repr);

        sub.def(
                "camera_from_object", [](const py::handle& h) -> py::object {
                    const auto object = as_object3d(h);
                    auto* session = Session::active();
                    if (!object || !session) return py::none();
                    const auto found =
                            session->findSensors(object.get(), editor::SensorConfig::Type::Camera);
                    if (found.empty()) return py::none();
                    const auto* entry = found.front();
                    return py::cast(std::make_shared<CameraHandle>(
                            entry->node->shared_from_this(), entry, Lifetime{session->lifetime()}));
                },
                py::arg("object"),
                "The live colour camera authored on `object`, or None when Play is not running, "
                "no camera is authored here, or the one that is could not come up. The lookup "
                "walks up the scene graph, so a script on a child finds the camera on its link.");
    }

}// namespace threepp_py
