// Cameras. Both concrete cameras derive from Camera, which derives from
// Object3D (so position/lookAt come from the core bindings). `near`/`far` are
// exposed under their three.js names in addition to the threepp field names.
#include "bindings.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"

using namespace threepp;

namespace threepp_py {

    void init_cameras(py::module_& m) {

        // ---- Camera (abstract base) -----------------------------------------
        py::class_<Camera, Object3D, std::shared_ptr<Camera>>(m, "Camera")
                .def_readwrite("zoom", &Camera::zoom)
                .def_readwrite("near_plane", &Camera::nearPlane)
                .def_readwrite("far_plane", &Camera::farPlane)
                .def_property("near", [](const Camera& c) { return c.nearPlane; }, [](Camera& c, float v) { c.nearPlane = v; })
                .def_property("far", [](const Camera& c) { return c.farPlane; }, [](Camera& c, float v) { c.farPlane = v; })
                // Intrinsics / extrinsics, returned as Matrix4 (use .to_numpy() for a
                // (4,4) array). The getters refresh the matrices first so a freshly
                // moved/reconfigured camera reads correctly without an explicit update:
                //   projection_matrix      — clip-from-view (the projection; pinhole K)
                //   matrix_world_inverse   — view matrix (world -> camera)
                //   matrix_world (Object3D)— camera pose (camera -> world, extrinsics)
                .def_property_readonly("projection_matrix", [](Camera& c) {
                    c.updateProjectionMatrix();
                    return c.projectionMatrix;
                })
                .def_property_readonly("projection_matrix_inverse", [](Camera& c) {
                    c.updateProjectionMatrix();
                    return c.projectionMatrixInverse;
                })
                .def_property_readonly("matrix_world_inverse", [](Camera& c) {
                    c.updateMatrixWorld(true);
                    return c.matrixWorldInverse;
                })
                .def("update_projection_matrix", &Camera::updateProjectionMatrix);

        // ---- PerspectiveCamera ----------------------------------------------
        py::class_<PerspectiveCamera, Camera, std::shared_ptr<PerspectiveCamera>>(m, "PerspectiveCamera")
                .def(py::init([](float fov, float aspect, float near, float far) {
                    return PerspectiveCamera::create(fov, aspect, near, far);
                }),
                     py::arg("fov") = 60.f, py::arg("aspect") = 1.f,
                     py::arg("near") = 0.1f, py::arg("far") = 2000.f)
                .def_readwrite("fov", &PerspectiveCamera::fov)
                .def_readwrite("aspect", &PerspectiveCamera::aspect)
                .def_readwrite("focus", &PerspectiveCamera::focus)
                .def_readwrite("film_gauge", &PerspectiveCamera::filmGauge)
                .def_readwrite("film_offset", &PerspectiveCamera::filmOffset)
                .def("set_focal_length", &PerspectiveCamera::setFocalLength, py::arg("focal_length"))
                .def("get_focal_length", &PerspectiveCamera::getFocalLength)
                .def("update_projection_matrix", &PerspectiveCamera::updateProjectionMatrix);

        // ---- OrthographicCamera ---------------------------------------------
        py::class_<OrthographicCamera, Camera, std::shared_ptr<OrthographicCamera>>(m, "OrthographicCamera")
                .def(py::init([](float left, float right, float top, float bottom, float near, float far) {
                    return OrthographicCamera::create(left, right, top, bottom, near, far);
                }),
                     py::arg("left") = -1.f, py::arg("right") = 1.f,
                     py::arg("top") = 1.f, py::arg("bottom") = -1.f,
                     py::arg("near") = 0.1f, py::arg("far") = 2000.f)
                .def_readwrite("left", &OrthographicCamera::left)
                .def_readwrite("right", &OrthographicCamera::right)
                .def_readwrite("top", &OrthographicCamera::top)
                .def_readwrite("bottom", &OrthographicCamera::bottom)
                .def("update_projection_matrix", &OrthographicCamera::updateProjectionMatrix);
    }

}// namespace threepp_py
